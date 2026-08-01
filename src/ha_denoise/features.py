from __future__ import annotations

import torch
import torch.nn.functional as F


class FeatureConfig:
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 256,
        hop_length: int = 64,
        bands: int = 32,
        min_gain: float = 0.08,
        max_gain: float = 1.0,
        mask_target: str = "magnitude",
        spatial_features: bool = False,
        spatial_frontend: str = "delay_sum",
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.bands = bands
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.mask_target = mask_target
        self.spatial_features = spatial_features
        self.spatial_frontend = spatial_frontend

    @property
    def feature_dim(self) -> int:
        return self.bands * (6 if self.spatial_features else 3)


def feature_config_from_dict(config: dict) -> FeatureConfig:
    bands = int(config["bands"])
    feature_dim = int(config.get("feature_dim", bands * 3))
    spatial_features = bool(config.get("spatial_features", feature_dim == bands * 6))
    return FeatureConfig(
        int(config["sample_rate"]),
        int(config["n_fft"]),
        int(config["hop_length"]),
        bands,
        min_gain=float(config.get("min_gain", 0.08)),
        max_gain=float(config.get("max_gain", 1.0)),
        mask_target=str(config.get("mask_target", "magnitude")),
        spatial_features=spatial_features,
        spatial_frontend=str(config.get("spatial_frontend", "delay_sum")),
    )


def make_band_matrix(n_fft: int = 256, bands: int = 32, sample_rate: int = 16_000) -> torch.Tensor:
    freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    # Mel-like spacing keeps more resolution below 4 kHz.
    mel = 2595.0 * torch.log10(1.0 + freqs / 700.0)
    edges = torch.linspace(float(mel[0]), float(mel[-1]), bands + 2)
    centers = edges[1:-1]
    width = edges[2:] - edges[:-2]
    weights = torch.clamp(1.0 - torch.abs(mel[:, None] - centers[None, :]) / (width[None, :] / 2.0), min=0.0)
    weights[0, 0] = 1.0
    weights[-1, -1] = 1.0
    weights = weights / torch.clamp(weights.sum(dim=0, keepdim=True), min=1e-8)
    return weights.float()


def stft(wav: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    window = torch.hann_window(cfg.n_fft, device=wav.device, dtype=wav.dtype)
    return torch.stft(
        wav,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.n_fft,
        window=window,
        center=True,
        return_complex=True,
    )


def istft(spec: torch.Tensor, length: int, cfg: FeatureConfig) -> torch.Tensor:
    window = torch.hann_window(cfg.n_fft, device=spec.device, dtype=spec.real.dtype)
    return torch.istft(
        spec,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.n_fft,
        window=window,
        center=True,
        length=length,
    )


def _band_power(spec: torch.Tensor, band_matrix: torch.Tensor) -> torch.Tensor:
    power = spec.abs().square().transpose(-2, -1)
    return power @ band_matrix.to(power.device, power.dtype)


def extract_features(mix: torch.Tensor, cfg: FeatureConfig, band_matrix: torch.Tensor | None = None) -> torch.Tensor:
    """Return features with shape [T, feature_dim] from a [2, N] waveform."""
    if mix.ndim != 2 or mix.shape[0] != 2:
        raise ValueError("mix must have shape [2, samples]")
    band_matrix = band_matrix if band_matrix is not None else make_band_matrix(cfg.n_fft, cfg.bands, cfg.sample_rate)
    spec0 = stft(mix[0], cfg)
    spec1 = stft(mix[1], cfg)
    p0 = _band_power(spec0, band_matrix)
    p1 = _band_power(spec1, band_matrix)
    log0 = torch.log(torch.clamp(p0, min=1e-8))
    log1 = torch.log(torch.clamp(p1, min=1e-8))
    ild = torch.log(torch.clamp(p0, min=1e-8) / torch.clamp(p1, min=1e-8))
    features = [log0, log1, ild]
    if cfg.spatial_features:
        cross = (spec0 * torch.conj(spec1)).transpose(-2, -1) @ band_matrix.to(spec0.device, spec0.real.dtype).to(torch.complex64)
        cross_real = cross.real
        cross_imag = cross.imag
        cross_abs = torch.clamp(torch.abs(cross), min=1e-8)
        ipd_cos = cross_real / cross_abs
        ipd_sin = cross_imag / cross_abs
        coherence = cross_abs / torch.sqrt(torch.clamp(p0 * p1, min=1e-8))
        features.extend([ipd_cos, ipd_sin, torch.clamp(coherence, 0.0, 1.0)])
    feat = torch.cat(features, dim=-1)
    return torch.clamp(feat, -20.0, 20.0)


def target_band_mask(
    mix_ref: torch.Tensor,
    clean_ref: torch.Tensor,
    cfg: FeatureConfig,
    band_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    band_matrix = band_matrix if band_matrix is not None else make_band_matrix(cfg.n_fft, cfg.bands, cfg.sample_rate)
    noisy_spec = stft(mix_ref, cfg)
    clean_spec = stft(clean_ref, cfg)
    noisy_power = _band_power(noisy_spec, band_matrix)
    if cfg.mask_target == "phase_sensitive":
        cross = (clean_spec * torch.conj(noisy_spec)).real.transpose(-2, -1)
        projected_clean = cross @ band_matrix.to(cross.device, cross.dtype)
        mask = projected_clean / torch.clamp(noisy_power, min=1e-8)
    elif cfg.mask_target == "magnitude":
        clean_power = _band_power(clean_spec, band_matrix)
        mask = torch.sqrt(torch.clamp(clean_power, min=1e-8) / torch.clamp(noisy_power, min=1e-8))
    else:
        raise ValueError(f"unsupported mask_target: {cfg.mask_target}")
    return torch.clamp(mask, cfg.min_gain, cfg.max_gain)


def bands_to_bins(mask: torch.Tensor, cfg: FeatureConfig, band_matrix: torch.Tensor | None = None) -> torch.Tensor:
    """Map [T, bands] mask to [freq_bins, T]."""
    band_matrix = band_matrix if band_matrix is not None else make_band_matrix(cfg.n_fft, cfg.bands, cfg.sample_rate)
    weights = band_matrix.to(mask.device, mask.dtype)
    bin_mask = mask @ weights.transpose(0, 1)
    return torch.clamp(bin_mask.transpose(0, 1), 0.0, 1.0)


def enhance_with_mask(mix_ref: torch.Tensor, band_mask: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    spec = stft(mix_ref, cfg)
    bin_mask = bands_to_bins(band_mask, cfg)
    # Align in case stft padding produces one extra frame for odd input lengths.
    frames = min(spec.shape[-1], bin_mask.shape[-1])
    enhanced = spec[:, :frames] * bin_mask[:, :frames]
    return istft(enhanced, length=mix_ref.numel(), cfg=cfg)


def mask_guided_post_filter(
    enhanced: torch.Tensor,
    band_mask: torch.Tensor,
    cfg: FeatureConfig,
    strength: float = 0.45,
    floor: float = 0.35,
    speech_threshold: float = 0.58,
    transition_width: float = 0.18,
    noise_alpha: float = 0.92,
) -> torch.Tensor:
    """Causal-ish spectral post-filter guided by the model mask.

    The learned mask already carries a speech-presence estimate. This post-filter
    only adds extra attenuation where the mask is low, so voiced bins are kept
    close to the denoiser output instead of being globally over-suppressed.
    """
    if strength <= 0.0:
        return enhanced
    spec = stft(enhanced, cfg)
    bin_mask = bands_to_bins(band_mask, cfg)
    frames = min(spec.shape[-1], bin_mask.shape[-1])
    if frames <= 0:
        return enhanced
    spec = spec[:, :frames]
    bin_mask = bin_mask[:, :frames]
    power = spec.abs().square()
    noise = power[:, 0].clone()
    out = torch.empty_like(spec)
    alpha = float(min(max(noise_alpha, 0.0), 0.9999))
    strength_t = spec.real.new_tensor(max(strength, 0.0))
    floor_t = spec.real.new_tensor(min(max(floor, 0.0), 1.0))
    width = max(float(transition_width), 1e-6)
    for t in range(frames):
        speech_presence = torch.clamp((bin_mask[:, t] - speech_threshold) / width, 0.0, 1.0)
        noise_update = 1.0 - speech_presence
        update_alpha = alpha + (0.999 - alpha) * speech_presence
        noise = update_alpha * noise + (1.0 - update_alpha) * power[:, t]
        snr = power[:, t] / torch.clamp(noise, min=1e-8)
        wiener = torch.sqrt(snr / torch.clamp(snr + strength_t, min=1e-8))
        noise_gain = floor_t + (1.0 - floor_t) * wiener
        guided_gain = speech_presence + (1.0 - speech_presence) * noise_gain
        out[:, t] = spec[:, t] * torch.clamp(guided_gain, floor_t, 1.0)
        noise = torch.where(noise_update > 0.5, noise, torch.minimum(noise, power[:, t] * 1.5))
    return istft(out, length=enhanced.numel(), cfg=cfg)


def stationary_noise_floor_filter(
    enhanced: torch.Tensor,
    band_mask: torch.Tensor,
    cfg: FeatureConfig,
    strength: float = 0.35,
    floor: float = 0.72,
    speech_threshold: float = 0.62,
    transition_width: float = 0.20,
    noise_percentile: float = 20.0,
    gain_smooth_alpha: float = 0.94,
) -> torch.Tensor:
    """Conservative stationary-noise suppressor for listening demos.

    The filter estimates one stable noise spectrum from a low percentile across
    the whole utterance. That avoids the frame-by-frame noise estimate pumping
    that tends to sound like airflow or breathing.
    """
    if strength <= 0.0:
        return enhanced
    spec = stft(enhanced, cfg)
    bin_mask = bands_to_bins(band_mask, cfg)
    frames = min(spec.shape[-1], bin_mask.shape[-1])
    if frames <= 1:
        return enhanced
    spec = spec[:, :frames]
    bin_mask = bin_mask[:, :frames]
    power = spec.abs().square()
    q = min(max(float(noise_percentile) / 100.0, 0.01), 0.80)
    noise = torch.quantile(power, q, dim=1, keepdim=True).clamp_min(1e-8)
    snr = power / noise
    raw_gain = torch.sqrt(snr / torch.clamp(snr + float(strength), min=1e-8))
    raw_gain = torch.clamp(raw_gain, min=float(floor), max=1.0)
    width = max(float(transition_width), 1e-6)
    speech_presence = torch.clamp((bin_mask - float(speech_threshold)) / width, 0.0, 1.0)
    gain = speech_presence + (1.0 - speech_presence) * raw_gain
    if gain_smooth_alpha > 0.0:
        out_gain = torch.empty_like(gain)
        out_gain[:, 0] = gain[:, 0]
        alpha = min(max(float(gain_smooth_alpha), 0.0), 0.999)
        for t in range(1, frames):
            out_gain[:, t] = alpha * out_gain[:, t - 1] + (1.0 - alpha) * gain[:, t]
        gain = out_gain
    return istft(spec * torch.clamp(gain, min=float(floor), max=1.0), length=enhanced.numel(), cfg=cfg)


def residual_dehiss_filter(
    enhanced: torch.Tensor,
    band_mask: torch.Tensor,
    cfg: FeatureConfig,
    strength: float = 1.15,
    low_floor: float = 0.78,
    high_floor: float = 0.36,
    speech_threshold: float = 0.64,
    transition_width: float = 0.18,
    noise_percentile: float = 18.0,
    high_start_hz: float = 2600.0,
    full_strength_hz: float = 5200.0,
    gain_smooth_alpha: float = 0.96,
) -> torch.Tensor:
    """Suppress steady high-frequency residual hiss without changing the model.

    This is intentionally offline/demo-oriented: it estimates a stable residual
    noise floor from the whole utterance, then applies extra Wiener attenuation
    mainly above the speech formant region. The model mask protects bins that
    look speech-like, which keeps consonants from being flattened as much as a
    plain high-cut filter would.
    """
    if strength <= 0.0:
        return enhanced
    spec = stft(enhanced, cfg)
    bin_mask = bands_to_bins(band_mask, cfg)
    frames = min(spec.shape[-1], bin_mask.shape[-1])
    if frames <= 1:
        return enhanced
    spec = spec[:, :frames]
    bin_mask = bin_mask[:, :frames]
    power = spec.abs().square()

    q = min(max(float(noise_percentile) / 100.0, 0.01), 0.80)
    noise = torch.quantile(power, q, dim=1, keepdim=True).clamp_min(1e-8)
    snr = power / noise

    freqs = torch.linspace(0.0, cfg.sample_rate / 2, cfg.n_fft // 2 + 1, device=spec.device, dtype=spec.real.dtype)
    width_hz = max(float(full_strength_hz) - float(high_start_hz), 1.0)
    high_weight = torch.clamp((freqs - float(high_start_hz)) / width_hz, 0.0, 1.0)[:, None]
    floor = float(low_floor) * (1.0 - high_weight) + float(high_floor) * high_weight
    band_strength = float(strength) * (0.35 + 0.65 * high_weight)

    wiener = torch.sqrt(snr / torch.clamp(snr + band_strength, min=1e-8))
    raw_gain = torch.minimum(torch.maximum(wiener, floor), torch.ones_like(wiener))

    width = max(float(transition_width), 1e-6)
    speech_presence = torch.clamp((bin_mask - float(speech_threshold)) / width, 0.0, 1.0)
    protect = torch.clamp(0.25 + 0.75 * speech_presence, 0.0, 1.0)
    gain = protect + (1.0 - protect) * raw_gain

    if gain.shape[0] > 2:
        gain_t = gain.transpose(0, 1).unsqueeze(1)
        kernel = gain.new_tensor([0.18, 0.64, 0.18]).view(1, 1, 3)
        gain = F.conv1d(F.pad(gain_t, (1, 1), mode="replicate"), kernel).squeeze(1).transpose(0, 1)

    if gain_smooth_alpha > 0.0:
        out_gain = torch.empty_like(gain)
        out_gain[:, 0] = gain[:, 0]
        alpha = min(max(float(gain_smooth_alpha), 0.0), 0.999)
        for t in range(1, frames):
            out_gain[:, t] = alpha * out_gain[:, t - 1] + (1.0 - alpha) * gain[:, t]
        gain = out_gain

    gain = torch.minimum(torch.maximum(gain, floor), torch.ones_like(gain))
    return istft(spec * gain, length=enhanced.numel(), cfg=cfg)


def residual_airflow_filter(
    enhanced: torch.Tensor,
    band_mask: torch.Tensor,
    cfg: FeatureConfig,
    strength: float = 1.55,
    low_floor: float = 0.34,
    mid_floor: float = 0.58,
    high_floor: float = 0.30,
    speech_threshold: float = 0.68,
    transition_width: float = 0.16,
    noise_percentile: float = 24.0,
    low_full_hz: float = 650.0,
    high_start_hz: float = 1800.0,
    high_full_hz: float = 4200.0,
    gain_smooth_alpha: float = 0.985,
) -> torch.Tensor:
    """Suppress stable breath/airflow-like residual noise for listening demos.

    This is intentionally stronger than residual_dehiss_filter. It estimates a
    stable per-bin residual floor from low-percentile energy, then attenuates
    bins that the denoiser mask does not mark as speech. Low rumble and high
    hiss get lower floors than the middle speech band, while temporal smoothing
    avoids fast pumping that sounds like wind.
    """
    if strength <= 0.0:
        return enhanced
    spec = stft(enhanced, cfg)
    bin_mask = bands_to_bins(band_mask, cfg)
    frames = min(spec.shape[-1], bin_mask.shape[-1])
    if frames <= 1:
        return enhanced
    spec = spec[:, :frames]
    bin_mask = bin_mask[:, :frames]
    power = spec.abs().square()

    q = min(max(float(noise_percentile) / 100.0, 0.01), 0.80)
    noise = torch.quantile(power, q, dim=1, keepdim=True).clamp_min(1e-8)
    snr = power / noise

    freqs = torch.linspace(0.0, cfg.sample_rate / 2, cfg.n_fft // 2 + 1, device=spec.device, dtype=spec.real.dtype)
    low_weight = torch.clamp((float(low_full_hz) - freqs) / max(float(low_full_hz), 1.0), 0.0, 1.0)[:, None]
    high_weight = torch.clamp((freqs - float(high_start_hz)) / max(float(high_full_hz) - float(high_start_hz), 1.0), 0.0, 1.0)[:, None]
    mid_weight = torch.clamp(1.0 - torch.maximum(low_weight, high_weight), 0.0, 1.0)
    floor = float(low_floor) * low_weight + float(mid_floor) * mid_weight + float(high_floor) * high_weight
    band_strength = float(strength) * (1.15 * low_weight + 0.70 * mid_weight + 1.00 * high_weight)

    wiener = torch.sqrt(snr / torch.clamp(snr + band_strength, min=1e-8))
    raw_gain = torch.clamp(wiener, min=0.0, max=1.0)

    width = max(float(transition_width), 1e-6)
    speech_presence = torch.clamp((bin_mask - float(speech_threshold)) / width, 0.0, 1.0)
    protect = torch.clamp(0.12 + 0.88 * speech_presence, 0.0, 1.0)
    gain = protect + (1.0 - protect) * raw_gain

    if gain.shape[0] > 4:
        gain_t = gain.transpose(0, 1).unsqueeze(1)
        kernel = gain.new_tensor([0.08, 0.20, 0.44, 0.20, 0.08]).view(1, 1, 5)
        gain = F.conv1d(F.pad(gain_t, (2, 2), mode="replicate"), kernel).squeeze(1).transpose(0, 1)

    if gain_smooth_alpha > 0.0:
        out_gain = torch.empty_like(gain)
        out_gain[:, 0] = gain[:, 0]
        alpha = min(max(float(gain_smooth_alpha), 0.0), 0.999)
        for t in range(1, frames):
            out_gain[:, t] = alpha * out_gain[:, t - 1] + (1.0 - alpha) * gain[:, t]
        gain = out_gain

    gain = torch.minimum(torch.maximum(gain, floor), torch.ones_like(gain))
    return istft(spec * gain, length=enhanced.numel(), cfg=cfg)


def enhance_with_deep_filter(
    mix_ref: torch.Tensor,
    band_mask: torch.Tensor,
    df_coef: torch.Tensor,
    cfg: FeatureConfig,
) -> torch.Tensor:
    """Apply ERB gain plus causal low-bin residual complex filtering.

    band_mask: [T, bands]
    df_coef: [T, df_bins, df_order, 2], residual real/imag coefficients.

    The ERB gain path is the stable base enhancement. Deep-filter coefficients
    add a learned complex residual in low bins, so zero coefficients exactly
    reproduce ordinary mask enhancement.
    """
    spec = stft(mix_ref, cfg)
    bin_mask = bands_to_bins(band_mask, cfg)
    frames = min(spec.shape[-1], bin_mask.shape[-1], df_coef.shape[0])
    spec = spec[:, :frames]
    enhanced = spec * bin_mask[:, :frames]

    df_bins = min(df_coef.shape[1], spec.shape[0])
    df_order = df_coef.shape[2]
    low = enhanced[:df_bins].clone()
    coef = torch.complex(df_coef[:frames, :df_bins, :, 0], df_coef[:frames, :df_bins, :, 1])
    for k in range(df_order):
        if k == 0:
            hist = spec[:df_bins, :frames]
        else:
            hist = torch.cat(
                [
                    torch.zeros(df_bins, k, device=spec.device, dtype=spec.dtype),
                    spec[:df_bins, : frames - k],
                ],
                dim=1,
            )
        low = low + coef[:, :, k].transpose(0, 1) * hist
    enhanced[:df_bins, :frames] = low
    return istft(enhanced, length=mix_ref.numel(), cfg=cfg)


def target_complex_mask(
    mix_ref: torch.Tensor,
    clean_ref: torch.Tensor,
    cfg: FeatureConfig,
    clip: float = 2.0,
) -> torch.Tensor:
    noisy_spec = stft(mix_ref, cfg)
    clean_spec = stft(clean_ref, cfg)
    frames = min(noisy_spec.shape[-1], clean_spec.shape[-1])
    noisy_spec = noisy_spec[:, :frames]
    clean_spec = clean_spec[:, :frames]
    denom = noisy_spec.real.square() + noisy_spec.imag.square() + 1e-8
    real = (clean_spec.real * noisy_spec.real + clean_spec.imag * noisy_spec.imag) / denom
    imag = (clean_spec.imag * noisy_spec.real - clean_spec.real * noisy_spec.imag) / denom
    mask = torch.stack([real, imag], dim=-1).transpose(0, 1)
    return torch.clamp(mask, -float(clip), float(clip))


def enhance_with_complex_mask(
    mix_ref: torch.Tensor,
    complex_mask: torch.Tensor,
    cfg: FeatureConfig,
) -> torch.Tensor:
    spec = stft(mix_ref, cfg)
    frames = min(spec.shape[-1], complex_mask.shape[0])
    bins = min(spec.shape[0], complex_mask.shape[1])
    spec = spec[:bins, :frames]
    mask = torch.complex(complex_mask[:frames, :bins, 0], complex_mask[:frames, :bins, 1]).transpose(0, 1)
    enhanced = spec * mask
    if bins < cfg.n_fft // 2 + 1:
        padded = torch.zeros(cfg.n_fft // 2 + 1, frames, device=spec.device, dtype=spec.dtype)
        padded[:bins] = enhanced
        enhanced = padded
    return istft(enhanced, length=mix_ref.numel(), cfg=cfg)


def enhance_with_mwf_masks(
    mix_pair: torch.Tensor,
    mwf_masks: torch.Tensor,
    cfg: FeatureConfig,
    covariance_alpha: float = 0.96,
    diagonal_loading: float = 0.01,
    min_mask: float = 0.02,
    speech_power: float = 0.90,
    noise_power: float = 1.35,
) -> torch.Tensor:
    """Apply a causal 2x2 MWF using predicted speech/noise masks.

    mix_pair: [2, samples]
    mwf_masks: [frames, freq_bins, 2], where channel 0 is speech probability
      and channel 1 is noise probability.

    The masks estimate speech and noise covariance matrices from the noisy
    two-mic STFT. The actual filtering is a small deterministic DSP block:
    w = (Rss + Rnn)^-1 Rss u, where u selects mic0 as the target reference.
    """
    if mix_pair.ndim != 2 or mix_pair.shape[0] != 2:
        raise ValueError("mix_pair must have shape [2, samples]")
    spec0 = stft(mix_pair[0], cfg)
    spec1 = stft(mix_pair[1], cfg)
    frames = min(spec0.shape[-1], spec1.shape[-1], mwf_masks.shape[0])
    bins = min(spec0.shape[0], spec1.shape[0], mwf_masks.shape[1])
    if frames <= 0 or bins <= 0:
        return mix_pair[0]

    x = torch.stack([spec0[:bins, :frames], spec1[:bins, :frames]], dim=0)
    masks = torch.clamp(mwf_masks[:frames, :bins].to(x.device), min=float(min_mask), max=1.0)
    speech = masks[:, :, 0].transpose(0, 1).pow(float(speech_power))
    noise = masks[:, :, 1].transpose(0, 1).pow(float(noise_power))
    total = torch.clamp(speech + noise, min=1e-6)
    speech = speech / total
    noise = noise / total

    alpha = float(min(max(covariance_alpha, 0.0), 0.999))
    one_minus = 1.0 - alpha
    r_ss = torch.zeros(bins, 2, 2, device=x.device, dtype=x.dtype)
    r_nn = torch.zeros_like(r_ss)
    enhanced = torch.empty(bins, frames, device=x.device, dtype=x.dtype)
    eye = torch.eye(2, device=x.device, dtype=x.dtype).unsqueeze(0)
    u = torch.zeros(bins, 2, 1, device=x.device, dtype=x.dtype)
    u[:, 0, 0] = 1.0

    for t in range(frames):
        xt = x[:, :, t].transpose(0, 1)
        outer = xt[:, :, None] * torch.conj(xt[:, None, :])
        r_ss = alpha * r_ss + one_minus * speech[:, t, None, None] * outer
        r_nn = alpha * r_nn + one_minus * noise[:, t, None, None] * outer
        trace = (r_ss[:, 0, 0].real + r_ss[:, 1, 1].real + r_nn[:, 0, 0].real + r_nn[:, 1, 1].real).clamp_min(1e-8)
        load = (float(diagonal_loading) * trace + 1e-8).to(x.dtype)
        r_yy = r_ss + r_nn + load[:, None, None] * eye
        rhs = torch.matmul(r_ss, u)
        w = torch.linalg.solve(r_yy, rhs).squeeze(-1)
        enhanced[:, t] = torch.sum(torch.conj(w) * xt, dim=-1)

    if bins < cfg.n_fft // 2 + 1:
        padded = torch.zeros(cfg.n_fft // 2 + 1, frames, device=x.device, dtype=x.dtype)
        padded[:bins] = enhanced
        enhanced = padded
    return istft(enhanced, length=mix_pair.shape[-1], cfg=cfg)


def match_loudness(
    reference: torch.Tensor,
    enhanced: torch.Tensor,
    target_ratio: float = 0.95,
    max_gain_db: float = 6.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale enhanced audio toward the reference RMS with a bounded gain."""
    n = min(reference.numel(), enhanced.numel())
    if n <= 0:
        return enhanced, torch.ones((), device=enhanced.device, dtype=enhanced.dtype)
    ref_rms = torch.sqrt(torch.mean(reference[:n].square()) + eps)
    enh_rms = torch.sqrt(torch.mean(enhanced[:n].square()) + eps)
    max_gain = float(10.0 ** (max_gain_db / 20.0))
    gain = torch.clamp((ref_rms * target_ratio) / enh_rms, max=enhanced.new_tensor(max_gain))
    peak = torch.max(torch.abs(enhanced * gain)).clamp_min(eps)
    limiter = torch.clamp(enhanced.new_tensor(0.98) / peak, max=enhanced.new_tensor(1.0))
    gain = gain * limiter
    return enhanced * gain, gain


def rms_ratio(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = min(reference.numel(), candidate.numel())
    if n <= 0:
        return candidate.new_tensor(1.0)
    ref_rms = torch.sqrt(torch.mean(reference[:n].square()) + eps)
    cand_rms = torch.sqrt(torch.mean(candidate[:n].square()) + eps)
    return cand_rms / ref_rms.clamp_min(eps)


def apply_high_snr_bypass(mask: torch.Tensor, threshold: float = 0.97, width: float = 0.02) -> torch.Tensor:
    """Blend masks toward identity when the predicted mask already indicates a clean frame."""
    if mask.ndim != 2:
        raise ValueError("mask must have shape [frames, bands]")
    mean_mask = mask.mean(dim=-1, keepdim=True)
    bypass = torch.clamp((mean_mask - threshold) / max(width, 1e-6), 0.0, 1.0)
    return torch.clamp(mask * (1.0 - bypass) + bypass, 0.0, 1.0)


def pad_sequence_batch(items):
    max_t = max(item[0].shape[0] for item in items)
    feat_dim = items[0][0].shape[1]
    bands = items[0][1].shape[1]
    feats = torch.zeros(len(items), feat_dim, max_t)
    masks = torch.zeros(len(items), bands, max_t)
    valid = torch.zeros(len(items), 1, max_t)
    for i, item in enumerate(items):
        feat, mask = item[:2]
        t = feat.shape[0]
        feats[i, :, :t] = feat.transpose(0, 1)
        masks[i, :, :t] = mask.transpose(0, 1)
        valid[i, :, :t] = 1.0
    if len(items[0]) == 4:
        max_n = max(item[2].numel() for item in items)
        mix_refs = torch.zeros(len(items), max_n)
        clean_refs = torch.zeros(len(items), max_n)
        audio_valid = torch.zeros(len(items), max_n)
        for i, item in enumerate(items):
            n = item[2].numel()
            mix_refs[i, :n] = item[2]
            clean_refs[i, :n] = item[3]
            audio_valid[i, :n] = 1.0
        return feats, masks, valid, mix_refs, clean_refs, audio_valid
    if len(items[0]) == 6:
        max_n = max(item[2].numel() for item in items)
        mix_refs = torch.zeros(len(items), max_n)
        clean_refs = torch.zeros(len(items), max_n)
        audio_valid = torch.zeros(len(items), max_n)
        mix_pairs = torch.zeros(len(items), 2, max_n)
        clean_pairs = torch.zeros(len(items), 2, max_n)
        for i, item in enumerate(items):
            n = item[2].numel()
            mix_refs[i, :n] = item[2]
            clean_refs[i, :n] = item[3]
            audio_valid[i, :n] = 1.0
            mix_pairs[i, :, : item[4].shape[-1]] = item[4]
            clean_pairs[i, :, : item[5].shape[-1]] = item[5]
        return feats, masks, valid, mix_refs, clean_refs, audio_valid, mix_pairs, clean_pairs
    return feats, masks, valid
