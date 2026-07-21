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
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.bands = bands
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.mask_target = mask_target
        self.spatial_features = spatial_features

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
    return feats, masks, valid
