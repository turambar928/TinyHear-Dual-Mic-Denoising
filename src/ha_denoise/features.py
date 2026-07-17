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
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.bands = bands
        self.min_gain = min_gain


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
    """Return features with shape [T, 3 * bands] from a [2, N] waveform."""
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
    feat = torch.cat([log0, log1, ild], dim=-1)
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
    clean_power = _band_power(clean_spec, band_matrix)
    mask = torch.sqrt(torch.clamp(clean_power, min=1e-8) / torch.clamp(noisy_power, min=1e-8))
    return torch.clamp(mask, cfg.min_gain, 1.0)


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


def pad_sequence_batch(items: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_t = max(x.shape[0] for x, _ in items)
    feat_dim = items[0][0].shape[1]
    bands = items[0][1].shape[1]
    feats = torch.zeros(len(items), feat_dim, max_t)
    masks = torch.zeros(len(items), bands, max_t)
    valid = torch.zeros(len(items), 1, max_t)
    for i, (feat, mask) in enumerate(items):
        t = feat.shape[0]
        feats[i, :, :t] = feat.transpose(0, 1)
        masks[i, :, :t] = mask.transpose(0, 1)
        valid[i, :, :t] = 1.0
    return feats, masks, valid

