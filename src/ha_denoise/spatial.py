from __future__ import annotations

import torch

from .features import FeatureConfig, istft, stft


def _shift_signal(x: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return x
    y = torch.zeros_like(x)
    if shift > 0:
        y[shift:] = x[:-shift]
    else:
        y[:shift] = x[-shift:]
    return y


def estimate_relative_delay(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    max_lag: int = 8,
    analysis_samples: int | None = None,
) -> int:
    """Estimate signed integer delay between two close-spaced microphones.

    Positive lag means candidate lags reference and should be shifted earlier
    by that many samples before delay-and-sum.
    """
    n = min(reference.numel(), candidate.numel())
    if analysis_samples is not None:
        n = min(n, max(1, int(analysis_samples)))
    ref = reference[:n] - reference[:n].mean()
    cand = candidate[:n] - candidate[:n].mean()
    best_lag = 0
    best_score = None
    eps = ref.new_tensor(1e-8)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = ref[: n - lag]
            b = cand[lag:n]
        else:
            a = ref[-lag:n]
            b = cand[: n + lag]
        if a.numel() <= 0:
            continue
        score = torch.sum(a * b) / torch.sqrt(torch.sum(a.square()) * torch.sum(b.square()) + eps)
        if best_score is None or score > best_score:
            best_score = score
            best_lag = lag
    return int(best_lag)


def delay_and_sum_beamform(
    mix: torch.Tensor,
    max_lag: int = 8,
    analysis_samples: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Return a mono delay-and-sum signal from a [2, samples] waveform."""
    if mix.ndim != 2 or mix.shape[0] != 2:
        raise ValueError("mix must have shape [2, samples]")
    lag = estimate_relative_delay(mix[0], mix[1], max_lag=max_lag, analysis_samples=analysis_samples)
    aligned_1 = _shift_signal(mix[1], -lag)
    return 0.5 * (mix[0] + aligned_1), lag


def _smoothed_coherence(
    spec0: torch.Tensor,
    spec1: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    frames = min(spec0.shape[-1], spec1.shape[-1])
    spec0 = spec0[:, :frames]
    spec1 = spec1[:, :frames]
    p0_avg = torch.zeros(spec0.shape[0], device=spec0.device, dtype=spec0.real.dtype)
    p1_avg = torch.zeros_like(p0_avg)
    cross_avg = torch.zeros(spec0.shape[0], device=spec0.device, dtype=spec0.dtype)
    out = torch.empty(spec0.shape[0], frames, device=spec0.device, dtype=spec0.real.dtype)
    a = float(min(max(alpha, 0.0), 0.999))
    eps = spec0.real.new_tensor(1e-8)
    for t in range(frames):
        p0 = spec0[:, t].abs().square()
        p1 = spec1[:, t].abs().square()
        cross = spec0[:, t] * torch.conj(spec1[:, t])
        p0_avg = a * p0_avg + (1.0 - a) * p0
        p1_avg = a * p1_avg + (1.0 - a) * p1
        cross_avg = a * cross_avg + (1.0 - a) * cross
        out[:, t] = torch.clamp(torch.abs(cross_avg) / torch.sqrt(torch.clamp(p0_avg * p1_avg, min=eps)), 0.0, 1.0)
    return out


def coherence_weighted_beamform(
    mix: torch.Tensor,
    cfg: FeatureConfig,
    max_lag: int = 8,
    analysis_samples: int | None = None,
    floor: float = 0.22,
    alpha: float = 0.88,
    gamma: float = 0.70,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Delay-align mics, then suppress bins with low inter-mic coherence.

    This is a light two-mic post-filter in the Zelinski/Wiener family. It is not
    a full MVDR solver, but it uses the information delay-and-sum ignores:
    target-direction speech remains coherent after alignment, while diffuse or
    off-axis residual noise tends to have lower short-term coherence.
    """
    if mix.ndim != 2 or mix.shape[0] != 2:
        raise ValueError("mix must have shape [2, samples]")
    lag = estimate_relative_delay(mix[0], mix[1], max_lag=max_lag, analysis_samples=analysis_samples)
    aligned_1 = _shift_signal(mix[1], -lag)
    summed = 0.5 * (mix[0] + aligned_1)
    spec0 = stft(mix[0], cfg)
    spec1 = stft(aligned_1, cfg)
    summed_spec = stft(summed, cfg)
    frames = min(spec0.shape[-1], spec1.shape[-1], summed_spec.shape[-1])
    coherence = _smoothed_coherence(spec0[:, :frames], spec1[:, :frames], alpha=alpha)
    floor_t = summed_spec.real.new_tensor(min(max(float(floor), 0.0), 1.0))
    gain = floor_t + (1.0 - floor_t) * torch.pow(torch.clamp(coherence, 0.0, 1.0), float(max(gamma, 1e-4)))
    enhanced_spec = summed_spec[:, :frames] * torch.clamp(gain, floor_t, 1.0)
    enhanced = istft(enhanced_spec, length=mix.shape[-1], cfg=cfg)
    return enhanced, {
        "mode": "coherence_mwf",
        "lag": lag,
        "mean_coherence": float(coherence.mean().detach().cpu()),
        "mean_spatial_gain": float(gain.mean().detach().cpu()),
    }


def apply_spatial_frontend(
    mix: torch.Tensor,
    cfg: FeatureConfig,
    max_lag: int = 8,
    analysis_samples: int | None = None,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    mode = getattr(cfg, "spatial_frontend", "delay_sum")
    if mode in {"delay_sum", "delay-and-sum", "beamform"}:
        enhanced, lag = delay_and_sum_beamform(mix, max_lag=max_lag, analysis_samples=analysis_samples)
        return enhanced, {"mode": "delay_sum", "lag": lag}
    if mode in {"coherence", "coherence_mwf", "zelinski"}:
        return coherence_weighted_beamform(mix, cfg, max_lag=max_lag, analysis_samples=analysis_samples)
    raise ValueError(f"unsupported spatial_frontend: {mode}")
