from __future__ import annotations

import torch


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
