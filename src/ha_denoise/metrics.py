from __future__ import annotations

import torch


def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    scale = torch.sum(estimate * target) / (torch.sum(target * target) + eps)
    projection = scale * target
    noise = estimate - projection
    return 10.0 * torch.log10((torch.sum(projection * projection) + eps) / (torch.sum(noise * noise) + eps))

