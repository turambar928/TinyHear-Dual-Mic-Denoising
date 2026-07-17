from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CausalDepthwiseBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=True,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = F.pad(x, (self.left_pad, 0))
        y = self.act(self.depthwise(y))
        y = self.act(self.pointwise(y))
        return y + residual


class TinyCausalTCN(nn.Module):
    def __init__(
        self,
        feature_dim: int = 96,
        bands: int = 32,
        channels: int = 112,
        blocks: int = 8,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.bands = bands
        self.channels = channels
        self.blocks = blocks
        self.kernel_size = kernel_size
        self.stem = nn.Sequential(nn.Conv1d(feature_dim, channels, kernel_size=1), nn.ReLU())
        dilations = [1, 2, 4, 8]
        self.tcn = nn.Sequential(
            *[
                CausalDepthwiseBlock(channels, kernel_size, dilations[i % len(dilations)])
                for i in range(blocks)
            ]
        )
        self.head = nn.Conv1d(channels, bands, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        y = self.tcn(y)
        # Hard sigmoid is easy to approximate with integer arithmetic.
        return torch.clamp(self.head(y) * 0.2 + 0.5, 0.0, 1.0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

