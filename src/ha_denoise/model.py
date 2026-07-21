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
        min_gain: float = 0.0,
        max_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.bands = bands
        self.channels = channels
        self.blocks = blocks
        self.kernel_size = kernel_size
        self.min_gain = min_gain
        self.max_gain = max_gain
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
        unit = torch.clamp(self.head(y) * 0.2 + 0.5, 0.0, 1.0)
        return self.min_gain + (self.max_gain - self.min_gain) * unit


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class TinyDeepFilterTCN(nn.Module):
    def __init__(
        self,
        feature_dim: int = 192,
        bands: int = 32,
        channels: int = 96,
        blocks: int = 8,
        kernel_size: int = 5,
        df_bins: int = 64,
        df_order: int = 3,
        coef_scale: float = 1.5,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.bands = bands
        self.channels = channels
        self.blocks = blocks
        self.kernel_size = kernel_size
        self.df_bins = df_bins
        self.df_order = df_order
        self.coef_scale = coef_scale
        self.stem = nn.Sequential(nn.Conv1d(feature_dim, channels, kernel_size=1), nn.ReLU())
        dilations = [1, 2, 4, 8]
        self.tcn = nn.Sequential(
            *[
                CausalDepthwiseBlock(channels, kernel_size, dilations[i % len(dilations)])
                for i in range(blocks)
            ]
        )
        self.gain_head = nn.Conv1d(channels, bands, kernel_size=1)
        self.df_head = nn.Conv1d(channels, df_bins * df_order * 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.stem(x)
        y = self.tcn(y)
        gain = torch.clamp(self.gain_head(y) * 0.2 + 0.5, 0.0, 1.0)
        coef = torch.tanh(self.df_head(y)) * self.coef_scale
        b, _, t = coef.shape
        coef = coef.view(b, self.df_bins, self.df_order, 2, t).permute(0, 4, 1, 2, 3).contiguous()
        return gain, coef

    def load_denoiser_backbone(self, state: dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        copied = {}
        for name, value in state.items():
            target_name = "gain_head" + name[len("head") :] if name.startswith("head") else name
            if target_name in own and own[target_name].shape == value.shape:
                copied[target_name] = value
        self.load_state_dict({**own, **copied})
