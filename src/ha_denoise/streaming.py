from __future__ import annotations

import torch
import torch.nn.functional as F

from ha_denoise.model import TinyCausalTCN


class StreamingTinyCausalTCN:
    """Frame-by-frame inference wrapper for TinyCausalTCN.

    Input and output frames use the same feature/mask layout as the batch model:
    one input frame is [feature_dim], one output frame is [bands].
    """

    def __init__(self, model: TinyCausalTCN) -> None:
        self.model = model
        self.reset()

    def reset(self) -> None:
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.histories = [
            torch.zeros(block.depthwise.in_channels, block.left_pad, device=device, dtype=dtype)
            for block in self.model.tcn
        ]

    def process_frame(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.ndim != 1 or frame.numel() != self.model.feature_dim:
            raise ValueError(f"frame must have shape [{self.model.feature_dim}]")
        x = frame.view(1, self.model.feature_dim, 1)
        y = self.model.stem(x).squeeze(0).squeeze(-1)

        for idx, block in enumerate(self.model.tcn):
            residual = y
            history = self.histories[idx]
            padded = torch.cat([history, y[:, None]], dim=1).unsqueeze(0)
            z = block.depthwise(padded).squeeze(0).squeeze(-1)
            z = F.relu(z)
            z = block.pointwise(z.view(1, -1, 1)).squeeze(0).squeeze(-1)
            y = F.relu(z) + residual
            self.histories[idx] = torch.cat([history[:, 1:], residual[:, None]], dim=1)

        out = self.model.head(y.view(1, -1, 1)).squeeze(0).squeeze(-1)
        return torch.clamp(out * 0.2 + 0.5, 0.0, 1.0)


def run_streaming_model(model: TinyCausalTCN, features: torch.Tensor) -> torch.Tensor:
    """Run a [T, feature_dim] feature sequence frame by frame and return [T, bands]."""
    if features.ndim != 2:
        raise ValueError("features must have shape [frames, feature_dim]")
    streamer = StreamingTinyCausalTCN(model)
    outputs = []
    with torch.no_grad():
        for t in range(features.shape[0]):
            outputs.append(streamer.process_frame(features[t]))
    return torch.stack(outputs, dim=0)
