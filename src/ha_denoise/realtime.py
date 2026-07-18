from __future__ import annotations

import torch

from ha_denoise.features import FeatureConfig, make_band_matrix
from ha_denoise.model import TinyCausalTCN
from ha_denoise.streaming import StreamingTinyCausalTCN


class StreamingDenoiser:
    """Causal dual-mic STFT, frame-by-frame model inference, and overlap-add synthesis."""

    def __init__(self, model: TinyCausalTCN, cfg: FeatureConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.window = torch.hann_window(cfg.n_fft, device=self.device, dtype=self.dtype)
        self.band_matrix = make_band_matrix(cfg.n_fft, cfg.bands, cfg.sample_rate).to(self.device, self.dtype)
        self.model_stream = StreamingTinyCausalTCN(model)
        self.reset()

    def reset(self) -> None:
        self.input_buffer = torch.zeros(2, self.cfg.n_fft, device=self.device, dtype=self.dtype)
        self.output_buffer = torch.zeros(self.cfg.n_fft, device=self.device, dtype=self.dtype)
        self.norm_buffer = torch.zeros(self.cfg.n_fft, device=self.device, dtype=self.dtype)
        self.model_stream.reset()

    def _features_from_spectrum(self, spec0: torch.Tensor, spec1: torch.Tensor) -> torch.Tensor:
        p0 = spec0.abs().square() @ self.band_matrix
        p1 = spec1.abs().square() @ self.band_matrix
        log0 = torch.log(torch.clamp(p0, min=1e-8))
        log1 = torch.log(torch.clamp(p1, min=1e-8))
        ild = torch.log(torch.clamp(p0, min=1e-8) / torch.clamp(p1, min=1e-8))
        return torch.clamp(torch.cat([log0, log1, ild], dim=0), -20.0, 20.0)

    def _mask_to_bins(self, band_mask: torch.Tensor) -> torch.Tensor:
        bin_mask = band_mask @ self.band_matrix.transpose(0, 1)
        return torch.clamp(bin_mask, 0.0, 1.0)

    def process_hop(self, hop: torch.Tensor) -> torch.Tensor:
        """Process one [2, hop_length] input hop and return one enhanced [hop_length] hop."""
        if hop.shape != (2, self.cfg.hop_length):
            raise ValueError(f"hop must have shape [2, {self.cfg.hop_length}]")
        h = self.cfg.hop_length
        self.input_buffer[:, :-h] = self.input_buffer[:, h:].clone()
        self.input_buffer[:, -h:] = hop.to(self.device, self.dtype)

        frame = self.input_buffer * self.window[None, :]
        spec0 = torch.fft.rfft(frame[0], n=self.cfg.n_fft)
        spec1 = torch.fft.rfft(frame[1], n=self.cfg.n_fft)
        feat = self._features_from_spectrum(spec0, spec1)
        band_mask = self.model_stream.process_frame(feat)
        enhanced_spec = spec0 * self._mask_to_bins(band_mask)
        enhanced_frame = torch.fft.irfft(enhanced_spec, n=self.cfg.n_fft) * self.window

        self.output_buffer += enhanced_frame
        self.norm_buffer += self.window.square()
        valid = self.norm_buffer[:h] > 1e-8
        out = torch.zeros(h, device=self.device, dtype=self.dtype)
        out[valid] = self.output_buffer[:h][valid] / self.norm_buffer[:h][valid]

        self.output_buffer[:-h] = self.output_buffer[h:].clone()
        self.output_buffer[-h:] = 0.0
        self.norm_buffer[:-h] = self.norm_buffer[h:].clone()
        self.norm_buffer[-h:] = 0.0
        return out

    def process(self, mix: torch.Tensor, flush: bool = True) -> torch.Tensor:
        """Process a [2, samples] waveform and return a mono enhanced waveform."""
        if mix.ndim != 2 or mix.shape[0] != 2:
            raise ValueError("mix must have shape [2, samples]")
        self.reset()
        h = self.cfg.hop_length
        samples = mix.shape[1]
        pad = (h - samples % h) % h
        if pad:
            mix = torch.nn.functional.pad(mix, (0, pad))
        hops = []
        with torch.no_grad():
            for start in range(0, mix.shape[1], h):
                hops.append(self.process_hop(mix[:, start : start + h]))
            if flush:
                zero_hop = torch.zeros(2, h, device=self.device, dtype=self.dtype)
                for _ in range(self.cfg.n_fft // h):
                    hops.append(self.process_hop(zero_hop))
        return torch.cat(hops, dim=0)[: samples + (self.cfg.n_fft if flush else 0)]


def estimate_delay(reference: torch.Tensor, candidate: torch.Tensor, max_lag: int) -> int:
    """Return lag where candidate[lag:] best matches reference for lag >= 0."""
    n = min(reference.numel(), candidate.numel())
    reference = reference[:n] - reference[:n].mean()
    candidate = candidate[:n] - candidate[:n].mean()
    best_lag = 0
    best_score = None
    for lag in range(max_lag + 1):
        if lag >= n:
            break
        a = reference[: n - lag]
        b = candidate[lag:n]
        score = torch.sum(a * b)
        if best_score is None or score > best_score:
            best_score = score
            best_lag = lag
    return best_lag


def align_by_delay(reference: torch.Tensor, candidate: torch.Tensor, max_lag: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    lag = estimate_delay(reference, candidate, max_lag)
    n = min(reference.numel(), candidate.numel() - lag)
    return reference[:n], candidate[lag : lag + n], lag
