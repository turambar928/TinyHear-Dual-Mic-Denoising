#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ha_denoise.audio import read_wav, write_wav
from ha_denoise.features import FeatureConfig, istft, stft
from ha_denoise.metrics import si_sdr


def moving_average(x: torch.Tensor, window: int, causal: bool) -> torch.Tensor:
    if window <= 1:
        return x
    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype) / float(window)
    flat = x.reshape(-1, 1, x.shape[-1])
    if causal:
        padded = F.pad(flat, (window - 1, 0), mode="replicate")
    else:
        left = window // 2
        right = window - 1 - left
        padded = F.pad(flat, (left, right), mode="replicate")
    out = F.conv1d(padded, kernel)
    return out.reshape_as(x)


def moving_average_complex(x: torch.Tensor, window: int, causal: bool) -> torch.Tensor:
    return torch.complex(moving_average(x.real, window, causal), moving_average(x.imag, window, causal))


def oracle_local_mwf(
    mix: torch.Tensor,
    clean: torch.Tensor,
    cfg: FeatureConfig,
    window: int,
    causal: bool,
    diagonal_loading: float,
) -> torch.Tensor:
    """Oracle two-mic local MWF estimating clean mic0.

    This is a teacher/upper-bound diagnostic, not a deployable algorithm,
    because it uses clean speech to estimate the local filter target.
    """
    mix_spec = torch.stack([stft(mix[0], cfg), stft(mix[1], cfg)], dim=0)
    clean_spec = torch.stack([stft(clean[0], cfg), stft(clean[1], cfg)], dim=0)
    frames = min(mix_spec.shape[-1], clean_spec.shape[-1])
    mix_spec = mix_spec[:, :, :frames]
    clean_spec = clean_spec[:, :, :frames]
    x0 = mix_spec[0]
    x1 = mix_spec[1]
    s0 = clean_spec[0]

    r00 = moving_average_complex(x0 * torch.conj(x0), window, causal).real
    r01 = moving_average_complex(x0 * torch.conj(x1), window, causal)
    r10 = torch.conj(r01)
    r11 = moving_average_complex(x1 * torch.conj(x1), window, causal).real
    p0 = moving_average_complex(x0 * torch.conj(s0), window, causal)
    p1 = moving_average_complex(x1 * torch.conj(s0), window, causal)

    trace = (r00 + r11).clamp_min(1e-8)
    load = float(diagonal_loading) * trace + 1e-8
    a = r00 + load
    d = r11 + load
    b = r01
    c = r10
    det = a * d - b * c + 1e-8
    w0 = (d * p0 - b * p1) / det
    w1 = (-c * p0 + a * p1) / det
    enhanced_spec = torch.conj(w0) * x0 + torch.conj(w1) * x1
    return istft(enhanced_spec, length=mix.shape[-1], cfg=cfg)


def quantile_indices(count: int, samples: int) -> list[int]:
    if samples >= count:
        return list(range(count))
    if samples == 1:
        return [count // 2]
    return sorted({round(i * (count - 1) / (samples - 1)) for i in range(samples)})


def process_one(mix_path: Path, cfg: FeatureConfig, window: int, causal: bool, diagonal_loading: float):
    clean_path = mix_path.with_name(mix_path.name.replace("mix_", "clean_"))
    sr, mix_np = read_wav(mix_path, cfg.sample_rate)
    _, clean_np = read_wav(clean_path, cfg.sample_rate)
    mix = torch.from_numpy(mix_np[:, :2].T)
    clean = torch.from_numpy(clean_np[:, :2].T)
    enhanced = oracle_local_mwf(mix, clean, cfg, window, causal, diagonal_loading)
    n = min(mix.shape[-1], clean.shape[-1], enhanced.numel())
    return sr, mix[0, :n], clean[0, :n], enhanced[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--save-audio")
    parser.add_argument("--save-listening")
    parser.add_argument("--listening-samples", type=int, default=5)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--diagonal-loading", type=float, default=0.01)
    args = parser.parse_args()

    cfg = FeatureConfig(args.sample_rate, args.n_fft, args.hop_length)
    split_dir = Path(args.data) / args.split
    mix_files = sorted(split_dir.glob("mix_*.wav"))
    if args.max_items is not None:
        mix_files = mix_files[: args.max_items]
    if not mix_files:
        raise FileNotFoundError(f"No mix_*.wav files under {split_dir}")

    save_dir = Path(args.save_audio) if args.save_audio else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mix_path in tqdm(mix_files):
        sr, noisy, clean, enhanced = process_one(mix_path, cfg, args.window, args.causal, args.diagonal_loading)
        noisy_score = float(si_sdr(noisy, clean))
        enhanced_score = float(si_sdr(enhanced, clean))
        rows.append(
            {
                "file": mix_path.name,
                "noisy_si_sdr": noisy_score,
                "enhanced_si_sdr": enhanced_score,
                "si_sdr_improvement": enhanced_score - noisy_score,
                "oracle_window": args.window,
                "oracle_causal": args.causal,
                "diagonal_loading": args.diagonal_loading,
            }
        )
        if save_dir:
            write_wav(save_dir / mix_path.name.replace("mix_", "oracle_mwf_"), sr, enhanced.numpy())

    summary = {
        "items": len(rows),
        "mean_noisy_si_sdr": sum(row["noisy_si_sdr"] for row in rows) / len(rows),
        "mean_enhanced_si_sdr": sum(row["enhanced_si_sdr"] for row in rows) / len(rows),
        "mean_si_sdr_improvement": sum(row["si_sdr_improvement"] for row in rows) / len(rows),
    }
    print(json.dumps(summary, indent=2))
    if save_dir:
        (save_dir / "metrics.json").write_text(json.dumps({"summary": summary, "items": rows}, indent=2), encoding="utf-8")

    if args.save_listening:
        listen_dir = Path(args.save_listening)
        listen_dir.mkdir(parents=True, exist_ok=True)
        ordered = sorted(enumerate(rows), key=lambda item: item[1]["noisy_si_sdr"])
        selected = [ordered[i][0] for i in quantile_indices(len(ordered), args.listening_samples)]
        listen_rows = []
        for out_idx, row_idx in enumerate(selected):
            mix_path = mix_files[row_idx]
            sr, noisy, clean, enhanced = process_one(mix_path, cfg, args.window, args.causal, args.diagonal_loading)
            row = rows[row_idx]
            prefix = f"sample_{out_idx:03d}"
            files = {
                "noisy": f"{prefix}_noisy.wav",
                "clean": f"{prefix}_clean.wav",
                "offline": f"{prefix}_offline.wav",
                "realtime": f"{prefix}_realtime.wav",
            }
            write_wav(listen_dir / files["noisy"], sr, noisy.numpy())
            write_wav(listen_dir / files["clean"], sr, clean.numpy())
            write_wav(listen_dir / files["offline"], sr, enhanced.numpy())
            write_wav(listen_dir / files["realtime"], sr, enhanced.numpy())
            listen_rows.append(
                {
                    "sample": prefix,
                    "source_mix": mix_path.name,
                    "noisy_si_sdr": row["noisy_si_sdr"],
                    "offline_si_sdr": row["enhanced_si_sdr"],
                    "realtime_si_sdr": row["enhanced_si_sdr"],
                    "offline_improvement": row["si_sdr_improvement"],
                    "realtime_improvement": row["si_sdr_improvement"],
                    "files": files,
                    "oracle_window": args.window,
                    "oracle_causal": args.causal,
                    "diagonal_loading": args.diagonal_loading,
                }
            )
        listen_summary = {
            "items": len(listen_rows),
            "mean_noisy_si_sdr": sum(row["noisy_si_sdr"] for row in listen_rows) / len(listen_rows),
            "mean_offline_si_sdr": sum(row["offline_si_sdr"] for row in listen_rows) / len(listen_rows),
            "mean_realtime_si_sdr": sum(row["realtime_si_sdr"] for row in listen_rows) / len(listen_rows),
            "mean_offline_improvement": sum(row["offline_improvement"] for row in listen_rows) / len(listen_rows),
            "mean_realtime_improvement": sum(row["realtime_improvement"] for row in listen_rows) / len(listen_rows),
        }
        (listen_dir / "index.json").write_text(
            json.dumps({"summary": listen_summary, "items": listen_rows}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
