#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ha_denoise.audio import read_wav, write_wav
from ha_denoise.features import FeatureConfig, enhance_with_mask, target_band_mask
from ha_denoise.metrics import si_sdr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--save-audio")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--bands", type=int, default=32)
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()

    cfg = FeatureConfig(args.sample_rate, args.n_fft, args.hop_length, args.bands)
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
        clean_path = mix_path.with_name(mix_path.name.replace("mix_", "clean_"))
        sr, mix_np = read_wav(mix_path, cfg.sample_rate)
        _, clean_np = read_wav(clean_path, cfg.sample_rate)
        mix = torch.from_numpy(mix_np[:, 0])
        clean = torch.from_numpy(clean_np[:, 0])
        mask = target_band_mask(mix, clean, cfg)
        enhanced = enhance_with_mask(mix, mask, cfg)
        noisy_score = float(si_sdr(mix, clean))
        enhanced_score = float(si_sdr(enhanced, clean))
        rows.append(
            {
                "file": mix_path.name,
                "noisy_si_sdr": noisy_score,
                "enhanced_si_sdr": enhanced_score,
                "si_sdr_improvement": enhanced_score - noisy_score,
            }
        )
        if save_dir:
            write_wav(save_dir / mix_path.name.replace("mix_", "oracle_"), sr, enhanced.numpy())

    mean_noisy = sum(row["noisy_si_sdr"] for row in rows) / len(rows)
    mean_enhanced = sum(row["enhanced_si_sdr"] for row in rows) / len(rows)
    summary = {
        "items": len(rows),
        "mean_noisy_si_sdr": mean_noisy,
        "mean_enhanced_si_sdr": mean_enhanced,
        "mean_si_sdr_improvement": mean_enhanced - mean_noisy,
    }
    print(json.dumps(summary, indent=2))
    if save_dir:
        (save_dir / "metrics.json").write_text(json.dumps({"summary": summary, "items": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
