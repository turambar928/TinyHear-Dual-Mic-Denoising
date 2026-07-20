#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ha_denoise.audio import read_wav, write_wav
from ha_denoise.features import (
    FeatureConfig,
    apply_high_snr_bypass,
    enhance_with_mask,
    extract_features,
    feature_config_from_dict,
)
from ha_denoise.metrics import si_sdr
from ha_denoise.model import TinyCausalTCN
from ha_denoise.streaming import run_streaming_model


def load_model(checkpoint: str, device: str) -> tuple[TinyCausalTCN, FeatureConfig]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg_d = ckpt["config"]
    cfg = feature_config_from_dict(cfg_d)
    model = TinyCausalTCN(cfg_d["feature_dim"], cfg_d["bands"], cfg_d["channels"], cfg_d["blocks"], cfg_d["kernel_size"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Directory containing split/mix_*.wav and split/clean_*.wav.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--save-audio", help="Optional directory for streaming enhanced examples.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--high-snr-bypass", action="store_true")
    parser.add_argument("--bypass-threshold", type=float, default=0.97)
    parser.add_argument("--bypass-width", type=float, default=0.02)
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint, args.device)
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
    with torch.no_grad():
        for mix_path in tqdm(mix_files):
            clean_path = mix_path.with_name(mix_path.name.replace("mix_", "clean_"))
            sr, mix_np = read_wav(mix_path, cfg.sample_rate)
            _, clean_np = read_wav(clean_path, cfg.sample_rate)
            mix = torch.from_numpy(mix_np[:, :2].T).to(args.device)
            clean = torch.from_numpy(clean_np[:, 0]).to(args.device)

            features = extract_features(mix, cfg)
            offline_mask = model(features.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
            streaming_mask = run_streaming_model(model, features)
            if args.high_snr_bypass:
                offline_mask = apply_high_snr_bypass(offline_mask, args.bypass_threshold, args.bypass_width)
                streaming_mask = apply_high_snr_bypass(streaming_mask, args.bypass_threshold, args.bypass_width)
            offline_enhanced = enhance_with_mask(mix[0], offline_mask, cfg)
            streaming_enhanced = enhance_with_mask(mix[0], streaming_mask, cfg)

            frames = min(offline_mask.shape[0], streaming_mask.shape[0])
            samples = min(offline_enhanced.numel(), streaming_enhanced.numel())
            mask_diff = torch.abs(offline_mask[:frames] - streaming_mask[:frames])
            wav_diff = offline_enhanced[:samples] - streaming_enhanced[:samples]
            offline_score = float(si_sdr(offline_enhanced.detach().cpu(), clean.detach().cpu()))
            streaming_score = float(si_sdr(streaming_enhanced.detach().cpu(), clean.detach().cpu()))
            rows.append(
                {
                    "file": mix_path.name,
                    "mask_max_abs_diff": float(mask_diff.max().cpu()),
                    "mask_mean_abs_diff": float(mask_diff.mean().cpu()),
                    "waveform_mse": float(torch.mean(wav_diff.detach().cpu() ** 2)),
                    "offline_si_sdr": offline_score,
                    "streaming_si_sdr": streaming_score,
                    "si_sdr_delta": streaming_score - offline_score,
                }
            )
            if save_dir:
                write_wav(save_dir / mix_path.name.replace("mix_", "streaming_"), sr, streaming_enhanced.detach().cpu().numpy())

    summary = {
        "items": len(rows),
        "max_mask_max_abs_diff": max(r["mask_max_abs_diff"] for r in rows),
        "mean_mask_mean_abs_diff": sum(r["mask_mean_abs_diff"] for r in rows) / len(rows),
        "mean_waveform_mse": sum(r["waveform_mse"] for r in rows) / len(rows),
        "mean_offline_si_sdr": sum(r["offline_si_sdr"] for r in rows) / len(rows),
        "mean_streaming_si_sdr": sum(r["streaming_si_sdr"] for r in rows) / len(rows),
        "mean_si_sdr_delta": sum(r["si_sdr_delta"] for r in rows) / len(rows),
    }
    print(json.dumps(summary, indent=2))
    if save_dir:
        (save_dir / "metrics.json").write_text(json.dumps({"summary": summary, "items": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
