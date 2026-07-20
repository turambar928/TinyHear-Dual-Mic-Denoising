#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ha_denoise.audio import read_wav, write_wav
from ha_denoise.features import FeatureConfig, apply_high_snr_bypass, enhance_with_mask, extract_features, target_band_mask
from ha_denoise.metrics import si_sdr
from ha_denoise.model import TinyCausalTCN


def load_model(checkpoint: str, device: str) -> tuple[TinyCausalTCN, FeatureConfig]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg_d = ckpt["config"]
    cfg = FeatureConfig(cfg_d["sample_rate"], cfg_d["n_fft"], cfg_d["hop_length"], cfg_d["bands"])
    model = TinyCausalTCN(cfg_d["feature_dim"], cfg_d["bands"], cfg_d["channels"], cfg_d["blocks"], cfg_d["kernel_size"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Directory containing mix_*.wav and clean_*.wav.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--save-audio", help="Optional directory for enhanced wav examples.")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--high-snr-bypass", action="store_true")
    parser.add_argument("--bypass-threshold", type=float, default=0.97)
    parser.add_argument("--bypass-width", type=float, default=0.02)
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint, args.device)
    split_dir = Path(args.data) / args.split
    mix_files = sorted(p for p in split_dir.glob("mix_*.wav"))
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
            feat = extract_features(mix, cfg).transpose(0, 1).unsqueeze(0)
            pred_mask = model(feat).squeeze(0).transpose(0, 1)
            if args.high_snr_bypass:
                pred_mask = apply_high_snr_bypass(pred_mask, args.bypass_threshold, args.bypass_width)
            target_mask = target_band_mask(mix[0], clean, cfg)
            t = min(pred_mask.shape[0], target_mask.shape[0])
            enhanced = enhance_with_mask(mix[0], pred_mask, cfg)

            noisy_score = float(si_sdr(mix[0].detach().cpu(), clean.detach().cpu()))
            enhanced_score = float(si_sdr(enhanced.detach().cpu(), clean.detach().cpu()))
            mask_mse = float(torch.mean((pred_mask[:t].cpu() - target_mask[:t].cpu()) ** 2))
            rows.append(
                {
                    "file": mix_path.name,
                    "noisy_si_sdr": noisy_score,
                    "enhanced_si_sdr": enhanced_score,
                    "si_sdr_improvement": enhanced_score - noisy_score,
                    "mask_mse": mask_mse,
                }
            )
            if save_dir:
                write_wav(save_dir / mix_path.name.replace("mix_", "enhanced_"), sr, enhanced.detach().cpu().numpy())

    mean_noisy = sum(r["noisy_si_sdr"] for r in rows) / len(rows)
    mean_enhanced = sum(r["enhanced_si_sdr"] for r in rows) / len(rows)
    mean_mask_mse = sum(r["mask_mse"] for r in rows) / len(rows)
    summary = {
        "items": len(rows),
        "mean_noisy_si_sdr": mean_noisy,
        "mean_enhanced_si_sdr": mean_enhanced,
        "mean_si_sdr_improvement": mean_enhanced - mean_noisy,
        "mean_mask_mse": mean_mask_mse,
    }
    print(json.dumps(summary, indent=2))
    if save_dir:
        (save_dir / "metrics.json").write_text(json.dumps({"summary": summary, "items": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
