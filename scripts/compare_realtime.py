#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ha_denoise.audio import read_wav, write_wav
from ha_denoise.features import FeatureConfig, enhance_with_mask, extract_features, feature_config_from_dict
from ha_denoise.metrics import si_sdr
from ha_denoise.model import TinyCausalTCN
from ha_denoise.realtime import StreamingDenoiser, align_by_delay


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
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--save-audio")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--high-snr-bypass", action="store_true")
    parser.add_argument("--bypass-threshold", type=float, default=0.97)
    parser.add_argument("--bypass-width", type=float, default=0.02)
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint, args.device)
    denoiser = StreamingDenoiser(model, cfg, args.high_snr_bypass, args.bypass_threshold, args.bypass_width)
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
            offline_enhanced = enhance_with_mask(mix[0], offline_mask, cfg)
            realtime_enhanced = denoiser.process(mix, flush=True)

            aligned_clean, aligned_realtime, delay = align_by_delay(clean, realtime_enhanced, cfg.n_fft)
            aligned_offline, aligned_realtime_for_diff, _ = align_by_delay(
                offline_enhanced,
                realtime_enhanced,
                cfg.n_fft,
            )
            samples = min(aligned_offline.numel(), aligned_realtime_for_diff.numel())
            diff = aligned_offline[:samples] - aligned_realtime_for_diff[:samples]
            noisy_score = float(si_sdr(mix[0].detach().cpu(), clean.detach().cpu()))
            offline_score = float(si_sdr(offline_enhanced.detach().cpu(), clean.detach().cpu()))
            realtime_score = float(si_sdr(aligned_realtime.detach().cpu(), aligned_clean.detach().cpu()))
            rows.append(
                {
                    "file": mix_path.name,
                    "estimated_delay_samples": delay,
                    "estimated_delay_ms": delay * 1000.0 / cfg.sample_rate,
                    "noisy_si_sdr": noisy_score,
                    "offline_si_sdr": offline_score,
                    "realtime_si_sdr": realtime_score,
                    "realtime_si_sdr_improvement": realtime_score - noisy_score,
                    "realtime_vs_offline_si_sdr_delta": realtime_score - offline_score,
                    "aligned_waveform_mse": float(torch.mean(diff.detach().cpu() ** 2)),
                }
            )
            if save_dir:
                write_wav(save_dir / mix_path.name.replace("mix_", "realtime_"), sr, realtime_enhanced.detach().cpu().numpy())

    summary = {
        "items": len(rows),
        "mean_estimated_delay_samples": sum(r["estimated_delay_samples"] for r in rows) / len(rows),
        "mean_estimated_delay_ms": sum(r["estimated_delay_ms"] for r in rows) / len(rows),
        "mean_noisy_si_sdr": sum(r["noisy_si_sdr"] for r in rows) / len(rows),
        "mean_offline_si_sdr": sum(r["offline_si_sdr"] for r in rows) / len(rows),
        "mean_realtime_si_sdr": sum(r["realtime_si_sdr"] for r in rows) / len(rows),
        "mean_realtime_si_sdr_improvement": sum(r["realtime_si_sdr_improvement"] for r in rows) / len(rows),
        "mean_realtime_vs_offline_si_sdr_delta": sum(r["realtime_vs_offline_si_sdr_delta"] for r in rows) / len(rows),
        "mean_aligned_waveform_mse": sum(r["aligned_waveform_mse"] for r in rows) / len(rows),
    }
    print(json.dumps(summary, indent=2))
    if save_dir:
        (save_dir / "metrics.json").write_text(json.dumps({"summary": summary, "items": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
