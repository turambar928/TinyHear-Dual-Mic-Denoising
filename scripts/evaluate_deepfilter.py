#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ha_denoise.audio import read_wav, write_wav
from ha_denoise.features import enhance_with_deep_filter, extract_features, feature_config_from_dict, match_loudness, rms_ratio
from ha_denoise.metrics import si_sdr
from ha_denoise.model import TinyDeepFilterTCN
from ha_denoise.spatial import apply_spatial_frontend


def load_model(checkpoint: str, device: str):
    ckpt = torch.load(checkpoint, map_location=device)
    cfg_d = ckpt["config"]
    cfg = feature_config_from_dict(cfg_d)
    model = TinyDeepFilterTCN(
        cfg_d["feature_dim"],
        cfg_d["bands"],
        cfg_d["channels"],
        cfg_d["blocks"],
        cfg_d["kernel_size"],
        cfg_d["df_bins"],
        cfg_d["df_order"],
        float(cfg_d.get("coef_scale", 1.5)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def quantile_indices(count: int, samples: int) -> list[int]:
    if samples >= count:
        return list(range(count))
    if samples == 1:
        return [count // 2]
    return sorted({round(i * (count - 1) / (samples - 1)) for i in range(samples)})


def process_one(model, cfg, mix_path: Path, device: str, loudness_match: bool, target_rms_ratio: float, max_gain_db: float):
    clean_path = mix_path.with_name(mix_path.name.replace("mix_", "clean_"))
    sr, mix_np = read_wav(mix_path, cfg.sample_rate)
    _, clean_np = read_wav(clean_path, cfg.sample_rate)
    mix = torch.from_numpy(mix_np[:, :2].T).to(device)
    clean = torch.from_numpy(clean_np[:, 0]).to(device)
    beamformed, spatial_info = apply_spatial_frontend(mix, cfg, max_lag=8, analysis_samples=cfg.sample_rate // 2)
    feat = extract_features(mix, cfg).transpose(0, 1).unsqueeze(0)
    gain, coef = model(feat)
    enhanced = enhance_with_deep_filter(beamformed, gain.squeeze(0).transpose(0, 1), coef.squeeze(0), cfg)
    n = min(mix.shape[-1], clean.numel(), enhanced.numel())
    noisy = mix[0, :n]
    beamformed = beamformed[:n]
    clean = clean[:n]
    enhanced = enhanced[:n]
    loudness_gain = torch.ones((), device=device, dtype=enhanced.dtype)
    if loudness_match:
        enhanced, loudness_gain = match_loudness(beamformed, enhanced, target_rms_ratio, max_gain_db)
    return sr, noisy, beamformed, clean, enhanced, spatial_info, float(loudness_gain.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--save-audio")
    parser.add_argument("--save-listening")
    parser.add_argument("--listening-samples", type=int, default=5)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--loudness-match", action="store_true")
    parser.add_argument("--target-rms-ratio", type=float, default=0.92)
    parser.add_argument("--max-gain-db", type=float, default=5.0)
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
            sr, noisy, beamformed, clean, enhanced, spatial_info, gain = process_one(
                model,
                cfg,
                mix_path,
                args.device,
                args.loudness_match,
                args.target_rms_ratio,
                args.max_gain_db,
            )
            noisy_score = float(si_sdr(noisy.detach().cpu(), clean.detach().cpu()))
            enhanced_score = float(si_sdr(enhanced.detach().cpu(), clean.detach().cpu()))
            rows.append(
                {
                    "file": mix_path.name,
                    "spatial_frontend": spatial_info.get("mode"),
                    "beamform_lag_samples": spatial_info.get("lag"),
                    "mean_coherence": spatial_info.get("mean_coherence"),
                    "mean_spatial_gain": spatial_info.get("mean_spatial_gain"),
                    "beamformed_si_sdr": float(si_sdr(beamformed.detach().cpu(), clean.detach().cpu())),
                    "noisy_si_sdr": noisy_score,
                    "enhanced_si_sdr": enhanced_score,
                    "si_sdr_improvement": enhanced_score - noisy_score,
                    "output_input_rms_ratio": float(rms_ratio(beamformed.detach().cpu(), enhanced.detach().cpu())),
                    "loudness_gain": gain,
                }
            )
            if save_dir:
                write_wav(save_dir / mix_path.name.replace("mix_", "deepfilter_"), sr, enhanced.detach().cpu().numpy())

    summary = {
        "items": len(rows),
        "mean_noisy_si_sdr": sum(row["noisy_si_sdr"] for row in rows) / len(rows),
        "mean_enhanced_si_sdr": sum(row["enhanced_si_sdr"] for row in rows) / len(rows),
        "mean_si_sdr_improvement": sum(row["si_sdr_improvement"] for row in rows) / len(rows),
        "mean_output_input_rms_ratio": sum(row["output_input_rms_ratio"] for row in rows) / len(rows),
        "mean_loudness_gain": sum(row["loudness_gain"] for row in rows) / len(rows),
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
            sr, noisy, beamformed, clean, enhanced, spatial_info, gain = process_one(
                model,
                cfg,
                mix_path,
                args.device,
                args.loudness_match,
                args.target_rms_ratio,
                args.max_gain_db,
            )
            row = rows[row_idx]
            prefix = f"sample_{out_idx:03d}"
            files = {
                "noisy": f"{prefix}_noisy.wav",
                "clean": f"{prefix}_clean.wav",
                "offline": f"{prefix}_offline.wav",
                "realtime": f"{prefix}_realtime.wav",
            }
            write_wav(listen_dir / files["noisy"], sr, noisy.detach().cpu().numpy())
            write_wav(listen_dir / files["clean"], sr, clean.detach().cpu().numpy())
            write_wav(listen_dir / files["offline"], sr, enhanced.detach().cpu().numpy())
            write_wav(listen_dir / files["realtime"], sr, enhanced.detach().cpu().numpy())
            listen_rows.append(
                {
                    "sample": prefix,
                    "source_mix": mix_path.name,
                    "spatial_frontend": spatial_info.get("mode"),
                    "beamform_lag_samples": spatial_info.get("lag"),
                    "beamformed_si_sdr": row.get("beamformed_si_sdr"),
                    "loudness_gain": gain,
                    "noisy_si_sdr": row["noisy_si_sdr"],
                    "offline_si_sdr": row["enhanced_si_sdr"],
                    "realtime_si_sdr": row["enhanced_si_sdr"],
                    "offline_improvement": row["si_sdr_improvement"],
                    "realtime_improvement": row["si_sdr_improvement"],
                    "files": files,
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
