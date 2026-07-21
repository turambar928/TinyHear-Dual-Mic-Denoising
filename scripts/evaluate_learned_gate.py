#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ha_denoise.audio import read_wav, write_wav
from ha_denoise.features import enhance_with_mask, extract_features, feature_config_from_dict
from ha_denoise.metrics import si_sdr
from ha_denoise.model import TinyCausalTCN
from train_gate import TinyGate, pooled_features


def load_denoiser(checkpoint: str, device: str):
    ckpt = torch.load(checkpoint, map_location=device)
    cfg_d = ckpt["config"]
    cfg = feature_config_from_dict(cfg_d)
    model = TinyCausalTCN(cfg_d["feature_dim"], cfg_d["bands"], cfg_d["channels"], cfg_d["blocks"], cfg_d["kernel_size"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def load_gate(checkpoint: str, device: str) -> TinyGate:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = TinyGate(int(cfg["input_dim"]), int(cfg["hidden"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def quantile_indices(count: int, samples: int) -> list[int]:
    if samples >= count:
        return list(range(count))
    if samples == 1:
        return [count // 2]
    return sorted({round(i * (count - 1) / (samples - 1)) for i in range(samples)})


def process_one(denoiser, gate_model, cfg, mix_path: Path, device: str):
    clean_path = mix_path.with_name(mix_path.name.replace("mix_", "clean_"))
    sr, mix_np = read_wav(mix_path, cfg.sample_rate)
    _, clean_np = read_wav(clean_path, cfg.sample_rate)
    mix = torch.from_numpy(mix_np[:, :2].T).to(device)
    clean = torch.from_numpy(clean_np[:, 0]).to(device)
    features = extract_features(mix, cfg)
    feat_batch = features.transpose(0, 1).unsqueeze(0)
    mask = denoiser(feat_batch).squeeze(0).transpose(0, 1)
    enhanced = enhance_with_mask(mix[0], mask, cfg)
    valid = torch.ones(1, 1, feat_batch.shape[-1], device=device, dtype=feat_batch.dtype)
    gate_input = pooled_features(feat_batch, valid)
    gate = float(torch.sigmoid(gate_model(gate_input)).item())
    n = min(mix.shape[-1], clean.numel(), enhanced.numel())
    noisy = mix[0, :n]
    clean = clean[:n]
    enhanced = enhanced[:n]
    gated = gate * enhanced + (1.0 - gate) * noisy
    return sr, noisy, clean, enhanced, gated, gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--save-audio")
    parser.add_argument("--save-listening")
    parser.add_argument("--listening-samples", type=int, default=5)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    denoiser, cfg = load_denoiser(args.checkpoint, args.device)
    gate_model = load_gate(args.gate, args.device)
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
            sr, noisy, clean, enhanced, gated, gate = process_one(denoiser, gate_model, cfg, mix_path, args.device)
            noisy_score = float(si_sdr(noisy.detach().cpu(), clean.detach().cpu()))
            enhanced_score = float(si_sdr(enhanced.detach().cpu(), clean.detach().cpu()))
            gated_score = float(si_sdr(gated.detach().cpu(), clean.detach().cpu()))
            rows.append(
                {
                    "file": mix_path.name,
                    "learned_gate": gate,
                    "noisy_si_sdr": noisy_score,
                    "enhanced_si_sdr": gated_score,
                    "si_sdr_improvement": gated_score - noisy_score,
                    "model_enhanced_si_sdr": enhanced_score,
                    "model_si_sdr_improvement": enhanced_score - noisy_score,
                }
            )
            if save_dir:
                write_wav(save_dir / mix_path.name.replace("mix_", "learned_gate_"), sr, gated.detach().cpu().numpy())

    mean_noisy = sum(row["noisy_si_sdr"] for row in rows) / len(rows)
    mean_model = sum(row["model_enhanced_si_sdr"] for row in rows) / len(rows)
    mean_gated = sum(row["enhanced_si_sdr"] for row in rows) / len(rows)
    summary = {
        "items": len(rows),
        "mean_noisy_si_sdr": mean_noisy,
        "mean_model_enhanced_si_sdr": mean_model,
        "mean_model_si_sdr_improvement": mean_model - mean_noisy,
        "mean_enhanced_si_sdr": mean_gated,
        "mean_si_sdr_improvement": mean_gated - mean_noisy,
        "mean_gate": sum(row["learned_gate"] for row in rows) / len(rows),
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
            sr, noisy, clean, enhanced, gated, gate = process_one(denoiser, gate_model, cfg, mix_path, args.device)
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
            write_wav(listen_dir / files["realtime"], sr, gated.detach().cpu().numpy())
            listen_rows.append(
                {
                    "sample": prefix,
                    "source_mix": mix_path.name,
                    "learned_gate": gate,
                    "noisy_si_sdr": row["noisy_si_sdr"],
                    "offline_si_sdr": row["model_enhanced_si_sdr"],
                    "realtime_si_sdr": row["enhanced_si_sdr"],
                    "offline_improvement": row["model_si_sdr_improvement"],
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
