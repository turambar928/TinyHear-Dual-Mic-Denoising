#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ha_denoise.audio import read_wav, write_wav
from ha_denoise.features import FeatureConfig, enhance_with_mask, extract_features
from ha_denoise.metrics import si_sdr
from ha_denoise.model import TinyCausalTCN
from ha_denoise.realtime import StreamingDenoiser, align_by_delay


def load_model(checkpoint: str, device: str) -> tuple[TinyCausalTCN, FeatureConfig]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg_d = ckpt["config"]
    cfg = FeatureConfig(cfg_d["sample_rate"], cfg_d["n_fft"], cfg_d["hop_length"], cfg_d["bands"])
    model = TinyCausalTCN(cfg_d["feature_dim"], cfg_d["bands"], cfg_d["channels"], cfg_d["blocks"], cfg_d["kernel_size"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def quantile_indices(count: int, samples: int) -> list[int]:
    if samples >= count:
        return list(range(count))
    if samples == 1:
        return [count // 2]
    return sorted({round(i * (count - 1) / (samples - 1)) for i in range(samples)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint, args.device)
    denoiser = StreamingDenoiser(model, cfg)
    split_dir = Path(args.data) / args.split
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mix_files = sorted(split_dir.glob("mix_*.wav"))
    if not mix_files:
        raise FileNotFoundError(f"No mix_*.wav files under {split_dir}")

    scored = []
    for mix_path in tqdm(mix_files, desc="score"):
        clean_path = mix_path.with_name(mix_path.name.replace("mix_", "clean_"))
        _, mix_np = read_wav(mix_path, cfg.sample_rate)
        _, clean_np = read_wav(clean_path, cfg.sample_rate)
        mix_ref = torch.from_numpy(mix_np[:, 0])
        clean = torch.from_numpy(clean_np[:, 0])
        scored.append((float(si_sdr(mix_ref, clean)), mix_path))
    scored.sort(key=lambda item: item[0])
    selected = [scored[i][1] for i in quantile_indices(len(scored), args.samples)]

    rows = []
    with torch.no_grad():
        for idx, mix_path in enumerate(tqdm(selected, desc="render")):
            clean_path = mix_path.with_name(mix_path.name.replace("mix_", "clean_"))
            sr, mix_np = read_wav(mix_path, cfg.sample_rate)
            _, clean_np = read_wav(clean_path, cfg.sample_rate)
            mix = torch.from_numpy(mix_np[:, :2].T).to(args.device)
            clean = torch.from_numpy(clean_np[:, 0]).to(args.device)

            features = extract_features(mix, cfg)
            mask = model(features.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
            offline = enhance_with_mask(mix[0], mask, cfg)
            realtime_raw = denoiser.process(mix, flush=True)
            aligned_clean, realtime, delay = align_by_delay(clean, realtime_raw, cfg.n_fft)
            length = min(clean.numel(), offline.numel(), realtime.numel(), aligned_clean.numel())

            noisy = mix[0, :length].detach().cpu()
            clean_out = aligned_clean[:length].detach().cpu()
            offline_out = offline[:length].detach().cpu()
            realtime_out = realtime[:length].detach().cpu()

            prefix = f"sample_{idx:03d}"
            files = {
                "noisy": f"{prefix}_noisy.wav",
                "clean": f"{prefix}_clean.wav",
                "offline": f"{prefix}_offline.wav",
                "realtime": f"{prefix}_realtime.wav",
            }
            write_wav(out_dir / files["noisy"], sr, noisy.numpy())
            write_wav(out_dir / files["clean"], sr, clean_out.numpy())
            write_wav(out_dir / files["offline"], sr, offline_out.numpy())
            write_wav(out_dir / files["realtime"], sr, realtime_out.numpy())

            noisy_score = float(si_sdr(noisy, clean_out))
            offline_score = float(si_sdr(offline_out, clean_out))
            realtime_score = float(si_sdr(realtime_out, clean_out))
            rows.append(
                {
                    "sample": prefix,
                    "source_mix": mix_path.name,
                    "estimated_delay_samples": delay,
                    "estimated_delay_ms": delay * 1000.0 / cfg.sample_rate,
                    "noisy_si_sdr": noisy_score,
                    "offline_si_sdr": offline_score,
                    "realtime_si_sdr": realtime_score,
                    "offline_improvement": offline_score - noisy_score,
                    "realtime_improvement": realtime_score - noisy_score,
                    "files": files,
                }
            )

    summary = {
        "items": len(rows),
        "mean_noisy_si_sdr": sum(r["noisy_si_sdr"] for r in rows) / len(rows),
        "mean_offline_si_sdr": sum(r["offline_si_sdr"] for r in rows) / len(rows),
        "mean_realtime_si_sdr": sum(r["realtime_si_sdr"] for r in rows) / len(rows),
        "mean_offline_improvement": sum(r["offline_improvement"] for r in rows) / len(rows),
        "mean_realtime_improvement": sum(r["realtime_improvement"] for r in rows) / len(rows),
    }
    payload = {"summary": summary, "items": rows}
    (out_dir / "index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
