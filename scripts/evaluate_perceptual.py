#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ha_denoise.audio import read_wav
from ha_denoise.metrics import si_sdr


def optional_imports():
    try:
        from pystoi import stoi  # type: ignore
    except Exception:
        stoi = None
    try:
        from pesq import pesq  # type: ignore
    except Exception:
        pesq = None
    return stoi, pesq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--enhanced-dir", required=True)
    parser.add_argument("--pattern", default="sample_*_realtime.wav")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    stoi_fn, pesq_fn = optional_imports()
    enhanced_dir = Path(args.enhanced_dir)
    clean_dir = Path(args.clean_dir)
    enhanced_files = sorted(enhanced_dir.glob(args.pattern))
    if not enhanced_files:
        raise FileNotFoundError(f"No files matching {args.pattern} under {enhanced_dir}")

    rows = []
    for enhanced_path in tqdm(enhanced_files):
        clean_name = enhanced_path.name.replace("_realtime.wav", "_clean.wav").replace("_offline.wav", "_clean.wav")
        clean_path = clean_dir / clean_name
        if not clean_path.exists():
            raise FileNotFoundError(f"Missing clean file: {clean_path}")
        _, enhanced_np = read_wav(enhanced_path, args.sample_rate)
        _, clean_np = read_wav(clean_path, args.sample_rate)
        n = min(enhanced_np.shape[0], clean_np.shape[0])
        enhanced = enhanced_np[:n, 0]
        clean = clean_np[:n, 0]
        row = {
            "file": enhanced_path.name,
            "si_sdr": float(si_sdr(torch.from_numpy(enhanced), torch.from_numpy(clean))),
            "stoi": None,
            "pesq_wb": None,
        }
        if stoi_fn is not None:
            row["stoi"] = float(stoi_fn(clean, enhanced, args.sample_rate, extended=False))
        if pesq_fn is not None:
            row["pesq_wb"] = float(pesq_fn(args.sample_rate, clean, enhanced, "wb"))
        rows.append(row)

    summary = {
        "items": len(rows),
        "mean_si_sdr": sum(r["si_sdr"] for r in rows) / len(rows),
        "mean_stoi": None if rows[0]["stoi"] is None else sum(float(r["stoi"]) for r in rows) / len(rows),
        "mean_pesq_wb": None if rows[0]["pesq_wb"] is None else sum(float(r["pesq_wb"]) for r in rows) / len(rows),
        "stoi_available": stoi_fn is not None,
        "pesq_available": pesq_fn is not None,
    }
    print(json.dumps({"summary": summary, "items": rows}, indent=2))


if __name__ == "__main__":
    main()
