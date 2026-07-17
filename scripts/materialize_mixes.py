#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from ha_denoise.audio import read_wav, write_wav
from ha_denoise.dataset import crop_or_tile, list_wavs, synthesize_dual_mic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset with split/clean and split/noise wav files.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.data) / args.split
    clean_files = list_wavs(root / "clean")
    noise_files = list_wavs(root / "noise")
    if not clean_files or not noise_files:
        raise FileNotFoundError(f"Need {root}/clean/*.wav and {root}/noise/*.wav")
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    length = int(args.sample_rate * args.seconds)

    for i in range(args.count):
        _, clean = read_wav(clean_files[i % len(clean_files)], args.sample_rate)
        _, noise = read_wav(rng.choice(noise_files), args.sample_rate)
        clean_crop = crop_or_tile(clean[:, 0], length, rng)
        noise_crop = crop_or_tile(noise[:, 0], length, rng)
        mix, clean_pair = synthesize_dual_mic(
            clean_crop,
            noise_crop,
            args.sample_rate,
            snr_db=rng.uniform(-5.0, 15.0),
            noise_angle_deg=rng.uniform(-90.0, 90.0),
            rng=rng,
        )
        write_wav(out / f"mix_{i:04d}.wav", args.sample_rate, mix.T)
        write_wav(out / f"clean_{i:04d}.wav", args.sample_rate, clean_pair.T)


if __name__ == "__main__":
    main()

