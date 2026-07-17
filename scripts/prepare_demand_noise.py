#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from ha_denoise.audio import read_wav, write_wav


def find_channel_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for directory in sorted([p for p in root.rglob("*") if p.is_dir()]):
        wavs = sorted(directory.glob("*.wav"))
        if len(wavs) >= 2:
            pairs.append((wavs[0], wavs[1]))
    if not pairs:
        wavs = sorted(root.rglob("*.wav"))
        for i in range(0, len(wavs) - 1, 2):
            pairs.append((wavs[i], wavs[i + 1]))
    return pairs


def write_segments(
    pairs: list[tuple[Path, Path]],
    out_dir: Path,
    count: int,
    seconds: float,
    sample_rate: int,
    rng: random.Random,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    length = int(seconds * sample_rate)
    for i in range(count):
        ch0_path, ch1_path = pairs[i % len(pairs)]
        _, ch0 = read_wav(ch0_path, sample_rate)
        _, ch1 = read_wav(ch1_path, sample_rate)
        n = min(ch0.shape[0], ch1.shape[0])
        if n < length:
            reps = int(np.ceil(length / max(1, n)))
            x0 = np.tile(ch0[:n, 0], reps)[:length]
            x1 = np.tile(ch1[:n, 0], reps)[:length]
        else:
            start = rng.randint(0, n - length)
            x0 = ch0[start : start + length, 0]
            x1 = ch1[start : start + length, 0]
        stereo = np.stack([x0, x1], axis=1).astype(np.float32)
        peak = float(np.max(np.abs(stereo)) + 1e-8)
        write_wav(out_dir / f"{i:06d}.wav", sample_rate, stereo / peak * 0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DEMAND multichannel noise into stereo noise chunks.")
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    pairs = find_channel_pairs(Path(args.src))
    if not pairs:
        raise FileNotFoundError(f"No channel wav pairs found under {args.src}")
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    out = Path(args.out)
    write_segments(pairs, out / "train" / "noise", args.train_count, args.seconds, args.sample_rate, rng)
    write_segments(pairs, out / "val" / "noise", args.val_count, args.seconds, args.sample_rate, rng)
    print(f"channel_pairs={len(pairs)} train_noise={args.train_count} val_noise={args.val_count} out={out}")


if __name__ == "__main__":
    main()

