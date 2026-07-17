#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path


def convert_flac(src: Path, dst: Path, sample_rate: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(dst),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a LibriSpeech/Mini LibriSpeech tree from FLAC to wav.")
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--val-count", type=int, default=80)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    files = sorted(Path(args.src).rglob("*.flac"))
    if not files:
        raise FileNotFoundError(f"No .flac files under {args.src}")
    rng = random.Random(args.seed)
    rng.shuffle(files)
    train = files[: min(args.train_count, len(files))]
    val = files[len(train) : len(train) + min(args.val_count, max(0, len(files) - len(train)))]
    out = Path(args.out)

    for split, chosen in [("train", train), ("val", val)]:
        for i, src in enumerate(chosen):
            convert_flac(src, out / split / "clean" / f"{i:06d}.wav", args.sample_rate)
    print(f"train_clean={len(train)} val_clean={len(val)} out={out}")


if __name__ == "__main__":
    main()

