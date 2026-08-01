#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ha_denoise.audio import read_wav, write_wav


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}


def decode_to_wav(src: Path, sample_rate: int, channels: int) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            str(tmp_path),
        ],
        check=True,
    )
    return tmp_path


def crop_or_tile(x: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    if x.shape[0] >= length:
        start = rng.randint(0, x.shape[0] - length)
        return x[start : start + length]
    reps = int(np.ceil(length / max(1, x.shape[0])))
    return np.tile(x, (reps, 1))[:length]


def load_audio(src: Path, sample_rate: int, channels: int) -> np.ndarray:
    tmp_path: Path | None = None
    try:
        tmp_path = decode_to_wav(src, sample_rate, channels)
        _, x = read_wav(tmp_path, sample_rate)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1 and channels == 2:
        x = np.repeat(x, 2, axis=1)
    return x[:, :channels].astype(np.float32)


def write_split(
    files: list[Path],
    out_dir: Path,
    count: int,
    seconds: float,
    sample_rate: int,
    channels: int,
    rng: random.Random,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    length = int(seconds * sample_rate)
    for i in range(count):
        src = files[i % len(files)]
        x = load_audio(src, sample_rate, channels)
        if x.shape[0] < max(1, sample_rate // 4):
            continue
        seg = crop_or_tile(x, length, rng)
        peak = float(np.max(np.abs(seg)) + 1e-8)
        write_wav(out_dir / f"{i:06d}.wav", sample_rate, seg / peak * 0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a generic audio tree into train/val noise chunks.")
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--val-count", type=int, default=160)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    files = sorted(p for p in Path(args.src).rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No audio files found under {args.src}")
    rng = random.Random(args.seed)
    rng.shuffle(files)
    out = Path(args.out)
    write_split(files, out / "train" / "noise", args.train_count, args.seconds, args.sample_rate, args.channels, rng)
    write_split(files, out / "val" / "noise", args.val_count, args.seconds, args.sample_rate, args.channels, rng)
    print(f"source_files={len(files)} train_noise={args.train_count} val_noise={args.val_count} out={out}")


if __name__ == "__main__":
    main()
