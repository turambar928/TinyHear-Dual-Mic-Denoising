#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def list_wavs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.wav") if p.is_file())


def copy_subset(files: list[Path], out_dir: Path, count: int, seed: int) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = files[:]
    rng.shuffle(chosen)
    chosen = chosen[: min(count, len(chosen))]
    for i, src in enumerate(chosen):
        dst = out_dir / f"{i:06d}_{src.stem[:40]}.wav"
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the train/val clean/noise folder layout from downloaded public wav datasets."
    )
    parser.add_argument("--clean-root", required=True, help="Directory containing clean speech wav files.")
    parser.add_argument("--noise-root", required=True, help="Directory containing noise wav files.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-clean", type=int, default=5000)
    parser.add_argument("--train-noise", type=int, default=1000)
    parser.add_argument("--val-clean", type=int, default=200)
    parser.add_argument("--val-noise", type=int, default=100)
    parser.add_argument("--rir-root", help="Optional directory containing mono or stereo RIR wav files.")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    clean_files = list_wavs(Path(args.clean_root))
    noise_files = list_wavs(Path(args.noise_root))
    if not clean_files:
        raise FileNotFoundError(f"No wav files under {args.clean_root}")
    if not noise_files:
        raise FileNotFoundError(f"No wav files under {args.noise_root}")

    out = Path(args.out)
    copy_subset(clean_files, out / "train" / "clean", args.train_clean, args.seed)
    copy_subset(noise_files, out / "train" / "noise", args.train_noise, args.seed + 1)
    copy_subset(clean_files, out / "val" / "clean", args.val_clean, args.seed + 2)
    copy_subset(noise_files, out / "val" / "noise", args.val_noise, args.seed + 3)

    if args.rir_root:
        rir_files = list_wavs(Path(args.rir_root))
        copy_subset(rir_files, out / "rir", len(rir_files), args.seed + 4)

    print(f"clean_files={len(clean_files)} noise_files={len(noise_files)}")
    print(f"dataset={out}")


if __name__ == "__main__":
    main()

