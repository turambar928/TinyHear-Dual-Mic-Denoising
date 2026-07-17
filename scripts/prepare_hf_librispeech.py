#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import Audio, load_dataset

from ha_denoise.audio import write_wav


def export_split(dataset_name: str, config: str, split: str, out_dir: Path, count: int, sample_rate: int) -> int:
    ds = load_dataset(dataset_name, config, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for item in ds:
        audio = item["audio"]["array"]
        wav = np.asarray(audio, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        write_wav(out_dir / f"{written:06d}.wav", sample_rate, wav)
        written += 1
        if written >= count:
            break
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a small LibriSpeech subset from Hugging Face datasets.")
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-split", default="train.100")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--val-count", type=int, default=80)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    out = Path(args.out)
    train = export_split(args.dataset, args.config, args.train_split, out / "train" / "clean", args.train_count, args.sample_rate)
    val = export_split(args.dataset, args.config, args.val_split, out / "val" / "clean", args.val_count, args.sample_rate)
    print(f"train_clean={train} val_clean={val} out={out}")


if __name__ == "__main__":
    main()

