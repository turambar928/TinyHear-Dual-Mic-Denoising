#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path


ZENODO_WIND_URL = "https://zenodo.org/records/6687982/files/wind_noise_dataset.zip?download=1"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public Zenodo wind-noise dataset.")
    parser.add_argument("--out", default="downloads/wind_noise")
    parser.add_argument("--url", default=ZENODO_WIND_URL)
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "wind_noise_dataset.zip"
    extract_dir = out / "wind_noise_dataset"

    if not zip_path.exists():
        run(["curl", "-L", "--fail", "--retry", "4", "--retry-delay", "5", "-o", str(zip_path), args.url])
    else:
        print(f"exists={zip_path}")

    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    else:
        print(f"exists={extract_dir}")

    audio_files = sorted(
        p for p in extract_dir.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    )
    print(f"audio_files={len(audio_files)} root={extract_dir}")

    if not args.keep_zip:
        shutil.rmtree(out / "__unused__", ignore_errors=True)


if __name__ == "__main__":
    main()
