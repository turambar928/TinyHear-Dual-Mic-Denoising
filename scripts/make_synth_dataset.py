#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ha_denoise.dataset import write_synth_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--num-train", type=int, default=200)
    parser.add_argument("--num-val", type=int, default=20)
    args = parser.parse_args()
    write_synth_dataset(args.out, args.num_train, args.num_val, args.seconds)


if __name__ == "__main__":
    main()

