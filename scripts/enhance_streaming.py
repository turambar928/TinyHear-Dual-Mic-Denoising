#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from ha_denoise.audio import read_wav, write_wav
from ha_denoise.features import FeatureConfig, enhance_with_mask, extract_features
from ha_denoise.model import TinyCausalTCN
from ha_denoise.streaming import run_streaming_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg_d = ckpt["config"]
    cfg = FeatureConfig(cfg_d["sample_rate"], cfg_d["n_fft"], cfg_d["hop_length"], cfg_d["bands"])
    model = TinyCausalTCN(cfg_d["feature_dim"], cfg_d["bands"], cfg_d["channels"], cfg_d["blocks"], cfg_d["kernel_size"])
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    sr, wav = read_wav(args.input, cfg.sample_rate)
    if wav.shape[1] < 2:
        raise ValueError("input wav must be stereo dual-mic audio")
    mix = torch.from_numpy(wav[:, :2].T).to(args.device)
    with torch.no_grad():
        features = extract_features(mix, cfg)
        mask = run_streaming_model(model, features)
        enhanced = enhance_with_mask(mix[0], mask, cfg).detach().cpu().numpy()
    write_wav(args.output, sr, enhanced)


if __name__ == "__main__":
    main()
