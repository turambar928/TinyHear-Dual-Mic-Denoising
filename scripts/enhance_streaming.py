#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from ha_denoise.audio import read_wav, write_wav
from ha_denoise.features import apply_high_snr_bypass, enhance_with_mask, extract_features, feature_config_from_dict
from ha_denoise.model import TinyCausalTCN
from ha_denoise.streaming import run_streaming_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--high-snr-bypass", action="store_true")
    parser.add_argument("--bypass-threshold", type=float, default=0.97)
    parser.add_argument("--bypass-width", type=float, default=0.02)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg_d = ckpt["config"]
    cfg = feature_config_from_dict(cfg_d)
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
        if args.high_snr_bypass:
            mask = apply_high_snr_bypass(mask, args.bypass_threshold, args.bypass_width)
        enhanced = enhance_with_mask(mix[0], mask, cfg).detach().cpu().numpy()
    write_wav(args.output, sr, enhanced)


if __name__ == "__main__":
    main()
