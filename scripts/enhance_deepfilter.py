#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from ha_denoise.audio import read_wav, write_wav
from ha_denoise.features import enhance_with_deep_filter, extract_features, feature_config_from_dict, match_loudness
from ha_denoise.model import TinyDeepFilterTCN
from ha_denoise.spatial import delay_and_sum_beamform


def load_model(checkpoint: str, device: str):
    ckpt = torch.load(checkpoint, map_location=device)
    cfg_d = ckpt["config"]
    cfg = feature_config_from_dict(cfg_d)
    model = TinyDeepFilterTCN(
        int(cfg_d["feature_dim"]),
        int(cfg_d["bands"]),
        int(cfg_d["channels"]),
        int(cfg_d["blocks"]),
        int(cfg_d["kernel_size"]),
        int(cfg_d["df_bins"]),
        int(cfg_d["df_order"]),
        float(cfg_d.get("coef_scale", 1.5)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--loudness-match", action="store_true")
    parser.add_argument("--target-rms-ratio", type=float, default=1.0)
    parser.add_argument("--max-gain-db", type=float, default=8.0)
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint, args.device)
    sr, wav = read_wav(args.input, cfg.sample_rate)
    if wav.shape[1] < 2:
        raise ValueError("input wav must be stereo dual-mic audio")
    mix = torch.from_numpy(wav[:, :2].T).to(args.device)
    with torch.no_grad():
        beamformed, lag = delay_and_sum_beamform(mix, max_lag=8, analysis_samples=cfg.sample_rate // 2)
        feat = extract_features(mix, cfg).transpose(0, 1).unsqueeze(0)
        gain, coef = model(feat)
        enhanced = enhance_with_deep_filter(beamformed, gain.squeeze(0).transpose(0, 1), coef.squeeze(0), cfg)
        if args.loudness_match:
            enhanced, _ = match_loudness(beamformed, enhanced, args.target_rms_ratio, args.max_gain_db)
    print(f"beamform_lag_samples={lag}")
    write_wav(args.output, sr, enhanced.detach().cpu().numpy())


if __name__ == "__main__":
    main()
