#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from ha_denoise.audio import read_wav, write_wav
from ha_denoise.features import (
    enhance_with_deep_filter,
    extract_features,
    feature_config_from_dict,
    match_loudness,
    stationary_noise_floor_filter,
)
from ha_denoise.model import TinyDeepFilterTCN
from ha_denoise.spatial import apply_spatial_frontend


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
    parser.add_argument(
        "--spatial-frontend-override",
        choices=["delay_sum", "coherence_mwf", "coherence_mwf_smooth"],
        help="Override the checkpoint spatial frontend for artifact/listening experiments.",
    )
    parser.add_argument("--stable-postfilter", action="store_true", help="Apply a conservative stationary-noise post-filter.")
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint, args.device)
    if args.spatial_frontend_override:
        cfg.spatial_frontend = args.spatial_frontend_override
    sr, wav = read_wav(args.input, cfg.sample_rate)
    if wav.shape[1] < 2:
        raise ValueError("input wav must be stereo dual-mic audio")
    mix = torch.from_numpy(wav[:, :2].T).to(args.device)
    with torch.no_grad():
        beamformed, spatial_info = apply_spatial_frontend(mix, cfg, max_lag=8, analysis_samples=cfg.sample_rate // 2)
        feat = extract_features(mix, cfg).transpose(0, 1).unsqueeze(0)
        gain, coef = model(feat)
        band_gain = gain.squeeze(0).transpose(0, 1)
        enhanced = enhance_with_deep_filter(beamformed, band_gain, coef.squeeze(0), cfg)
        if args.stable_postfilter:
            enhanced = stationary_noise_floor_filter(enhanced, band_gain, cfg)
        if args.loudness_match:
            enhanced, _ = match_loudness(beamformed, enhanced, args.target_rms_ratio, args.max_gain_db)
    print(f"spatial_frontend={spatial_info.get('mode')} beamform_lag_samples={spatial_info.get('lag')}")
    write_wav(args.output, sr, enhanced.detach().cpu().numpy())


if __name__ == "__main__":
    main()
