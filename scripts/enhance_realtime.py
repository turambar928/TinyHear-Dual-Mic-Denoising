#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from ha_denoise.audio import read_wav, write_wav
from ha_denoise.features import extract_features, feature_config_from_dict, mask_guided_post_filter, match_loudness
from ha_denoise.model import TinyCausalTCN
from ha_denoise.realtime import StreamingDenoiser
from train_gate import TinyGate, pooled_features


def load_gate(checkpoint: str, device: str) -> TinyGate:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = TinyGate(int(cfg["input_dim"]), int(cfg["hidden"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--high-snr-bypass", action="store_true")
    parser.add_argument("--bypass-threshold", type=float, default=0.97)
    parser.add_argument("--bypass-width", type=float, default=0.02)
    parser.add_argument("--gate", help="Optional learned gate checkpoint to mix enhanced output back with input.")
    parser.add_argument("--min-gate", type=float, default=0.0, help="Minimum enhanced-output blend ratio when --gate is used.")
    parser.add_argument("--mask-gamma", type=float, default=1.0, help="Raise predicted masks to this power; >1 suppresses residual noise more.")
    parser.add_argument("--loudness-match", action="store_true", help="Match output RMS toward input RMS with bounded gain.")
    parser.add_argument("--target-rms-ratio", type=float, default=0.95)
    parser.add_argument("--max-gain-db", type=float, default=6.0)
    parser.add_argument("--post-filter", action="store_true", help="Apply a mask-guided spectral post-filter after streaming enhancement.")
    parser.add_argument("--post-filter-strength", type=float, default=0.45)
    parser.add_argument("--post-filter-floor", type=float, default=0.35)
    parser.add_argument("--post-filter-threshold", type=float, default=0.58)
    parser.add_argument("--post-filter-width", type=float, default=0.18)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg_d = ckpt["config"]
    cfg = feature_config_from_dict(cfg_d)
    model = TinyCausalTCN(cfg_d["feature_dim"], cfg_d["bands"], cfg_d["channels"], cfg_d["blocks"], cfg_d["kernel_size"], float(cfg_d.get("output_min_gain", 0.0)), float(cfg_d.get("output_max_gain", 1.0)))
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    sr, wav = read_wav(args.input, cfg.sample_rate)
    if wav.shape[1] < 2:
        raise ValueError("input wav must be stereo dual-mic audio")
    mix = torch.from_numpy(wav[:, :2].T).to(args.device)
    denoiser = StreamingDenoiser(model, cfg, args.high_snr_bypass, args.bypass_threshold, args.bypass_width, args.mask_gamma)
    gate_model = load_gate(args.gate, args.device) if args.gate else None
    enhanced = denoiser.process(mix, flush=True)
    feat = None
    feat_batch = None
    if gate_model is not None or args.post_filter:
        feat = extract_features(mix, cfg)
        feat_batch = feat.transpose(0, 1).unsqueeze(0)
    if gate_model is not None:
        valid = torch.ones(1, 1, feat_batch.shape[-1], device=args.device, dtype=feat_batch.dtype)
        gate_input = pooled_features(feat_batch, valid)
        gate = float(torch.sigmoid(gate_model(gate_input)).item())
        gate = max(min(gate, 1.0), args.min_gate)
        n = min(mix.shape[-1], enhanced.numel())
        enhanced = gate * enhanced[:n] + (1.0 - gate) * mix[0, :n]
    if args.post_filter:
        mask = model(feat_batch).squeeze(0).transpose(0, 1)
        if args.mask_gamma != 1.0:
            mask = torch.pow(torch.clamp(mask, min=1e-4), args.mask_gamma)
        enhanced = mask_guided_post_filter(
            enhanced,
            mask,
            cfg,
            args.post_filter_strength,
            args.post_filter_floor,
            args.post_filter_threshold,
            args.post_filter_width,
        )
    if args.loudness_match:
        enhanced, _ = match_loudness(mix[0], enhanced, args.target_rms_ratio, args.max_gain_db)
    enhanced = enhanced.detach().cpu().numpy()
    write_wav(args.output, sr, enhanced)


if __name__ == "__main__":
    main()
