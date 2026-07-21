#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ha_denoise.model import TinyCausalTCN


def quantize_activation(x: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(max(np.max(np.abs(x)) / 127.0, 1e-8))
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale


def conv1d_int32(
    x_q: np.ndarray,
    w_q: np.ndarray,
    b_q: np.ndarray,
    groups: int,
    dilation: int,
    left_pad: int,
) -> np.ndarray:
    in_ch, in_t = x_q.shape
    out_ch, in_per_group, kernel = w_q.shape
    out_t = in_t
    x_pad = np.pad(x_q.astype(np.int32), ((0, 0), (left_pad, 0)))
    y = np.zeros((out_ch, out_t), dtype=np.int32)
    out_per_group = out_ch // groups
    for oc in range(out_ch):
        group = oc // out_per_group
        ic_start = group * in_per_group
        acc_bias = int(b_q[oc])
        for t in range(out_t):
            acc = acc_bias
            for icg in range(in_per_group):
                ic = ic_start + icg
                for k in range(kernel):
                    acc += int(x_pad[ic, t + k * dilation]) * int(w_q[oc, icg, k])
            y[oc, t] = acc
    return y


def layer_weight_scale(meta: dict) -> float:
    return float(meta["weight_scale"])


def load_quant_layer(export_dir: Path, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    base = meta.get("array_name", meta["name"].replace(".", "_"))
    return np.load(export_dir / f"{base}_weight_int8.npy"), np.load(export_dir / f"{base}_bias_int32.npy")


def conv_quant_reference(
    x: np.ndarray,
    export_dir: Path,
    meta: dict,
    float_bias: np.ndarray,
    left_pad: int,
    input_scale: float | None = None,
) -> np.ndarray:
    w_q, _ = load_quant_layer(export_dir, meta)
    if input_scale is None:
        x_q, x_scale = quantize_activation(x)
    else:
        x_scale = float(input_scale)
        x_q = np.clip(np.round(x / x_scale), -127, 127).astype(np.int8)
    w_scale = layer_weight_scale(meta)
    b_q = np.round(float_bias.astype(np.float32) / (x_scale * w_scale)).astype(np.int32)
    y_i32 = conv1d_int32(x_q, w_q, b_q, int(meta["groups"]), int(meta["dilation"]), left_pad)
    return y_i32.astype(np.float32) * (x_scale * w_scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fixed-scales", action="store_true", help="Use activation_scales from model_int8.json.")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    meta = json.loads((export_dir / "model_int8.json").read_text(encoding="utf-8"))
    layer_meta = {layer["name"]: layer for layer in meta["layers"]}
    activation_scales = meta.get("activation_scales", {}) if args.fixed_scales else {}
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = TinyCausalTCN(cfg["feature_dim"], cfg["bands"], cfg["channels"], cfg["blocks"], cfg["kernel_size"], float(cfg.get("output_min_gain", 0.0)), float(cfg.get("output_max_gain", 1.0)))
    model.load_state_dict(ckpt["model"])
    model.eval()

    rng = np.random.default_rng(args.seed)
    x = rng.normal(0.0, 1.0, size=(cfg["feature_dim"], args.frames)).astype(np.float32)
    with torch.no_grad():
        y_float = model(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()

    y = conv_quant_reference(
        x,
        export_dir,
        layer_meta["stem.0"],
        model.stem[0].bias.detach().numpy(),
        left_pad=0,
        input_scale=activation_scales.get("stem.0"),
    )
    y = np.maximum(y, 0.0)

    for i, block in enumerate(model.tcn):
        residual = y
        depth_name = f"tcn.{i}.depthwise"
        point_name = f"tcn.{i}.pointwise"
        depth_meta = layer_meta[depth_name]
        left_pad = (int(depth_meta["kernel_size"]) - 1) * int(depth_meta["dilation"])
        y = conv_quant_reference(
            y,
            export_dir,
            depth_meta,
            block.depthwise.bias.detach().numpy(),
            left_pad=left_pad,
            input_scale=activation_scales.get(depth_name),
        )
        y = np.maximum(y, 0.0)
        y = conv_quant_reference(
            y,
            export_dir,
            layer_meta[point_name],
            block.pointwise.bias.detach().numpy(),
            left_pad=0,
            input_scale=activation_scales.get(point_name),
        )
        y = np.maximum(y, 0.0)
        y = y + residual

    y = conv_quant_reference(
        y,
        export_dir,
        layer_meta["head"],
        model.head.bias.detach().numpy(),
        left_pad=0,
        input_scale=activation_scales.get("head"),
    )
    y = np.clip(y * 0.2 + 0.5, 0.0, 1.0)
    diff = np.abs(y - y_float)
    print(json.dumps({
        "frames": args.frames,
        "fixed_scales": args.fixed_scales,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
