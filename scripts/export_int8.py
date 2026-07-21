#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ha_denoise.model import TinyCausalTCN, count_parameters


def quantize_tensor(x: torch.Tensor) -> tuple[np.ndarray, float]:
    arr = x.detach().cpu().numpy().astype(np.float32)
    scale = float(max(np.max(np.abs(arr)) / 127.0, 1e-8))
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, scale


def c_array(name: str, arr: np.ndarray, c_type: str) -> str:
    flat = arr.reshape(-1)
    values = ", ".join(str(int(v)) for v in flat)
    return f"static const {c_type} {name}[{flat.size}] = {{{values}}};\n"


def sanitize(name: str) -> str:
    return name.replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = TinyCausalTCN(cfg["feature_dim"], cfg["bands"], cfg["channels"], cfg["blocks"], cfg["kernel_size"], float(cfg.get("output_min_gain", 0.0)), float(cfg.get("output_max_gain", 1.0)))
    model.load_state_dict(ckpt["model"])
    model.eval()

    metadata = {"config": cfg, "parameters": count_parameters(model), "layers": []}
    header = [
        "#pragma once\n",
        "#include <stdint.h>\n\n",
        "/* INT8 weights, int32 bias. Scales are stored in model_int8.json. */\n",
    ]
    weight_bytes = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            q_w, w_scale = quantize_tensor(module.weight)
            q_b = module.bias.detach().cpu().numpy().astype(np.float32) if module.bias is not None else np.zeros(module.out_channels)
            # Bias quantization still needs activation scale on device. Store float scale metadata and int32 with weight scale only.
            b_scale = w_scale
            q_b_i32 = np.round(q_b / b_scale).astype(np.int32)
            base = sanitize(name)
            np.save(out / f"{base}_weight_int8.npy", q_w)
            np.save(out / f"{base}_bias_int32.npy", q_b_i32)
            header.append(c_array(f"{base}_weight", q_w, "int8_t"))
            header.append(c_array(f"{base}_bias", q_b_i32, "int32_t"))
            metadata["layers"].append(
                {
                    "name": name,
                    "array_name": base,
                    "weight_shape": list(q_w.shape),
                    "bias_shape": list(q_b_i32.shape),
                    "weight_scale": w_scale,
                    "bias_scale_without_activation": b_scale,
                    "groups": module.groups,
                    "kernel_size": module.kernel_size[0],
                    "dilation": module.dilation[0],
                    "padding": module.padding[0],
                }
            )
            weight_bytes += q_w.nbytes + q_b_i32.nbytes

    (out / "model_int8.h").write_text("\n".join(header), encoding="utf-8")
    (out / "model_int8.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"parameters={metadata['parameters']}")
    print(f"exported_weight_and_bias_bytes={weight_bytes}")
    print(f"int8_weight_bytes_only~={metadata['parameters']}")


if __name__ == "__main__":
    main()
