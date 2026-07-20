#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ha_denoise.audio import read_wav
from ha_denoise.features import FeatureConfig, extract_features, feature_config_from_dict
from ha_denoise.model import TinyCausalTCN


def load_model(checkpoint: str, device: str) -> tuple[TinyCausalTCN, FeatureConfig]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg_d = ckpt["config"]
    cfg = feature_config_from_dict(cfg_d)
    model = TinyCausalTCN(cfg_d["feature_dim"], cfg_d["bands"], cfg_d["channels"], cfg_d["blocks"], cfg_d["kernel_size"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def scale_from_values(values: list[np.ndarray], percentile: float) -> tuple[float, float]:
    if not values:
        return 1.0, 0.0
    merged = np.concatenate(values)
    peak = float(np.percentile(merged, percentile))
    max_abs = float(np.max(merged))
    scale = max(peak / 127.0, 1e-8)
    return scale, max_abs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--data", required=True, help="Materialized dataset with split/mix_*.wav.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint, args.device)
    activations: dict[str, list[np.ndarray]] = defaultdict(list)
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            def make_hook(layer_name: str):
                def hook(_module, inputs, _output):
                    x = inputs[0].detach().abs().flatten().cpu().numpy().astype(np.float32)
                    activations[layer_name].append(x)
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))

    split_dir = Path(args.data) / args.split
    mix_files = sorted(split_dir.glob("mix_*.wav"))
    if args.max_items:
        mix_files = mix_files[: args.max_items]
    if not mix_files:
        raise FileNotFoundError(f"No mix_*.wav files under {split_dir}")

    with torch.no_grad():
        for mix_path in tqdm(mix_files):
            _, wav = read_wav(mix_path, cfg.sample_rate)
            mix = torch.from_numpy(wav[:, :2].T).to(args.device)
            feat = extract_features(mix, cfg).transpose(0, 1).unsqueeze(0)
            model(feat)

    for hook in hooks:
        hook.remove()

    activation_scales = {}
    report_layers = []
    for name in sorted(activations):
        scale, max_abs = scale_from_values(activations[name], args.percentile)
        activation_scales[name] = scale
        report_layers.append(
            {
                "name": name,
                "input_scale": scale,
                "observed_max_abs": max_abs,
                "percentile": args.percentile,
            }
        )

    export_dir = Path(args.export_dir)
    json_path = export_dir / "model_int8.json"
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    metadata["activation_scales"] = activation_scales
    metadata["calibration"] = {
        "data": str(Path(args.data)),
        "split": args.split,
        "items": len(mix_files),
        "percentile": args.percentile,
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = {"calibration": metadata["calibration"], "layers": report_layers}
    (export_dir / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(metadata["calibration"], indent=2))


if __name__ == "__main__":
    main()
