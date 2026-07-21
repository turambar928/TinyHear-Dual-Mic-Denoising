#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ha_denoise.audio import read_wav
from ha_denoise.features import FeatureConfig, extract_features, feature_config_from_dict
from ha_denoise.model import TinyCausalTCN
from ha_denoise.realtime import StreamingDenoiser
from train_gate import TinyGate, pooled_features


LAYER_ORDER = ["stem.0"]
for i in range(8):
    LAYER_ORDER.append(f"tcn.{i}.depthwise")
    LAYER_ORDER.append(f"tcn.{i}.pointwise")
LAYER_ORDER.append("head")


def c_float_array(name: str, values: list[float]) -> str:
    body = ", ".join(f"{v:.9e}f" for v in values)
    return f"static const float {name}[{len(values)}] = {{{body}}};\n"


def c_vector(name: str, arr: np.ndarray) -> str:
    flat = arr.astype(np.float32).reshape(-1)
    body = ", ".join(f"{float(v):.9e}f" for v in flat)
    return f"static const float {name}[{flat.size}] = {{{body}}};\n"


def c_matrix(name: str, arr: np.ndarray) -> str:
    rows, cols = arr.shape
    row_blocks = []
    for row in arr.astype(np.float32):
        row_blocks.append("{" + ", ".join(f"{float(v):.9e}f" for v in row) + "}")
    body = ", ".join(row_blocks)
    return f"static const float {name}[{rows}][{cols}] = {{{body}}};\n"


def c_int_array(name: str, values: np.ndarray, c_type: str = "int32_t") -> str:
    flat = values.reshape(-1)
    body = ", ".join(str(int(v)) for v in flat)
    return f"static const {c_type} {name}[{flat.size}] = {{{body}}};\n"


def load_gate(path: str) -> TinyGate:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    gate = TinyGate(int(cfg["input_dim"]), int(cfg["hidden"]))
    gate.load_state_dict(ckpt["model"])
    gate.eval()
    return gate


def gate_from_prefix_stats(gate: TinyGate, feat_sum: torch.Tensor, feat_sumsq: torch.Tensor, frames: int) -> torch.Tensor:
    denom = max(frames, 1)
    mean = feat_sum / float(denom)
    var = feat_sumsq / float(denom) - mean.square()
    pooled = torch.cat([mean, torch.sqrt(var.clamp_min(1e-8))], dim=0).unsqueeze(0)
    return torch.sigmoid(gate(pooled)).squeeze()


def gated_streaming_process(denoiser: StreamingDenoiser, gate: TinyGate, mix: torch.Tensor, flush: bool = True) -> torch.Tensor:
    denoiser.reset()
    h = denoiser.cfg.hop_length
    samples = mix.shape[1]
    pad = (h - samples % h) % h
    if pad:
        mix = torch.nn.functional.pad(mix, (0, pad))
    feat_sum = torch.zeros(denoiser.cfg.feature_dim)
    feat_sumsq = torch.zeros(denoiser.cfg.feature_dim)
    frames = 0
    hops = []
    total_hops = list(range(0, mix.shape[1], h))
    with torch.no_grad():
        for start in total_hops:
            hops.append(gated_streaming_hop(denoiser, gate, mix[:, start : start + h], feat_sum, feat_sumsq, frames))
            frames += 1
        if flush:
            zero_hop = torch.zeros(2, h)
            for _ in range(denoiser.cfg.n_fft // h):
                hops.append(gated_streaming_hop(denoiser, gate, zero_hop, feat_sum, feat_sumsq, frames))
                frames += 1
    return torch.cat(hops, dim=0)[: samples + (denoiser.cfg.n_fft if flush else 0)]


def gated_streaming_hop(
    denoiser: StreamingDenoiser,
    gate: TinyGate,
    hop: torch.Tensor,
    feat_sum: torch.Tensor,
    feat_sumsq: torch.Tensor,
    frames_seen: int,
) -> torch.Tensor:
    h = denoiser.cfg.hop_length
    denoiser.input_buffer[:, :-h] = denoiser.input_buffer[:, h:].clone()
    denoiser.input_buffer[:, -h:] = hop.to(denoiser.device, denoiser.dtype)

    frame = denoiser.input_buffer * denoiser.window[None, :]
    spec0 = torch.fft.rfft(frame[0], n=denoiser.cfg.n_fft)
    spec1 = torch.fft.rfft(frame[1], n=denoiser.cfg.n_fft)
    feat = denoiser._features_from_spectrum(spec0, spec1).detach().cpu()
    feat_sum += feat
    feat_sumsq += feat.square()
    gate_value = gate_from_prefix_stats(gate, feat_sum, feat_sumsq, frames_seen + 1).to(denoiser.device, denoiser.dtype)

    band_mask = denoiser.model_stream.process_frame(feat.to(denoiser.device, denoiser.dtype))
    enhanced_spec = spec0 * denoiser._mask_to_bins(band_mask)
    enhanced_frame = torch.fft.irfft(enhanced_spec, n=denoiser.cfg.n_fft)
    blended = gate_value * enhanced_frame + (1.0 - gate_value) * frame[0]

    denoiser.output_buffer += blended * denoiser.window
    denoiser.norm_buffer += denoiser.window.square()
    valid = denoiser.norm_buffer[:h] > 1e-6
    out = torch.zeros(h, device=denoiser.device, dtype=denoiser.dtype)
    out[valid] = denoiser.output_buffer[:h][valid] / denoiser.norm_buffer[:h][valid]
    denoiser.output_buffer[:-h] = denoiser.output_buffer[h:].clone()
    denoiser.output_buffer[-h:] = 0.0
    denoiser.norm_buffer[:-h] = denoiser.norm_buffer[h:].clone()
    denoiser.norm_buffer[-h:] = 0.0
    return out.cpu()


def quantize_multiplier(real_scale: float) -> tuple[int, int]:
    if real_scale <= 0.0:
        return 0, 0
    significand, shift = np.frexp(real_scale)
    multiplier = int(round(significand * (1 << 31)))
    if multiplier == (1 << 31):
        multiplier //= 2
        shift += 1
    return multiplier, int(shift)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--gate")
    parser.add_argument("--out-dir", default="c_reference/generated")
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    metadata = json.loads((export_dir / "model_int8.json").read_text(encoding="utf-8"))
    layer_meta = {layer["name"]: layer for layer in metadata["layers"]}
    activation_scales = metadata.get("activation_scales", {})
    missing = [name for name in LAYER_ORDER if name not in activation_scales]
    if missing:
        raise RuntimeError(f"Missing activation scales for: {missing}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg_d = ckpt["config"]
    cfg = feature_config_from_dict(cfg_d)
    model = TinyCausalTCN(cfg_d["feature_dim"], cfg_d["bands"], cfg_d["channels"], cfg_d["blocks"], cfg_d["kernel_size"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    gate = load_gate(args.gate) if args.gate else None

    _, wav = read_wav(args.input_wav, cfg.sample_rate)
    mix = torch.from_numpy(wav[:, :2].T)
    feat = extract_features(mix, cfg).transpose(0, 1)
    feat = feat[:, : args.frames].contiguous()
    with torch.no_grad():
        expected = model(feat.unsqueeze(0)).squeeze(0).contiguous().numpy()
    features_np = feat.numpy()
    if gate is not None:
        valid = torch.ones(1, 1, feat.shape[-1], dtype=feat.dtype)
        with torch.no_grad():
            expected_gate = float(torch.sigmoid(gate(pooled_features(feat.unsqueeze(0), valid))).item())
    else:
        expected_gate = 1.0

    scales_h = [
        "#pragma once\n",
        "#define TINY_TCN_NUM_LAYERS 18\n",
        f"#define TINY_TCN_FEATURE_DIM {cfg_d['feature_dim']}\n",
        f"#define TINY_TCN_CHANNELS {cfg_d['channels']}\n",
        f"#define TINY_TCN_BANDS {cfg_d['bands']}\n",
        f"#define TINY_TCN_BLOCKS {cfg_d['blocks']}\n",
        f"#define TINY_TCN_KERNEL {cfg_d['kernel_size']}\n\n",
        f"#define TINY_TCN_SPATIAL_FEATURES {1 if cfg.spatial_features else 0}\n",
        f"#define TINY_TCN_N_FFT {cfg.n_fft}\n",
        f"#define TINY_TCN_HOP_LENGTH {cfg.hop_length}\n\n",
        c_float_array("kActivationScales", [float(activation_scales[name]) for name in LAYER_ORDER]),
        c_float_array("kWeightScales", [float(layer_meta[name]["weight_scale"]) for name in LAYER_ORDER]),
    ]
    (out_dir / "model_config.h").write_text("".join(scales_h), encoding="utf-8")

    module_map = dict(model.named_modules())
    output_scales: list[float] = []
    for idx, name in enumerate(LAYER_ORDER):
        if idx + 1 < len(LAYER_ORDER):
            output_scales.append(float(activation_scales[LAYER_ORDER[idx + 1]]))
        else:
            output_scales.append(1.0 / 32.0)

    multipliers = []
    shifts = []
    bias_blocks = ["#pragma once\n", "#include <stdint.h>\n\n"]
    for idx, name in enumerate(LAYER_ORDER):
        in_scale = float(activation_scales[name])
        w_scale = float(layer_meta[name]["weight_scale"])
        out_scale = output_scales[idx]
        mult, shift = quantize_multiplier(in_scale * w_scale / out_scale)
        multipliers.append(mult)
        shifts.append(shift)
        bias = module_map[name].bias.detach().cpu().numpy().astype(np.float64)
        bias_q = np.round(bias / (in_scale * w_scale)).astype(np.int32)
        bias_blocks.append(c_int_array(f"kBias_{name.replace('.', '_')}", bias_q))

    residual_multipliers = []
    residual_shifts = []
    for block in range(8):
        residual_scale = float(activation_scales[f"tcn.{block}.depthwise"])
        if block < 7:
            block_out_scale = float(activation_scales[f"tcn.{block + 1}.depthwise"])
        else:
            block_out_scale = float(activation_scales["head"])
        mult, shift = quantize_multiplier(residual_scale / block_out_scale)
        residual_multipliers.append(mult)
        residual_shifts.append(shift)

    hard_mult, hard_shift = quantize_multiplier((1.0 / 32.0) * 0.2 * 32768.0)
    bias_blocks.extend(
        [
            "\n",
            c_int_array("kRequantMultipliers", np.array(multipliers, dtype=np.int32)),
            c_int_array("kRequantShifts", np.array(shifts, dtype=np.int32)),
            c_int_array("kResidualMultipliers", np.array(residual_multipliers, dtype=np.int32)),
            c_int_array("kResidualShifts", np.array(residual_shifts, dtype=np.int32)),
            f"static const int32_t kHardSigmoidMultiplier = {hard_mult};\n",
            f"static const int32_t kHardSigmoidShift = {hard_shift};\n",
        ]
    )
    (out_dir / "model_requant.h").write_text("".join(bias_blocks), encoding="utf-8")

    vectors_h = [
        "#pragma once\n",
        f"#define TEST_FRAMES {features_np.shape[1]}\n",
        f"#define TEST_INPUT_SIZE {features_np.size}\n",
        f"#define TEST_OUTPUT_SIZE {expected.size}\n\n",
        c_vector("kTestInput", features_np),
        c_vector("kExpectedOutput", expected),
        f"static const float kExpectedGate = {expected_gate:.9e}f;\n",
    ]
    (out_dir / "test_vectors.h").write_text("".join(vectors_h), encoding="utf-8")

    realtime_samples = args.frames * cfg.hop_length
    realtime_mix = mix[:, :realtime_samples].contiguous()
    denoiser = StreamingDenoiser(model, cfg)
    with torch.no_grad():
        if gate is not None:
            realtime_expected = gated_streaming_process(denoiser, gate, realtime_mix, flush=True).contiguous().numpy()
        else:
            realtime_expected = denoiser.process(realtime_mix, flush=True).contiguous().numpy()
    realtime_h = [
        "#pragma once\n",
        "#define REALTIME_INPUT_HOPS " + str(args.frames) + "\n",
        "#define REALTIME_FLUSH_HOPS " + str(cfg.n_fft // cfg.hop_length) + "\n",
        "#define REALTIME_TOTAL_HOPS (REALTIME_INPUT_HOPS + REALTIME_FLUSH_HOPS)\n",
        "#define REALTIME_INPUT_SAMPLES " + str(realtime_mix.shape[1]) + "\n",
        "#define REALTIME_OUTPUT_SAMPLES " + str(realtime_expected.size) + "\n\n",
        c_vector("kRealtimeInput", realtime_mix.numpy()),
        c_vector("kRealtimeExpectedOutput", realtime_expected),
    ]
    (out_dir / "realtime_vectors.h").write_text("".join(realtime_h), encoding="utf-8")

    band_h = [
        "#pragma once\n",
        "#define TINY_TCN_FREQ_BINS " + str(cfg.n_fft // 2 + 1) + "\n\n",
        c_matrix("kBandMatrix", denoiser.band_matrix.detach().cpu().numpy()),
    ]
    (out_dir / "band_matrix.h").write_text("".join(band_h), encoding="utf-8")
    print(json.dumps({"frames": int(features_np.shape[1]), "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
