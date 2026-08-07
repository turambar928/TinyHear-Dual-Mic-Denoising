#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ha_denoise.audio import read_wav
from ha_denoise.metrics import si_sdr


def db_ratio(num: float, den: float, eps: float = 1e-10) -> float:
    return 20.0 * math.log10((num + eps) / (den + eps))


def dbfs(value: float, eps: float = 1e-10) -> float:
    return 20.0 * math.log10(value + eps)


def rms(x: np.ndarray, eps: float = 1e-10) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps))


def frame_rms(x: np.ndarray, frame: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size < frame:
        padded = np.pad(x, (0, frame - x.size))
    else:
        extra = (hop - ((x.size - frame) % hop)) % hop
        padded = np.pad(x, (0, extra))
    starts = np.arange(0, max(1, padded.size - frame + 1), hop)
    values = np.empty(starts.size, dtype=np.float64)
    for i, start in enumerate(starts):
        values[i] = rms(padded[start : start + frame])
    return starts, values


def samples_from_frame_mask(mask: np.ndarray, starts: np.ndarray, length: int, frame: int) -> np.ndarray:
    out = np.zeros(length, dtype=bool)
    for active, start in zip(mask, starts):
        if active:
            out[start : min(length, start + frame)] = True
    return out


def band_rms(x: np.ndarray, sr: int, max_hz: float | None = None, min_hz: float = 0.0) -> float:
    if x.size == 0:
        return 0.0
    spec = np.fft.rfft(x * np.hanning(x.size))
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    keep = freqs >= min_hz
    if max_hz is not None:
        keep &= freqs <= max_hz
    if not np.any(keep):
        return 0.0
    power = np.mean(np.square(np.abs(spec[keep])), dtype=np.float64) / max(1, x.size)
    return float(np.sqrt(power + 1e-10))


def load_metric_lookup(metrics_path: Path | None) -> dict[str, dict]:
    if metrics_path is None or not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = payload.get("items", [])
    return {row.get("file", ""): row for row in rows}


def enhanced_to_mix_name(path: Path) -> str:
    name = path.name
    if name.startswith("deepfilter_"):
        return "mix_" + name.removeprefix("deepfilter_")
    if name.startswith("enhanced_"):
        return "mix_" + name.removeprefix("enhanced_")
    return name


def classify(row: dict) -> str:
    if row["si_sdr_improvement"] < 0.0:
        return "regression"
    if row["speech_preservation_db"] < -4.5:
        return "speech_too_small"
    if row["quiet_noise_reduction_db"] < 3.0:
        return "residual_noise"
    if row["quiet_low_rms_dbfs"] > -34.0 and row["quiet_low_high_ratio_db"] > 3.0:
        return "low_freq_wind"
    if row["quiet_high_rms_dbfs"] > -38.0:
        return "high_freq_hiss"
    return "ok"


def summarize(rows: list[dict]) -> dict:
    counts = Counter(row["failure_mode"] for row in rows)
    keys = [
        "si_sdr_improvement",
        "speech_preservation_db",
        "quiet_noise_reduction_db",
        "quiet_enhanced_rms_dbfs",
        "quiet_low_rms_dbfs",
        "quiet_high_rms_dbfs",
    ]
    summary = {"items": len(rows), "failure_counts": dict(counts)}
    for key in keys:
        values = [float(row[key]) for row in rows]
        summary[f"mean_{key}"] = sum(values) / max(1, len(values))
        summary[f"p10_{key}"] = float(np.percentile(values, 10)) if values else 0.0
        summary[f"p90_{key}"] = float(np.percentile(values, 90)) if values else 0.0
    return summary


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    worst_residual = sorted(rows, key=lambda row: row["quiet_enhanced_rms_dbfs"], reverse=True)[:10]
    worst_speech = sorted(rows, key=lambda row: row["speech_preservation_db"])[:10]
    regressions = [row for row in rows if row["failure_mode"] == "regression"][:10]

    def table(selected: list[dict]) -> str:
        lines = [
            "| file | mode | SI-SDR imp | speech preserve | quiet NR | quiet low | quiet high |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in selected:
            lines.append(
                "| {file} | {failure_mode} | {si_sdr_improvement:.2f} | {speech_preservation_db:.2f} | "
                "{quiet_noise_reduction_db:.2f} | {quiet_low_rms_dbfs:.2f} | {quiet_high_rms_dbfs:.2f} |".format(**row)
            )
        return "\n".join(lines)

    counts = summary["failure_counts"]
    text = f"""# Failure Mode Analysis

## Summary

- Items: `{summary["items"]}`
- Mean SI-SDR improvement: `{summary["mean_si_sdr_improvement"]:.2f} dB`
- Mean speech preservation: `{summary["mean_speech_preservation_db"]:.2f} dB`
- Mean quiet noise reduction: `{summary["mean_quiet_noise_reduction_db"]:.2f} dB`
- Mean quiet enhanced RMS: `{summary["mean_quiet_enhanced_rms_dbfs"]:.2f} dBFS`
- Failure counts: `{json.dumps(counts, ensure_ascii=False)}`

## Interpretation

- `speech_too_small`: enhanced speech-active RMS is more than 4.5 dB below clean speech.
- `residual_noise`: quiet-region noise reduction is below 3 dB.
- `low_freq_wind`: quiet low-frequency residual is high and dominates high-frequency residual.
- `high_freq_hiss`: quiet high-frequency residual is high.
- `regression`: enhanced SI-SDR is lower than noisy SI-SDR.

## Highest Quiet Residual

{table(worst_residual)}

## Most Speech Attenuation

{table(worst_speech)}

## Regressions

{table(regressions) if regressions else "No regression samples in the selected set."}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset root containing split/mix_*.wav and clean_*.wav.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--eval-audio", required=True, help="Directory containing deepfilter_*.wav enhanced files.")
    parser.add_argument("--metrics", help="Optional metrics.json from evaluate_deepfilter.py.")
    parser.add_argument("--out", required=True, help="Output directory for analysis files.")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--frame-ms", type=float, default=32.0)
    parser.add_argument("--hop-ms", type=float, default=16.0)
    parser.add_argument("--active-threshold-ratio", type=float, default=0.08)
    parser.add_argument("--quiet-threshold-ratio", type=float, default=0.025)
    args = parser.parse_args()

    split_dir = Path(args.data) / args.split
    eval_dir = Path(args.eval_audio)
    out_dir = Path(args.out)
    metric_lookup = load_metric_lookup(Path(args.metrics) if args.metrics else None)
    frame = max(32, int(args.sample_rate * args.frame_ms / 1000.0))
    hop = max(16, int(args.sample_rate * args.hop_ms / 1000.0))

    rows = []
    for enhanced_path in sorted(eval_dir.glob("deepfilter_*.wav")):
        mix_name = enhanced_to_mix_name(enhanced_path)
        mix_path = split_dir / mix_name
        clean_path = split_dir / mix_name.replace("mix_", "clean_")
        if not mix_path.exists() or not clean_path.exists():
            continue
        sr, mix_np = read_wav(mix_path, args.sample_rate)
        _, clean_np = read_wav(clean_path, args.sample_rate)
        _, enh_np = read_wav(enhanced_path, args.sample_rate)
        noisy = mix_np[:, 0]
        clean = clean_np[:, 0]
        enhanced = enh_np[:, 0]
        n = min(noisy.size, clean.size, enhanced.size)
        noisy = noisy[:n]
        clean = clean[:n]
        enhanced = enhanced[:n]

        starts, clean_frame_rms = frame_rms(clean, frame, hop)
        ref = float(np.percentile(clean_frame_rms, 95)) + 1e-10
        active_frames = clean_frame_rms >= args.active_threshold_ratio * ref
        quiet_frames = clean_frame_rms <= args.quiet_threshold_ratio * ref
        active_mask = samples_from_frame_mask(active_frames, starts, n, frame)
        quiet_mask = samples_from_frame_mask(quiet_frames, starts, n, frame)
        if active_mask.sum() < frame:
            active_mask[:] = True
        if quiet_mask.sum() < frame:
            quiet_mask = ~active_mask
        if quiet_mask.sum() < frame:
            quiet_mask[:] = True

        active_clean_rms = rms(clean[active_mask])
        active_enh_rms = rms(enhanced[active_mask])
        quiet_noisy_rms = rms(noisy[quiet_mask])
        quiet_enh_rms = rms(enhanced[quiet_mask])
        quiet_enh = enhanced[quiet_mask]
        quiet_low_rms = band_rms(quiet_enh, sr, max_hz=800.0)
        quiet_high_rms = band_rms(quiet_enh, sr, min_hz=2600.0, max_hz=7600.0)

        metric_row = metric_lookup.get(mix_name, {})
        noisy_score = metric_row.get("noisy_si_sdr")
        enhanced_score = metric_row.get("enhanced_si_sdr")
        if noisy_score is None or enhanced_score is None:
            noisy_score = float(si_sdr(torch.from_numpy(noisy), torch.from_numpy(clean)))
            enhanced_score = float(si_sdr(torch.from_numpy(enhanced), torch.from_numpy(clean)))
        row = {
            "file": mix_name,
            "noisy_si_sdr": float(noisy_score),
            "enhanced_si_sdr": float(enhanced_score),
            "si_sdr_improvement": float(enhanced_score) - float(noisy_score),
            "speech_preservation_db": db_ratio(active_enh_rms, active_clean_rms),
            "quiet_noise_reduction_db": db_ratio(quiet_noisy_rms, quiet_enh_rms),
            "quiet_enhanced_rms_dbfs": dbfs(quiet_enh_rms),
            "quiet_low_rms_dbfs": dbfs(quiet_low_rms),
            "quiet_high_rms_dbfs": dbfs(quiet_high_rms),
            "quiet_low_high_ratio_db": db_ratio(quiet_low_rms, quiet_high_rms),
            "active_fraction": float(active_mask.mean()),
            "quiet_fraction": float(quiet_mask.mean()),
        }
        row["failure_mode"] = classify(row)
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No deepfilter_*.wav files matched dataset files under {eval_dir}")

    summary = summarize(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "failure_analysis.json").write_text(json.dumps({"summary": summary, "items": rows}, indent=2), encoding="utf-8")
    write_csv(out_dir / "failure_analysis.csv", rows)
    write_markdown(out_dir / "failure_analysis.md", summary, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
