# Effect Improvement Notes

This document tracks changes added after the first listening check showed that the baseline can sound over-processed.

## 1. High-SNR Bypass

Problem:

- Some high-SNR samples are already clean.
- The model may still apply suppression and reduce speech naturalness.

Implemented option:

```bash
--high-snr-bypass
--bypass-threshold 0.97
--bypass-width 0.02
```

Supported scripts:

- `scripts/enhance_wav.py`
- `scripts/enhance_streaming.py`
- `scripts/enhance_realtime.py`
- `scripts/evaluate.py`
- `scripts/compare_streaming.py`
- `scripts/compare_realtime.py`
- `scripts/make_listening_eval.py`

Example:

```bash
PYTHONPATH=src python3 scripts/compare_realtime.py \
  --checkpoint runs/arctic_demand/best.pt \
  --data data/arctic_demand_eval \
  --split val \
  --high-snr-bypass
```

The bypass is intentionally disabled by default so old metrics remain reproducible.

Initial listening-sample check:

```text
baseline realtime improvement, 5 samples: 3.726 dB
conservative bypass realtime improvement, 5 samples: 3.688 dB
```

This means the current heuristic is safe to test but not a guaranteed improvement. A learned SNR/noise gate or real high-SNR training data is likely needed for a robust fix.

## 2. Stronger 150 KB-Range Model

The default model uses `channels=112` and has 121,104 parameters.

`scripts/train.py` now supports:

```bash
--channels
--blocks
--kernel-size
```

Recommended stronger configuration:

```bash
PYTHONPATH=src python3 scripts/train.py \
  --data data/arctic_demand \
  --on-the-fly \
  --seconds 2.0 \
  --epochs 60 \
  --batch-size 8 \
  --spatial-features \
  --channels 120 \
  --band-mag-loss-weight 0.1 \
  --out runs/arctic_demand_c120 \
  --device cpu
```

`channels=120` with the 192-dim spatial feature set remains under the 150 KB target.

## 3. IPD And Coherence Features

The old feature set was:

```text
log power mic0: 32
log power mic1: 32
ILD/log power ratio: 32
total: 96
```

For close microphone spacing, ILD can be weak. The optional spatial feature set adds:

```text
IPD cos: 32
IPD sin: 32
band coherence: 32
total: 192
```

Enable it during training:

```bash
--spatial-features
```

Old checkpoints remain compatible because checkpoint config records `spatial_features`.

## 4. Spectral And SI-SDR Training Losses

The default training objective is still mask MSE for reproducibility.

Optional losses:

```bash
--band-mag-loss-weight 0.1
--si-sdr-loss-weight 0.02
```

`--band-mag-loss-weight` adds a band magnitude L1 term. It is cheap and recommended for the next training run.

`--si-sdr-loss-weight` reconstructs waveform inside the training loop and adds negative SI-SDR. It is much slower, so use it only for fine-tuning or small experiments.

## 5. Perceptual Metrics

Added:

```bash
scripts/evaluate_perceptual.py
```

It always reports SI-SDR. If optional packages are installed, it also reports:

- STOI via `pystoi`
- PESQ wideband via `pesq`

Install optional metrics:

```bash
pip install pystoi pesq
```

Example on listening samples:

```bash
PYTHONPATH=src python3 scripts/evaluate_perceptual.py \
  --clean-dir runs/arctic_demand/listening_eval \
  --enhanced-dir runs/arctic_demand/listening_eval \
  --pattern "sample_*_realtime.wav"
```

## 6. Data Upgrade

The most important listening-quality improvement is better data. See `docs/data_upgrade.md`.

Recommended order:

1. Add LibriSpeech or DNS Challenge clean speech.
2. Add MUSAN/DNS noise.
3. Add RIR augmentation.
4. Fine-tune with real hearing-aid/ear-worn dual-mic recordings.
