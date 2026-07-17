# Experiments

## YESNO + DEMAND Baseline

Date: 2026-07-18

Purpose: replace synthetic noise with public multichannel environmental noise that is closer to a dual-mic denoising setting.

Datasets:

- Clean speech: OpenSLR YESNO, used as a small real-speech sanity dataset.
- Noise: DEMAND 16 kHz multichannel noise, environments `DKITCHEN` and `DLIVING`.

Why DEMAND:

- It provides real multichannel environmental noise recordings.
- Channel pairs can be converted into stereo noise chunks and mixed with clean speech.
- It is more relevant to dual-mic spatial denoising than mono synthetic noise.

Local preparation:

```bash
PYTHONPATH=src python scripts/prepare_demand_noise.py \
  --src downloads/demand \
  --out data/yesno_demand \
  --train-count 240 \
  --val-count 60 \
  --seconds 4.0
```

Training:

```bash
PYTHONPATH=src python scripts/train.py \
  --data data/yesno_demand \
  --on-the-fly \
  --seconds 2.0 \
  --epochs 40 \
  --batch-size 8 \
  --out runs/yesno_demand \
  --device cpu
```

Result:

- Model parameters: 121,104.
- INT8 weights: about 121 KB.
- Fixed validation set: 48 materialized mixtures.

Evaluation:

```json
{
  "items": 48,
  "mean_noisy_si_sdr": 5.9568006582558155,
  "mean_enhanced_si_sdr": 11.975179682175318,
  "mean_si_sdr_improvement": 6.018379023919502,
  "mean_mask_mse": 0.045445856638252735
}
```

INT8 fixed-scale reference:

```json
{
  "frames": 16,
  "fixed_scales": true,
  "max_abs_diff": 0.016605496406555176,
  "mean_abs_diff": 0.003191741183400154
}
```

Notes:

- This is still a small baseline because YESNO has limited speakers and vocabulary.
- The next stronger training set should replace YESNO with Mini LibriSpeech/LibriSpeech once network bandwidth allows the download.
- DEMAND can be expanded with more environments such as `OHALLWAY`, `NFIELD`, and `PSTATION`.

