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

## ARCTIC + Expanded DEMAND Baseline

Date: 2026-07-18

Purpose: replace YESNO with a richer public clean speech set and expand DEMAND noise environments.

Clean speech:

- CMU ARCTIC, speakers `bdl` and `slt`.
- 2,264 utterances downloaded, 800 training clean clips and 160 validation clean clips used.

Noise:

- DEMAND 16 kHz multichannel noise.
- Environments: `DKITCHEN`, `DLIVING`, `OHALLWAY`, `NFIELD`, `OOFFICE`, `PSTATION`.
- Converted into stereo noise chunks with `scripts/prepare_demand_noise.py`.

Why ARCTIC:

- Mini LibriSpeech from OpenSLR was too slow in the current network environment.
- Hugging Face LibriSpeech streaming was unreachable from the current server.
- CMU ARCTIC is still public, real speech, and much richer than YESNO.

Training:

```bash
PYTHONPATH=src python scripts/train.py \
  --data data/arctic_demand \
  --on-the-fly \
  --seconds 2.0 \
  --epochs 20 \
  --batch-size 8 \
  --out runs/arctic_demand \
  --device cpu
```

Fine-tuning/resume:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES= python3 scripts/train.py \
  --data data/arctic_demand \
  --on-the-fly \
  --seconds 2.0 \
  --epochs 20 \
  --batch-size 8 \
  --lr 3e-4 \
  --out runs/arctic_demand \
  --device cpu \
  --resume runs/arctic_demand/best.pt \
  --start-epoch 20
```

Best checkpoint after 40 total epochs:

- Epoch: 40.
- Validation mask MSE during training: 0.880537.

Evaluation:

```json
{
  "items": 160,
  "mean_noisy_si_sdr": 4.331035113846883,
  "mean_enhanced_si_sdr": 9.080238467641175,
  "mean_si_sdr_improvement": 4.749203353794292,
  "mean_mask_mse": 0.12110779838403687
}
```

INT8 fixed-scale reference:

```json
{
  "frames": 16,
  "fixed_scales": true,
  "max_abs_diff": 0.058467328548431396,
  "mean_abs_diff": 0.011820799671113491
}
```

Deployment package:

- INT8 weights only: about 121 KB.
- Exported weights and biases: 126,912 bytes.
- Tracked C reference model header: `runs/arctic_demand/int8/model_int8.h`.
- Activation scale calibration: 100th percentile on 100 fixed validation items.
- C reference test vector, Q15 integer path: max abs diff 0.126340210, mean abs diff 0.014770669.
- C streaming Q15 output matches batch Q15 output exactly on the reference vector.

Python streaming model validation:

```bash
PYTHONPATH=src python3 scripts/compare_streaming.py \
  --checkpoint runs/arctic_demand/best.pt \
  --data data/arctic_demand_eval \
  --split val \
  --save-audio runs/arctic_demand/streaming_eval \
  --device cpu
```

```json
{
  "items": 160,
  "max_mask_max_abs_diff": 2.3245811462402344e-06,
  "mean_mask_mean_abs_diff": 6.453811007833821e-08,
  "mean_waveform_mse": 2.853925897961481e-17,
  "mean_offline_si_sdr": 9.080238467641175,
  "mean_streaming_si_sdr": 9.08023853506893,
  "mean_si_sdr_delta": 6.742775440216065e-08
}
```

Notes:

- This validates the model state cache for frame-by-frame inference.
- The feature and reconstruction code intentionally reuse the existing 256 FFT / 64 hop STFT path, so the measured difference isolates model streaming behavior instead of DSP boundary handling.

Comparison:

| Experiment | Clean speech | DEMAND envs | Eval items | SI-SDR improvement | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| YESNO + DEMAND | YESNO, 60 clips | 2 | 48 | 6.02 dB | Easier, limited speech diversity |
| ARCTIC + DEMAND | CMU ARCTIC, 2 speakers | 6 | 160 | 4.75 dB | Harder and more representative |

The ARCTIC baseline is the recommended current baseline despite lower SI-SDR improvement because it uses a larger and more diverse clean speech distribution and a broader noise set.
