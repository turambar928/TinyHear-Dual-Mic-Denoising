# arctic_wind_demand_tiny_deepfilter_coherence_mwf

This run fine-tunes `runs/arctic_demand_tiny_deepfilter_coherence_mwf_aug2/best.pt` with added real wind-noise data.

## Motivation

The previous `aug2` model improved the original ARCTIC/DEMAND evaluation set, but subjective listening still had strong breath/wind-like residual noise. A dedicated wind-noise evaluation set confirmed the gap:

| Model | Eval set | Mean noisy SI-SDR | Mean enhanced SI-SDR | Mean improvement | Mean output/input RMS |
| --- | --- | ---: | ---: | ---: | ---: |
| aug2 | Zenodo wind eval | 5.39 | 7.80 | +2.41 | 0.92 |
| wind fine-tune | Zenodo wind eval | 5.39 | 9.74 | +4.35 | 0.81 |
| wind fine-tune | original DEMAND eval | 4.33 | 11.39 | +7.06 | 0.72 |

The wind fine-tune improves real wind-noise suppression, but it also reduces the original DEMAND score compared with `aug2`. This is a real tradeoff, not just a metric artifact, so both listening sets are included in the web demo.

## Data

New data source:

- Zenodo Wind Noise Dataset, 478 audio files downloaded into `downloads/wind_noise/wind_noise_dataset`.

Prepared project data:

- `data/wind_noise_zenodo/train/noise`: 1000 wind-noise chunks.
- `data/wind_noise_zenodo/val/noise`: 160 wind-noise chunks.
- `data/arctic_wind_demand_flat`: file-level symlink dataset combining ARCTIC clean speech, DEMAND noise, and Zenodo wind noise.
- `data/arctic_wind_eval`: fixed 160-item wind-only evaluation set.

The raw/prepared datasets are intentionally not tracked in git because they are local training assets. The scripts used to reproduce them are tracked:

```bash
PYTHONPATH=src python3 scripts/download_wind_noise_dataset.py --out downloads/wind_noise
PYTHONPATH=src python3 scripts/prepare_noise_wavs.py \
  --src downloads/wind_noise/wind_noise_dataset \
  --out data/wind_noise_zenodo \
  --train-count 1000 \
  --val-count 160 \
  --seconds 4.0 \
  --sample-rate 16000 \
  --channels 2
```

## Training

```bash
PYTHONPATH=src python3 scripts/train_deepfilter.py \
  --data data/arctic_wind_demand_flat \
  --out runs/arctic_wind_demand_tiny_deepfilter_coherence_mwf \
  --resume-denoiser runs/arctic_demand_tiny_deepfilter_coherence_mwf_aug2/best.pt \
  --epochs 3 \
  --batch-size 8 \
  --seconds 1.0 \
  --on-the-fly \
  --virtual-multiplier 2 \
  --snr-min-db -12 \
  --snr-max-db 8 \
  --noise-mix-prob 0.65 \
  --mic-distance-min-m 0.015 \
  --mic-distance-max-m 0.024 \
  --self-noise-prob 0.45 \
  --self-noise-db -42 \
  --wind-noise-prob 0.15 \
  --wind-noise-db -30 \
  --spatial-frontend coherence_mwf \
  --channels 96 \
  --blocks 8 \
  --kernel-size 5 \
  --df-bins 64 \
  --df-order 3 \
  --waveform-loss-weight 0.62 \
  --stft-mag-loss-weight 0.20 \
  --si-sdr-loss-weight 0.03 \
  --coef-reg-weight 0.002 \
  --residual-noise-loss-weight 0.32 \
  --residual-noise-threshold 0.08 \
  --silence-floor-weight 0.28 \
  --silence-threshold 0.045 \
  --min-gain 0.01 \
  --device cuda
```

Training losses:

| Epoch | Train loss | Val loss |
| ---: | ---: | ---: |
| 1 | 1.8585 | 1.8030 |
| 2 | 1.6803 | 1.6497 |
| 3 | 1.6085 | 1.6305 |

## Listening Sets

- `listening_eval_wind_zenodo`: wind-only fixed eval, best for judging the breath/wind-like residual issue.
- `listening_eval_demand_nomatch`: original ARCTIC/DEMAND eval without loudness matching, best for checking whether general denoising regressed.

The web demo labels are:

- `wind_finetune_on_wind_eval`
- `aug2_baseline_on_wind_eval`
- `wind_finetune_on_demand_eval`
