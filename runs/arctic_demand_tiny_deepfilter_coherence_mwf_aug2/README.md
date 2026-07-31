# tiny_deepfilter_coherence_mwf_aug2

This run continues from `runs/arctic_demand_tiny_deepfilter_coherence_mwf/best.pt`.
It keeps the same deployable Tiny DeepFilter TCN family and focuses on reducing residual background noise by making the training distribution harder and closer to a real close-spaced dual-microphone device.

## Why this version exists

Earlier compact models showed that the `tiny_deepfilter_coherence_mwf` line was more useful than the covariance-student branch for subjective listening. The main remaining issue was residual broadband/wind-like background noise. This version does not change the model size target; instead, it fine-tunes the same model with stronger on-the-fly mixtures so the network sees more low-SNR, mixed-noise, near-field dual-mic, self-noise, and low-frequency wind-like cases during training.

## Model

- Model type: `tiny_deepfilter_tcn`
- Parameters: `137984`
- Spatial frontend: `coherence_mwf`
- Channels: `96`
- Blocks: `8`
- Kernel size: `5`
- DeepFilter bins: `64`
- DeepFilter order: `3`
- Coefficient scale: inherited from the resumed checkpoint
- Checkpoint: `best.pt`

The frontend first produces a coherence-weighted MWF-style single-channel signal from the dual-mic input. The Tiny DeepFilter model then predicts multi-frame complex filtering coefficients over the low-frequency STFT bins. This is stronger than a pure magnitude-mask model because it can use neighboring frames and partially adjust complex spectra instead of only scaling the noisy magnitude.

## Training changes

The training script now supports stronger on-the-fly augmentation:

- `--virtual-multiplier`: virtually repeats the training set with new random mixtures.
- `--snr-min-db` / `--snr-max-db`: widens the training SNR range.
- `--noise-mix-prob`: mixes two noise recordings to increase noise variety.
- `--mic-distance-min-m` / `--mic-distance-max-m`: randomizes close-spaced dual-mic geometry.
- `--self-noise-prob` / `--self-noise-db`: adds small independent microphone self-noise.
- `--wind-noise-prob` / `--wind-noise-db`: adds low-frequency random noise to simulate wind/handling rumble.

Training command used:

```bash
PYTHONPATH=src python3 scripts/train_deepfilter.py \
  --data data/arctic_demand \
  --out runs/arctic_demand_tiny_deepfilter_coherence_mwf_aug2 \
  --resume-denoiser runs/arctic_demand_tiny_deepfilter_coherence_mwf/best.pt \
  --epochs 3 \
  --batch-size 8 \
  --seconds 1.0 \
  --on-the-fly \
  --virtual-multiplier 2 \
  --snr-min-db -10 \
  --snr-max-db 8 \
  --noise-mix-prob 0.7 \
  --mic-distance-min-m 0.015 \
  --mic-distance-max-m 0.024 \
  --self-noise-prob 0.6 \
  --self-noise-db -40 \
  --wind-noise-prob 0.5 \
  --wind-noise-db -26 \
  --spatial-frontend coherence_mwf \
  --channels 96 \
  --blocks 8 \
  --kernel-size 5 \
  --df-bins 64 \
  --df-order 3 \
  --waveform-loss-weight 0.6 \
  --stft-mag-loss-weight 0.18 \
  --si-sdr-loss-weight 0.03 \
  --coef-reg-weight 0.002 \
  --residual-noise-loss-weight 0.25 \
  --residual-noise-threshold 0.10 \
  --silence-floor-weight 0.22 \
  --silence-threshold 0.05 \
  --min-gain 0.01 \
  --device cuda
```

## Evaluation

Two listening sets were generated:

- `listening_eval_loud`: uses loudness matching with target RMS ratio `0.90` and max gain `4 dB`.
- `listening_eval_nomatch`: does not apply loudness matching, so it may avoid lifting residual noise.

Both use the same dehiss postfilter:

```text
strength=1.15, low_floor=0.78, high_floor=0.36
```

Summary on `data/arctic_demand_eval/val`, 160 items:

| Set | Mean noisy SI-SDR | Mean enhanced SI-SDR | Mean improvement | Mean output/input RMS |
| --- | ---: | ---: | ---: | ---: |
| loud | 4.33 | 13.35 | +9.02 | 0.89 |
| nomatch | 4.33 | 13.35 | +9.02 | 0.80 |

The loud set is easier for speech volume comparison. The nomatch set is better for checking whether residual background noise is being amplified by loudness recovery.

## Dataset note

This run still trains on the local ARCTIC/DEMAND-style prepared data. For a more reliable subjective improvement, the next data step should add broader speech/noise sources such as DNS Challenge noise, MUSAN, VoiceBank-DEMAND, CHiME-style real noise, and more clean speech from LibriSpeech or VCTK. The current implementation already has on-the-fly augmentation hooks, so new local noise and clean folders can be integrated without changing the model architecture.
