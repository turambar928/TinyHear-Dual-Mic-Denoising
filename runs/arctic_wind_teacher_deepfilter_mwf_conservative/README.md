# Conservative MWF Teacher DeepFilter

This run keeps the deployable `TinyDeepFilterTCN` architecture and fine-tunes it on the ARCTIC speech + DEMAND + Zenodo wind-noise mixture set.

## Purpose

The previous wind fine-tune improved wind-noise suppression, but listening tests still had audible low-frequency airflow and residual background noise. Oracle local MWF on the same wind eval set reached a much higher upper bound, so this run distills part of that teacher behavior into the tiny model without replacing the model with a non-deployable oracle front end.

## Model

- Architecture: `TinyDeepFilterTCN`
- Parameters: 137,984
- Spatial frontend: `coherence_mwf`
- Deep filtering: 64 bins, order 3
- Initialization: `runs/arctic_wind_demand_tiny_deepfilter_coherence_mwf/best.pt`
- Checkpoint: `best.pt`

## Teacher Target

The teacher is an oracle local two-mic MWF computed from the mixture pair and clean pair during training:

- Local covariance window: 9 STFT frames
- Diagonal loading: 0.01
- Causal teacher: false
- Teacher blend: 0.25
- Teacher mask weight: 0.35

The conservative blend is intentional. A stronger teacher run with `teacher_blend=0.90` reduced robustness and lowered SI-SDR on both wind and DEMAND evals. This run treats MWF as a soft hint while keeping clean speech as the main target.

## Training

```bash
PYTHONPATH=src python3 scripts/train_deepfilter.py \
  --data data/arctic_wind_demand_flat \
  --out runs/arctic_wind_teacher_deepfilter_mwf_conservative \
  --resume runs/arctic_wind_demand_tiny_deepfilter_coherence_mwf/best.pt \
  --reset-best-on-resume \
  --epochs 4 \
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
  --wind-noise-prob 0.10 \
  --wind-noise-db -32 \
  --spatial-frontend coherence_mwf \
  --teacher-mwf \
  --teacher-mask-weight 0.35 \
  --teacher-blend 0.25 \
  --oracle-window 9 \
  --diagonal-loading 0.01 \
  --channels 96 \
  --blocks 8 \
  --kernel-size 5 \
  --df-bins 64 \
  --df-order 3 \
  --waveform-loss-weight 0.70 \
  --stft-mag-loss-weight 0.22 \
  --si-sdr-loss-weight 0.04 \
  --coef-reg-weight 0.002 \
  --residual-noise-loss-weight 0.34 \
  --residual-noise-threshold 0.075 \
  --silence-floor-weight 0.28 \
  --silence-threshold 0.04 \
  --min-gain 0.005 \
  --lr 1e-4 \
  --device cuda
```

## Results

Evaluation uses the same dehiss postfilter settings as the previous wind fine-tune.

| Eval set | Noisy SI-SDR | Enhanced SI-SDR | Improvement |
| --- | ---: | ---: | ---: |
| Zenodo wind eval | 5.39 dB | 10.19 dB | +4.80 dB |
| Original DEMAND eval | 4.33 dB | 10.96 dB | +6.63 dB |

Compared with `runs/arctic_wind_demand_tiny_deepfilter_coherence_mwf`, this run improves wind eval from `+4.35 dB` to `+4.80 dB`, while DEMAND eval drops from `+7.06 dB` to `+6.63 dB`. The recommended listening target is the wind eval demo first.

## Listening Demo

The web demo labels are:

- `teacher_deepfilter_conservative_on_wind_eval`
- `teacher_deepfilter_conservative_on_demand_eval`

