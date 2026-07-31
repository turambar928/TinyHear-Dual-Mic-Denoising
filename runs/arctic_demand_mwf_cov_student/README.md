# MWF Covariance Student 92K

This run upgrades the previous teacher-student complex-mask model into a more
MWF-like deployable pipeline.

## Design

The model no longer predicts a complex mask for a single reference channel. Instead,
it predicts two full-bin masks:

- speech mask
- noise mask

Those masks drive a deterministic 2x2 multichannel Wiener filter:

```text
Rss = smoothed speech-mask-weighted covariance of [mic0, mic1]
Rnn = smoothed noise-mask-weighted covariance of [mic0, mic1]
w   = (Rss + Rnn)^-1 Rss u
y   = w^H x
```

where `u = [1, 0]` selects mic0 as the target reference. This keeps the small neural
network responsible for estimating speech/noise structure, while the actual spatial
filtering is handled by a fixed DSP block.

## Model

- Type: causal TCN speech/noise mask estimator
- Parameters: 92,018
- Approximate int8 weight size: 92 KB
- Input features: 32 ERB bands x 6 spatial features = 192 dimensions
- Output: 129-bin speech/noise masks
- Filtering: causal recursive 2x2 MWF covariance solver

## Why This Version Matters

The previous `mwf_student_92k` learned the oracle MWF teacher as a complex mask over
a reference channel. That was deployable, but still left the model responsible for
both spatial filtering and spectral reconstruction.

This version moves spatial filtering back into a real MWF block. That is closer to
the oracle teacher and to hearing-aid literature: the neural network estimates masks,
and the signal-processing frontend performs the multichannel filtering.

## Training Command

```bash
PYTHONPATH=src python3 scripts/train_mwf_cov_student.py \
  --data data/arctic_demand \
  --out runs/arctic_demand_mwf_cov_student \
  --epochs 4 \
  --batch-size 4 \
  --seconds 0.75 \
  --max-train-items 160 \
  --max-val-items 40 \
  --on-the-fly \
  --channels 80 \
  --blocks 8 \
  --kernel-size 5 \
  --device cuda
```

This was intentionally a bounded training run because the differentiable MWF solver
is slower than the direct complex-mask path.

## Evaluation Command

```bash
PYTHONPATH=src python3 scripts/evaluate_mwf_cov_student.py \
  --checkpoint runs/arctic_demand_mwf_cov_student/best.pt \
  --data data/arctic_demand_eval \
  --split val \
  --max-items 160 \
  --save-audio runs/arctic_demand_mwf_cov_student/eval_audio \
  --save-listening runs/arctic_demand_mwf_cov_student/listening_eval \
  --listening-samples 20 \
  --device cuda \
  --loudness-match \
  --target-rms-ratio 0.95 \
  --max-gain-db 5.0
```

## Results

Fixed 160-item ARCTIC + DEMAND eval:

```text
mean_noisy_si_sdr:        4.331 dB
mean_enhanced_si_sdr:    15.676 dB
mean_si_sdr_improvement: 11.345 dB
mean_output_input_rms_ratio: 0.946
mean_loudness_gain: 1.187
```

Compared with the previous `mwf_student_92k`:

```text
mwf_student_92k:      +9.444 dB SI-SDR improvement
mwf_cov_student_92k: +11.345 dB SI-SDR improvement
```

The oracle MWF teacher is still much stronger, but this version closes part of that
gap while staying deployable in principle.

For a more listenable demo, the repository also tracks a stronger post-filtered
variant under `listening_eval_strong`, built from the same checkpoint but with
more aggressive residual suppression.

Strong demo run:

```text
mean_enhanced_si_sdr:    15.118 dB
mean_si_sdr_improvement: 10.787 dB
```

`listening_eval_strong_nomatch` disables noisy-RMS loudness matching. This usually
sounds more like real denoising because residual noise is not amplified back toward
the noisy input level.

## Next Step

The next useful improvement is longer training with more on-the-fly mixtures and a
faster/vectorized MWF training layer. The current result was trained on only 160
training items, so it should not be treated as the final model capacity limit.
