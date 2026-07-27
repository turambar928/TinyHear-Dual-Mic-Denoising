# MWF Student 92K

This run is the first deployable teacher-student step after the oracle MWF experiment.

## Purpose

The previous `oracle_mwf_teacher` run proved that a strong two-microphone MWF-style
front-end can remove much more residual noise on the current ARCTIC + DEMAND data.
However, that teacher uses clean speech during evaluation, so it cannot run on a real
device.

This version trains a small student model that only takes noisy two-microphone input
features at inference time. The clean signal is used only during training to compute
the oracle MWF teacher target.

## Model

- Type: causal TCN complex-mask student
- Parameters: 92,018
- Approximate int8 weight size: 92 KB
- Input features: 32 ERB bands x 6 spatial features = 192 dimensions
- Spatial features:
  - mic0 log power
  - mic1 log power
  - ILD
  - IPD cosine
  - IPD sine
  - coherence
- Output: full-bin complex mask with 129 FFT bins x real/imag
- Inference input: noisy dual-mic waveform only

## Training Flow

For each synthetic training item:

1. Synthesize a noisy two-microphone mixture from clean speech and DEMAND noise.
2. Extract dual-mic spatial features from the noisy mixture.
3. Apply the existing `coherence_mwf` spatial frontend to get a stable reference channel.
4. Compute an oracle local 2-mic MWF teacher using noisy dual-mic STFT and clean dual-mic STFT.
5. Convert the teacher waveform into a complex mask target relative to the reference channel.
6. Train the student complex-mask model with a mixed loss:
   - teacher complex-mask MSE
   - teacher waveform L1
   - teacher STFT loss
   - clean STFT loss
   - clean SI-SDR loss

The student therefore learns the direction of the MWF teacher while still being regularized
toward clean speech.

## Commands

Training:

```bash
PYTHONPATH=src python3 scripts/train_mwf_student.py \
  --data data/arctic_demand \
  --out runs/arctic_demand_mwf_student \
  --epochs 8 \
  --batch-size 8 \
  --seconds 1.0 \
  --on-the-fly \
  --channels 80 \
  --blocks 8 \
  --kernel-size 5 \
  --device cuda
```

Evaluation and listening samples:

```bash
PYTHONPATH=src python3 scripts/evaluate_complex_mask.py \
  --checkpoint runs/arctic_demand_mwf_student/best.pt \
  --data data/arctic_demand_eval \
  --split val \
  --max-items 160 \
  --save-audio runs/arctic_demand_mwf_student/eval_audio \
  --save-listening runs/arctic_demand_mwf_student/listening_eval \
  --listening-samples 20 \
  --device cuda \
  --loudness-match \
  --target-rms-ratio 0.95 \
  --max-gain-db 5.0
```

## Results

Fixed 160-item ARCTIC + DEMAND eval:

```text
mean_noisy_si_sdr:       4.331 dB
mean_enhanced_si_sdr:   13.775 dB
mean_si_sdr_improvement: 9.444 dB
mean_output_input_rms_ratio: 0.941
mean_loudness_gain: 1.173
```

## Interpretation

This model is not expected to match the oracle MWF teacher yet. The teacher has direct
access to clean speech and reaches about +16 dB SI-SDR improvement on the same fixed
eval set. The value of this run is that the deployable student now learns from the
strong MWF behavior instead of only fitting clean magnitude or complex masks.

The next improvement should move from predicting a complex mask over a reference channel
to predicting speech/noise covariance masks for a real 2x2 MWF/WF DSP block. That is
closer to the teacher algorithm and should give the student more headroom against
residual airflow-like noise.
