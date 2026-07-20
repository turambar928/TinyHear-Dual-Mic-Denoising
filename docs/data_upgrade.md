# Data Upgrade Plan

The current recommended baseline uses CMU ARCTIC clean speech and DEMAND multichannel noise. It is good enough for an engineering prototype, but listening quality is limited because it is not real hearing-aid dual-mic data.

## Current Baseline

```text
clean: downloads/cmu_arctic/extracted
noise: downloads/demand
train: data/arctic_demand
eval:  data/arctic_demand_eval
```

Current prepared counts:

```text
train clean: 800
val clean: 160
train noise: about 800
val noise: about 160
fixed eval mixtures: 160
```

## Better Public Data Recipe

Priority order:

1. DNS Challenge clean speech + DNS noise/RIR.
2. LibriSpeech clean speech + MUSAN noise/music + DEMAND multichannel noise.
3. Real device dual-mic recordings for fine-tuning.

The repository already supports generic clean/noise wav preparation:

```bash
PYTHONPATH=src python3 scripts/prepare_wav_dataset.py \
  --clean-root /path/to/librispeech_or_dns_clean_wavs \
  --noise-root /path/to/musan_or_dns_noise_wavs \
  --rir-root /path/to/optional_rir_wavs \
  --out data/strong_public \
  --train-clean 5000 \
  --train-noise 1500 \
  --val-clean 300 \
  --val-noise 150
```

Then add DEMAND stereo noise chunks:

```bash
PYTHONPATH=src python3 scripts/prepare_demand_noise.py \
  --src downloads/demand \
  --out data/strong_public \
  --train-count 1500 \
  --val-count 300
```

Train the stronger 150 KB-range model:

```bash
PYTHONPATH=src python3 scripts/train.py \
  --data data/strong_public \
  --on-the-fly \
  --seconds 2.0 \
  --epochs 60 \
  --batch-size 8 \
  --channels 120 \
  --out runs/strong_public \
  --device cpu
```

Materialize a fixed evaluation set:

```bash
PYTHONPATH=src python3 scripts/materialize_mixes.py \
  --data data/strong_public \
  --split val \
  --out data/strong_public_eval \
  --count 300
```

## Real Dual-Mic Fine-Tuning

For the hearing-aid scenario, the most important improvement is real device data:

```text
target speaker: 0 degree / normal wearing position
noise: front/side/back directions
conditions: quiet, street, kitchen, office, wind, handling noise
labels: close-talk clean mic or controlled playback clean source
```

Recommended split:

```text
train: multiple rooms/noise positions
val: held-out rooms/noise positions
test: held-out speakers and held-out noise scenes
```

## Expected Benefits

- More speakers reduce overfitting to ARCTIC voices.
- More noise categories improve robustness.
- RIR/device data reduce mismatch between synthetic dual-mic mixing and real wearing conditions.
- Real high-SNR examples help train a bypass/gating behavior.
