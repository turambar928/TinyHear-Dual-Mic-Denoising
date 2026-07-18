# Listening Evaluation Samples

Date: 2026-07-19

This folder contains local listening samples for the current ARCTIC + DEMAND baseline:

```text
runs/arctic_demand/listening_eval/
```

The directory is intentionally ignored by git because it contains generated wav files. Regenerate it with:

```bash
PYTHONPATH=src python3 scripts/make_listening_eval.py \
  --checkpoint runs/arctic_demand/best.pt \
  --data data/arctic_demand_eval \
  --split val \
  --out runs/arctic_demand/listening_eval \
  --samples 5 \
  --device cpu
```

Each sample has four files:

- `*_noisy.wav`: reference microphone noisy input.
- `*_clean.wav`: clean reference.
- `*_offline.wav`: offline enhanced output.
- `*_realtime.wav`: delay-aligned realtime enhanced output.

The realtime files are delay-aligned for easier A/B listening. The estimated realtime delay is 192 samples, or 12 ms at 16 kHz.

## Summary

```json
{
  "items": 5,
  "mean_noisy_si_sdr": 4.396818375587463,
  "mean_offline_si_sdr": 8.131581318378448,
  "mean_realtime_si_sdr": 8.123062026500701,
  "mean_offline_improvement": 3.734762942790985,
  "mean_realtime_improvement": 3.7262436509132386
}
```

## Samples

| Sample | Source | Noisy SI-SDR | Offline SI-SDR | Realtime SI-SDR | Realtime Improvement |
| --- | --- | ---: | ---: | ---: | ---: |
| `sample_000` | `mix_0110.wav` | -5.029 | -0.694 | -0.690 | 4.339 |
| `sample_001` | `mix_0072.wav` | -0.826 | 10.011 | 9.988 | 10.814 |
| `sample_002` | `mix_0112.wav` | 3.933 | 10.198 | 10.197 | 6.264 |
| `sample_003` | `mix_0147.wav` | 9.268 | 11.470 | 11.449 | 2.180 |
| `sample_004` | `mix_0145.wav` | 14.637 | 9.673 | 9.671 | -4.966 |

## Listening Order

For each sample, listen in this order:

1. `sample_xxx_noisy.wav`
2. `sample_xxx_offline.wav`
3. `sample_xxx_realtime.wav`
4. `sample_xxx_clean.wav`

The first three samples show clear enhancement under harder noisy conditions. `sample_004` is a high-SNR boundary case where enhancement reduces SI-SDR, which is useful for discussing over-processing and the need for speech-presence/noise-condition gating.

## Notes

- These are objective and informal listening samples, not a formal MOS test.
- The samples are selected by noisy SI-SDR quantiles from the fixed validation set, so they cover difficult, medium, and easy cases.
- `runs/arctic_demand/listening_eval/index.json` contains the same metadata in machine-readable form.
