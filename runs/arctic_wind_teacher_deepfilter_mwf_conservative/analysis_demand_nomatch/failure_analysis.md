# Failure Mode Analysis

## Summary

- Items: `160`
- Mean SI-SDR improvement: `6.63 dB`
- Mean speech preservation: `-1.62 dB`
- Mean quiet noise reduction: `21.61 dB`
- Mean quiet enhanced RMS: `-48.26 dBFS`
- Failure counts: `{"regression": 13, "ok": 117, "low_freq_wind": 26, "residual_noise": 1, "speech_too_small": 3}`

## Interpretation

- `speech_too_small`: enhanced speech-active RMS is more than 4.5 dB below clean speech.
- `residual_noise`: quiet-region noise reduction is below 3 dB.
- `low_freq_wind`: quiet low-frequency residual is high and dominates high-frequency residual.
- `high_freq_hiss`: quiet high-frequency residual is high.
- `regression`: enhanced SI-SDR is lower than noisy SI-SDR.

## Highest Quiet Residual

| file | mode | SI-SDR imp | speech preserve | quiet NR | quiet low | quiet high |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mix_0047.wav | residual_noise | 3.32 | -1.06 | 1.47 | -18.07 | -45.72 |
| mix_0103.wav | low_freq_wind | 5.59 | -3.35 | 10.17 | -22.11 | -53.48 |
| mix_0146.wav | low_freq_wind | 4.48 | -1.64 | 6.95 | -21.80 | -50.26 |
| mix_0102.wav | low_freq_wind | 5.95 | -1.77 | 8.94 | -22.23 | -58.63 |
| mix_0080.wav | low_freq_wind | 6.96 | -1.28 | 8.24 | -23.27 | -52.67 |
| mix_0036.wav | low_freq_wind | 6.33 | -1.84 | 10.11 | -22.01 | -58.43 |
| mix_0110.wav | low_freq_wind | 6.31 | -2.60 | 13.68 | -24.39 | -50.65 |
| mix_0011.wav | low_freq_wind | 4.92 | -1.69 | 6.86 | -24.94 | -63.79 |
| mix_0020.wav | low_freq_wind | 3.30 | -3.65 | 11.30 | -26.82 | -53.19 |
| mix_0066.wav | low_freq_wind | 4.50 | -1.54 | 6.92 | -25.20 | -53.09 |

## Most Speech Attenuation

| file | mode | SI-SDR imp | speech preserve | quiet NR | quiet low | quiet high |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mix_0090.wav | speech_too_small | 2.01 | -8.60 | 23.18 | -37.31 | -44.29 |
| mix_0130.wav | speech_too_small | 2.36 | -7.15 | 14.88 | -27.75 | -52.74 |
| mix_0069.wav | speech_too_small | 5.93 | -4.59 | 26.46 | -46.09 | -52.74 |
| mix_0010.wav | regression | -3.48 | -4.42 | 20.60 | -45.19 | -52.70 |
| mix_0089.wav | regression | -1.68 | -3.77 | 17.58 | -56.65 | -55.34 |
| mix_0020.wav | low_freq_wind | 3.30 | -3.65 | 11.30 | -26.82 | -53.19 |
| mix_0133.wav | regression | -3.79 | -3.57 | 16.96 | -57.34 | -57.66 |
| mix_0057.wav | ok | 3.69 | -3.55 | 17.14 | -37.75 | -53.99 |
| mix_0099.wav | ok | 0.61 | -3.41 | 21.87 | -53.02 | -57.35 |
| mix_0053.wav | low_freq_wind | 6.10 | -3.39 | 17.93 | -33.66 | -55.23 |

## Regressions

| file | mode | SI-SDR imp | speech preserve | quiet NR | quiet low | quiet high |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mix_0000.wav | regression | -2.68 | -2.39 | 25.16 | -56.95 | -64.21 |
| mix_0010.wav | regression | -3.48 | -4.42 | 20.60 | -45.19 | -52.70 |
| mix_0031.wav | regression | -0.47 | -1.47 | 5.29 | -33.11 | -59.43 |
| mix_0039.wav | regression | -0.70 | -1.78 | 12.97 | -41.34 | -68.38 |
| mix_0040.wav | regression | -2.60 | -1.15 | 18.23 | -58.85 | -69.80 |
| mix_0075.wav | regression | -3.32 | -0.82 | 29.69 | -57.01 | -79.74 |
| mix_0085.wav | regression | -0.07 | -2.70 | 21.18 | -52.55 | -54.07 |
| mix_0089.wav | regression | -1.68 | -3.77 | 17.58 | -56.65 | -55.34 |
| mix_0105.wav | regression | -4.70 | -3.18 | 21.36 | -55.16 | -65.32 |
| mix_0133.wav | regression | -3.79 | -3.57 | 16.96 | -57.34 | -57.66 |
