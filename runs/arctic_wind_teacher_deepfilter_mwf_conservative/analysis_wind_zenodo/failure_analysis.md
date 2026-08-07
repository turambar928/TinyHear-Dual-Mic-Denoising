# Failure Mode Analysis

## Summary

- Items: `160`
- Mean SI-SDR improvement: `4.80 dB`
- Mean speech preservation: `-0.96 dB`
- Mean quiet noise reduction: `15.73 dB`
- Mean quiet enhanced RMS: `-44.32 dBFS`
- Failure counts: `{"ok": 117, "low_freq_wind": 35, "regression": 7, "residual_noise": 1}`

## Interpretation

- `speech_too_small`: enhanced speech-active RMS is more than 4.5 dB below clean speech.
- `residual_noise`: quiet-region noise reduction is below 3 dB.
- `low_freq_wind`: quiet low-frequency residual is high and dominates high-frequency residual.
- `high_freq_hiss`: quiet high-frequency residual is high.
- `regression`: enhanced SI-SDR is lower than noisy SI-SDR.

## Highest Quiet Residual

| file | mode | SI-SDR imp | speech preserve | quiet NR | quiet low | quiet high |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mix_0047.wav | residual_noise | 3.01 | -0.71 | 1.10 | -18.09 | -45.82 |
| mix_0039.wav | low_freq_wind | 6.85 | -2.50 | 8.40 | -19.68 | -53.15 |
| mix_0019.wav | low_freq_wind | 4.11 | -0.62 | 9.44 | -22.38 | -70.75 |
| mix_0081.wav | low_freq_wind | 7.15 | -1.71 | 9.90 | -29.58 | -80.24 |
| mix_0001.wav | low_freq_wind | 7.63 | -1.25 | 11.32 | -26.94 | -70.96 |
| mix_0092.wav | ok | 7.73 | -0.85 | 12.78 | -36.31 | -65.98 |
| mix_0094.wav | low_freq_wind | 7.11 | -0.93 | 8.45 | -22.38 | -66.31 |
| mix_0073.wav | low_freq_wind | 7.17 | -0.75 | 11.30 | -25.69 | -67.41 |
| mix_0015.wav | low_freq_wind | 5.56 | -1.43 | 9.56 | -27.79 | -71.23 |
| mix_0154.wav | low_freq_wind | 6.23 | -1.02 | 11.53 | -27.33 | -64.75 |

## Most Speech Attenuation

| file | mode | SI-SDR imp | speech preserve | quiet NR | quiet low | quiet high |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mix_0039.wav | low_freq_wind | 6.85 | -2.50 | 8.40 | -19.68 | -53.15 |
| mix_0027.wav | low_freq_wind | 7.57 | -2.44 | 14.55 | -30.89 | -68.99 |
| mix_0114.wav | ok | 0.62 | -2.42 | 15.69 | -38.40 | -80.73 |
| mix_0130.wav | ok | 8.43 | -2.21 | 21.72 | -42.34 | -83.94 |
| mix_0020.wav | low_freq_wind | 5.75 | -2.12 | 14.44 | -25.65 | -90.45 |
| mix_0142.wav | low_freq_wind | 6.81 | -2.04 | 19.27 | -32.03 | -66.44 |
| mix_0084.wav | low_freq_wind | 5.15 | -1.86 | 12.06 | -31.39 | -74.83 |
| mix_0118.wav | ok | 1.71 | -1.83 | 20.05 | -52.55 | -73.20 |
| mix_0133.wav | ok | 6.70 | -1.81 | 17.10 | -37.55 | -67.57 |
| mix_0145.wav | ok | 8.84 | -1.81 | 23.46 | -36.45 | -68.14 |

## Regressions

| file | mode | SI-SDR imp | speech preserve | quiet NR | quiet low | quiet high |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mix_0040.wav | regression | -0.10 | -0.95 | 16.91 | -44.66 | -88.88 |
| mix_0070.wav | regression | -1.53 | -0.66 | 11.79 | -51.76 | -69.95 |
| mix_0075.wav | regression | -4.14 | -0.82 | 17.62 | -49.53 | -84.19 |
| mix_0108.wav | regression | -0.99 | -1.24 | 14.82 | -39.86 | -70.61 |
| mix_0122.wav | regression | -1.19 | -1.16 | 12.85 | -40.39 | -68.09 |
| mix_0134.wav | regression | -0.37 | -0.85 | 15.59 | -44.31 | -68.12 |
| mix_0151.wav | regression | -0.03 | -0.53 | 16.49 | -57.13 | -73.88 |
