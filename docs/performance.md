# Performance And Memory Notes

Date: 2026-07-19

This page records the current PC-side C reference benchmark. It is not a Cortex-M55/U55 measurement. The goal is to separate model cost from DSP reference cost and to keep a reproducible baseline before replacing reference kernels with CMSIS-DSP/CMSIS-NN.

## Command

```bash
make -C c_reference clean bench
```

## Result

Host: current Linux development server, compiled with `cc -std=c99 -O2 -Wall -Wextra -Werror`.

```text
model_frames=3200
model_total_ms=854.433
model_ms_per_frame=0.267010
model_realtime_budget_ms=4.000000
model_realtime_ratio=0.066753
dsp_hops=400
dsp_total_ms=1011.967
dsp_ms_per_hop=2.529917
dsp_realtime_budget_ms=4.000000
dsp_realtime_ratio=0.632479
sizeof_TinyTcnState=13444
sizeof_TinyRealtimeDspState=17540
sizeof_model_frame_input=384
sizeof_model_frame_output_q15=64
```

## Interpretation

- The Q15 streaming model reference is about 0.27 ms per 4 ms frame on this host.
- The full realtime DSP reference is about 2.53 ms per 4 ms hop on this host.
- The C realtime DSP reference currently uses naive DFT/IDFT for clarity and portability. This is a numerical reference, not the intended embedded FFT implementation.
- `TinyTcnState` is about 13.1 KB after sizing each block history to `(kernel_size - 1) * dilation`.
- `TinyRealtimeDspState` is about 17.1 KB including model history, input ring buffer, output overlap-add buffer, and normalization buffer.

## Embedded Replacement Plan

Replace these reference parts before claiming MCU performance:

- `tiny_rfft_forward` in `c_reference/fft_backend.c` -> CMSIS-DSP RFFT.
- `tiny_rfft_inverse` in `c_reference/fft_backend.c` -> CMSIS-DSP inverse RFFT.
- `tiny_tcn_process_frame_q15` conv loops -> CMSIS-NN or vendor optimized int8 kernels.
- Keep `TinyTcnState` history arrays sized per block; avoid regressing to fixed maximum history rows.

The current C reference should remain as the numerical baseline for validating optimized kernels.
