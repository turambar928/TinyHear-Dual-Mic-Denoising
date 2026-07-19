# CMSIS Porting Notes

This document describes the next embedded porting step after the current PC C reference.

## Current State

The repository already has:

- C Q15 streaming model reference: `tiny_tcn_process_frame_q15`.
- C realtime DSP reference: `tiny_realtime_process_hop`.
- Replaceable FFT backend interface: `fft_backend.h`.
- Default PC FFT backend: `fft_backend.c`, using naive DFT/IDFT for numerical clarity.
- Optional CMSIS-DSP FFT backend skeleton: `fft_backend_cmsis.c`.

The default build remains dependency-free:

```bash
make -C c_reference clean run bench
```

## CMSIS-DSP FFT Backend

To build with the CMSIS-DSP backend, provide CMSIS include paths and link sources/libraries from the target SDK:

```bash
make -C c_reference clean run \
  FFT_BACKEND=cmsis \
  CMSIS_CFLAGS="-I/path/to/CMSIS-DSP/Include -I/path/to/CMSIS-Core/Include" \
  CMSIS_LDLIBS="/path/to/libCMSISDSP.a"
```

The backend functions to keep stable are:

```c
void tiny_rfft_forward(const float *time_data, TinyComplex32 *freq_data);
void tiny_rfft_inverse(const TinyComplex32 *freq_data, float *time_data);
```

`fft_backend_cmsis.c` currently maps these functions to `arm_rfft_fast_f32`.

## CMSIS-NN / U55 Model Kernels

The current model reference uses plain C loops:

- depthwise causal conv
- pointwise conv
- requantization
- residual add
- hard sigmoid to Q15

Replacement targets:

- `conv1d_i8_requant`
- `conv1d_i8_requant_frame`
- pointwise 1x1 conv calls inside `tiny_tcn_process_frame_q15`

The numerical contract should remain:

```text
input: int8 activation
weight: int8 weight
accumulate: int32
bias: int32
requant: fixed-point multiplier/shift
output: int8 activation or Q15 final mask
```

For Ethos-U55, the practical route is usually:

1. Export or reimplement the model in a TFLite Micro compatible graph.
2. Use Vela to check operator placement.
3. Keep this C reference as the numerical baseline for per-layer debugging.

## Required Board Measurements

PC benchmark numbers in `docs/performance.md` are not MCU results. On M55/U55, measure:

- cycles per `tiny_tcn_process_frame_q15`
- cycles per `tiny_realtime_process_hop`
- peak SRAM
- flash/code size
- stack high-water mark
- worst-case hop time under 4 ms

Recommended pass criteria:

- `tiny_realtime_process_hop` worst-case time < 4 ms.
- model + DSP state SRAM comfortably below product budget.
- C optimized output remains close to PC C reference on generated vectors.

## Validation Order

1. Build default PC reference and save output:

```bash
make -C c_reference clean run bench
```

2. Build CMSIS FFT backend only, compare `realtime_dsp_*_diff`.

3. Replace model kernels one by one and compare:

- first pointwise conv
- then depthwise conv
- then full TCN block
- finally complete `tiny_tcn_process_frame_q15`

4. Run listening samples again after optimized kernels are integrated.

## Notes

- Keep `fft_backend.c` as the portable reference backend.
- Keep `fft_backend_cmsis.c` target-specific and SDK-dependent.
- Do not remove generated C vectors; they are the fastest way to catch numerical regressions.
