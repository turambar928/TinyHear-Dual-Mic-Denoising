# Project Roadmap

## Completed

- Public-data training baseline with CMU ARCTIC + DEMAND.
- 121 KB INT8 TinyCausalTCN.
- Offline SI-SDR evaluation.
- Python model streaming validation.
- Python full realtime DSP validation.
- INT8 export and activation calibration.
- C Q15 streaming model reference.
- C full realtime DSP reference.
- Replaceable FFT backend interface.
- PC-side benchmark and SRAM estimate.
- Listening sample generation workflow.
- Project report and presentation outline.

## Next Engineering Tasks

1. Integrate CMSIS-DSP FFT backend on the target SDK.
2. Replace reference conv loops with CMSIS-NN or U55-compatible kernels.
3. Add board-side cycle and SRAM logging.
4. Build an automated PC-vs-board vector comparison flow.
5. Collect real device dual-mic recordings and fine-tune.
6. Add systematic listening evaluation, such as ABX/MOS-style sheets.

## Research Improvements

- Per-channel weight quantization.
- Quantization-aware training.
- Speech-presence or high-SNR bypass gate to reduce over-processing.
- More diverse clean speech and noise data.
- Real HRTF or device RIR augmentation.
