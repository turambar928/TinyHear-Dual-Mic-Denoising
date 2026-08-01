# Aggressive MWF Teacher DeepFilter

This run is an ablation of oracle local MWF teacher distillation with a high teacher blend.

## Configuration

- Architecture: `TinyDeepFilterTCN`
- Parameters: 137,984
- Initialization: `runs/arctic_wind_demand_tiny_deepfilter_coherence_mwf/best.pt`
- Spatial frontend: `coherence_mwf`
- Teacher: oracle local two-mic MWF
- Local covariance window: 9 STFT frames
- Diagonal loading: 0.01
- Teacher blend: 0.90
- Teacher mask weight: 0.75

## Result

This version is not the recommended checkpoint. It confirms that over-weighting the oracle MWF teacher can hurt the small model:

| Eval set | Noisy SI-SDR | Enhanced SI-SDR | Improvement |
| --- | ---: | ---: | ---: |
| Zenodo wind eval | 5.39 dB | 9.13 dB | +3.74 dB |
| Original DEMAND eval | 4.33 dB | 9.54 dB | +5.21 dB |

The conservative teacher run performs better and should be used for listening comparison.
