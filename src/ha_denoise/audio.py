from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def read_wav(path: str | Path, target_sr: int = 16_000) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    if data.dtype.kind in {"i", "u"}:
        max_abs = np.iinfo(data.dtype).max
        wav = data.astype(np.float32) / max_abs
    else:
        wav = data.astype(np.float32)
    if wav.ndim == 1:
        wav = wav[:, None]
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // gcd, sr // gcd, axis=0).astype(np.float32)
        sr = target_sr
    wav = np.clip(wav, -1.0, 1.0)
    return sr, wav


def write_wav(path: str | Path, sr: int, wav: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    wav_i16 = (wav * 32767.0).astype(np.int16)
    wavfile.write(path, sr, wav_i16)


def rms(x: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps))


def normalize_peak(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(x)) + 1e-8)
    return (x * min(1.0, peak / max_abs)).astype(np.float32)

