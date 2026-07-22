from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.signal import fftconvolve
from torch.utils.data import Dataset

from .audio import normalize_peak, read_wav, rms, write_wav
from .features import FeatureConfig, extract_features, make_band_matrix, target_band_mask
from .spatial import delay_and_sum_beamform


def fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    n = np.arange(x.shape[0], dtype=np.float32)
    return np.interp(n - delay_samples, n, x, left=0.0, right=0.0).astype(np.float32)


def crop_or_tile(x: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    if x.shape[0] >= length:
        start = rng.randint(0, x.shape[0] - length)
        return x[start : start + length].astype(np.float32)
    repeats = int(math.ceil(length / max(1, x.shape[0])))
    return np.tile(x, repeats)[:length].astype(np.float32)


def convolve_rir(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    if rir.ndim == 2:
        rir = rir[:, 0]
    rir = rir.astype(np.float32)
    peak = float(np.max(np.abs(rir)) + 1e-8)
    rir = rir / peak
    y = fftconvolve(x.astype(np.float32), rir, mode="full")[: x.shape[0]]
    return y.astype(np.float32)


def synthesize_dual_mic(
    clean: np.ndarray,
    noise: np.ndarray,
    sr: int = 16_000,
    snr_db: float = 0.0,
    mic_distance_m: float = 0.018,
    noise_angle_deg: float = 60.0,
    clean_rir: np.ndarray | None = None,
    noise_rir: np.ndarray | None = None,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or random
    clean = clean[:, 0] if clean.ndim == 2 else clean
    n = min(clean.shape[0], noise.shape[0])
    clean = clean[:n].astype(np.float32)
    noise = noise[:n].astype(np.float32)
    noise_is_stereo = noise.ndim == 2 and noise.shape[1] >= 2
    clean = clean / max(rms(clean), 1e-4) * 0.08
    noise = noise / max(rms(noise), 1e-4)
    noise = noise * (rms(clean) / (10.0 ** (snr_db / 20.0)))

    theta = math.radians(noise_angle_deg)
    delay = mic_distance_m * math.sin(theta) / 343.0 * sr
    if clean_rir is not None:
        if clean_rir.ndim == 2 and clean_rir.shape[1] >= 2:
            clean_0 = convolve_rir(clean, clean_rir[:, 0])
            clean_1 = convolve_rir(clean, clean_rir[:, 1])
        else:
            clean_0 = convolve_rir(clean, clean_rir)
            clean_1 = fractional_delay(clean_0, rng.uniform(-0.15, 0.15)) * rng.uniform(0.95, 1.02)
    else:
        clean_0 = clean
        clean_1 = fractional_delay(clean, rng.uniform(-0.15, 0.15)) * rng.uniform(0.95, 1.02)
    if noise_rir is not None:
        if noise_rir.ndim == 2 and noise_rir.shape[1] >= 2:
            noise_0 = convolve_rir(noise, noise_rir[:, 0])
            noise_1 = convolve_rir(noise, noise_rir[:, 1])
        else:
            noise_0 = convolve_rir(noise, noise_rir)
            noise_1 = fractional_delay(noise_0, delay) * rng.uniform(0.8, 1.05)
    else:
        if noise_is_stereo:
            noise_0 = noise[:, 0]
            noise_1 = noise[:, 1]
        else:
            noise_0 = noise
            noise_1 = fractional_delay(noise, delay) * rng.uniform(0.8, 1.05)
    mix = np.stack([clean_0 + noise_0, clean_1 + noise_1], axis=0)
    clean_pair = np.stack([clean_0, clean_1], axis=0)
    scale = max(float(np.max(np.abs(mix))), float(np.max(np.abs(clean_pair))), 1e-4)
    if scale > 0.98:
        mix /= scale / 0.98
        clean_pair /= scale / 0.98
    return mix.astype(np.float32), clean_pair.astype(np.float32)


def list_wavs(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.wav") if p.is_file()])


class WavPairDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        cfg: FeatureConfig,
        seconds: float = 2.0,
        on_the_fly: bool = False,
        return_audio: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.cfg = cfg
        self.seconds = seconds
        self.on_the_fly = on_the_fly
        self.return_audio = return_audio
        self.length = int(cfg.sample_rate * seconds)
        self.rng = random.Random(1234 if split == "train" else 4321)
        self.band_matrix = make_band_matrix(cfg.n_fft, cfg.bands, cfg.sample_rate)

        split_dir = self.root / split
        if on_the_fly:
            self.clean_files = list_wavs(split_dir / "clean")
            self.noise_files = list_wavs(split_dir / "noise")
            self.rir_files = list_wavs(split_dir / "rir") + list_wavs(self.root / "rir")
            if not self.clean_files or not self.noise_files:
                raise FileNotFoundError(f"Need {split_dir}/clean/*.wav and {split_dir}/noise/*.wav")
            self.items = self.clean_files
        else:
            self.mix_files = list_wavs(split_dir)
            self.mix_files = [p for p in self.mix_files if p.name.startswith("mix_")]
            if not self.mix_files:
                raise FileNotFoundError(f"Need precomputed mix_*.wav under {split_dir}")
            self.items = self.mix_files

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.on_the_fly:
            _, clean = read_wav(self.clean_files[idx % len(self.clean_files)], self.cfg.sample_rate)
            _, noise = read_wav(self.rng.choice(self.noise_files), self.cfg.sample_rate)
            clean = crop_or_tile(clean[:, 0], self.length, self.rng)
            noise_src = noise[:, :2] if noise.shape[1] >= 2 else noise[:, 0]
            noise = crop_or_tile(noise_src, self.length, self.rng)
            clean_rir = None
            noise_rir = None
            if self.rir_files:
                _, clean_rir_wav = read_wav(self.rng.choice(self.rir_files), self.cfg.sample_rate)
                _, noise_rir_wav = read_wav(self.rng.choice(self.rir_files), self.cfg.sample_rate)
                clean_rir = clean_rir_wav
                noise_rir = noise_rir_wav
            mix, clean_pair = synthesize_dual_mic(
                clean,
                noise,
                self.cfg.sample_rate,
                snr_db=self.rng.uniform(-5.0, 15.0),
                noise_angle_deg=self.rng.uniform(-90.0, 90.0),
                clean_rir=clean_rir,
                noise_rir=noise_rir,
                rng=self.rng,
            )
        else:
            mix_path = self.mix_files[idx]
            clean_path = mix_path.with_name(mix_path.name.replace("mix_", "clean_"))
            _, mix_np = read_wav(mix_path, self.cfg.sample_rate)
            _, clean_np = read_wav(clean_path, self.cfg.sample_rate)
            mix = mix_np[:, :2].T
            clean_pair = clean_np[:, :2].T

        mix_t = torch.from_numpy(mix.astype(np.float32))
        clean_t = torch.from_numpy(clean_pair.astype(np.float32))
        beamformed, _ = delay_and_sum_beamform(mix_t, max_lag=8, analysis_samples=self.cfg.sample_rate // 2)
        feat = extract_features(mix_t, self.cfg, self.band_matrix)
        mask = target_band_mask(beamformed, clean_t[0], self.cfg, self.band_matrix)
        t = min(feat.shape[0], mask.shape[0])
        if self.return_audio:
            return feat[:t], mask[:t], beamformed, clean_t[0]
        return feat[:t], mask[:t]


def write_synth_dataset(
    out: str | Path,
    num_train: int,
    num_val: int,
    seconds: float,
    sr: int = 16_000,
) -> None:
    out = Path(out)
    rng = random.Random(2026)
    length = int(sr * seconds)
    for split, count in [("train", num_train), ("val", num_val)]:
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            t = np.arange(length, dtype=np.float32) / sr
            f0 = rng.uniform(90.0, 220.0)
            clean = (
                0.55 * np.sin(2 * np.pi * f0 * t)
                + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
                + 0.12 * np.sin(2 * np.pi * 3 * f0 * t)
            )
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(1.5, 4.0) * t + rng.uniform(0, 2 * np.pi))
            clean = (clean * envelope).astype(np.float32)
            noise = rng.uniform(0.1, 0.4) * np.random.default_rng(i).standard_normal(length).astype(np.float32)
            noise += 0.1 * np.sin(2 * np.pi * rng.uniform(200.0, 3000.0) * t).astype(np.float32)
            mix, clean_pair = synthesize_dual_mic(
                clean,
                noise,
                sr,
                snr_db=rng.uniform(-5.0, 12.0),
                noise_angle_deg=rng.uniform(-90.0, 90.0),
                rng=rng,
            )
            write_wav(split_dir / f"mix_{i:04d}.wav", sr, normalize_peak(mix.T, 0.98))
            write_wav(split_dir / f"clean_{i:04d}.wav", sr, normalize_peak(clean_pair.T, 0.98))
