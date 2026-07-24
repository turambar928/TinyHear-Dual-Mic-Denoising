#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ha_denoise.dataset import WavPairDataset
from ha_denoise.features import FeatureConfig, enhance_with_mask, pad_sequence_batch
from ha_denoise.metrics import si_sdr
from ha_denoise.model import TinyGRUDenoiser, count_parameters


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * valid).sum() / valid.sum().clamp_min(1.0)


def vad_targets(mask: torch.Tensor) -> torch.Tensor:
    speech = torch.clamp((mask.mean(dim=1, keepdim=True) - 0.25) / 0.55, 0.0, 1.0)
    return speech


def vad_loss(pred_vad: torch.Tensor, target_mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    target = vad_targets(target_mask)
    loss = F.binary_cross_entropy(pred_vad.clamp(1e-4, 1.0 - 1e-4), target, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def temporal_gain_loss(gain: torch.Tensor, vad: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if gain.shape[-1] <= 1:
        return gain.new_tensor(0.0)
    diff = torch.abs(gain[:, :, 1:] - gain[:, :, :-1])
    frame_valid = valid[:, :, 1:] * valid[:, :, :-1]
    noise_weight = 1.0 - vad[:, :, 1:].detach()
    return (diff * frame_valid * (0.35 + 0.65 * noise_weight)).sum() / frame_valid.sum().clamp_min(1.0)


def high_snr_preserve_loss(
    gain: torch.Tensor,
    valid: torch.Tensor,
    mix_refs: torch.Tensor,
    clean_refs: torch.Tensor,
    threshold_db: float,
) -> torch.Tensor:
    weights = []
    for i in range(gain.shape[0]):
        score = si_sdr(mix_refs[i].detach(), clean_refs[i].detach())
        weights.append((score >= threshold_db).to(gain.dtype))
    sample_weight = torch.stack(weights).to(gain.device).view(-1, 1, 1)
    weighted_valid = valid * sample_weight
    return (((gain - 1.0) ** 2) * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)


def waveform_l1_loss(gain: torch.Tensor, mix_refs: torch.Tensor, clean_refs: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    losses = []
    for i in range(gain.shape[0]):
        enhanced = enhance_with_mask(mix_refs[i], gain[i].transpose(0, 1), cfg)
        n = min(enhanced.numel(), clean_refs[i].numel())
        losses.append(torch.mean(torch.abs(enhanced[:n] - clean_refs[i, :n])))
    return torch.stack(losses).mean()


def stft_logmag_loss(gain: torch.Tensor, mix_refs: torch.Tensor, clean_refs: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    window = torch.hann_window(cfg.n_fft, device=gain.device, dtype=mix_refs.dtype)
    losses = []
    for i in range(gain.shape[0]):
        enhanced = enhance_with_mask(mix_refs[i], gain[i].transpose(0, 1), cfg)
        n = min(enhanced.numel(), clean_refs[i].numel())
        enh_spec = torch.stft(
            enhanced[:n],
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        clean_spec = torch.stft(
            clean_refs[i, :n],
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        losses.append(torch.mean(torch.abs(torch.log1p(enh_spec.abs()) - torch.log1p(clean_spec.abs()))))
    return torch.stack(losses).mean()


def si_sdr_loss(gain: torch.Tensor, mix_refs: torch.Tensor, clean_refs: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    losses = []
    for i in range(gain.shape[0]):
        enhanced = enhance_with_mask(mix_refs[i], gain[i].transpose(0, 1), cfg)
        n = min(enhanced.numel(), clean_refs[i].numel())
        losses.append(-si_sdr(enhanced[:n], clean_refs[i, :n]) / 20.0)
    return torch.stack(losses).mean()


def run_epoch(
    model: TinyGRUDenoiser,
    loader: DataLoader,
    optimizer,
    device: str,
    cfg: FeatureConfig,
    vad_weight: float,
    temporal_weight: float,
    high_snr_weight: float,
    high_snr_threshold: float,
    waveform_weight: float,
    stft_weight: float,
    sisdr_weight: float,
) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, leave=False):
            feats, masks, valid, mix_refs, clean_refs, _ = batch
            feats = feats.to(device)
            masks = masks.to(device)
            valid = valid.to(device)
            mix_refs = mix_refs.to(device)
            clean_refs = clean_refs.to(device)
            gain, vad = model(feats)
            loss = masked_mse(gain, masks, valid)
            if vad_weight > 0.0:
                loss = loss + vad_weight * vad_loss(vad, masks, valid)
            if temporal_weight > 0.0:
                loss = loss + temporal_weight * temporal_gain_loss(gain, vad, valid)
            if high_snr_weight > 0.0:
                loss = loss + high_snr_weight * high_snr_preserve_loss(
                    gain,
                    valid,
                    mix_refs,
                    clean_refs,
                    high_snr_threshold,
                )
            if waveform_weight > 0.0:
                loss = loss + waveform_weight * waveform_l1_loss(gain, mix_refs, clean_refs, cfg)
            if stft_weight > 0.0:
                loss = loss + stft_weight * stft_logmag_loss(gain, mix_refs, clean_refs, cfg)
            if sisdr_weight > 0.0:
                loss = loss + sisdr_weight * si_sdr_loss(gain, mix_refs, clean_refs, cfg)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total += float(loss.item())
    return total / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--on-the-fly", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--min-gain", type=float, default=0.04)
    parser.add_argument("--vad-loss-weight", type=float, default=0.08)
    parser.add_argument("--temporal-loss-weight", type=float, default=0.20)
    parser.add_argument("--high-snr-preserve-weight", type=float, default=0.18)
    parser.add_argument("--high-snr-threshold", type=float, default=10.0)
    parser.add_argument("--waveform-loss-weight", type=float, default=0.30)
    parser.add_argument("--stft-mag-loss-weight", type=float, default=0.10)
    parser.add_argument("--si-sdr-loss-weight", type=float, default=0.02)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = FeatureConfig(min_gain=args.min_gain, max_gain=1.0, spatial_features=True, spatial_frontend="delay_sum")
    train_ds = WavPairDataset(args.data, "train", cfg, args.seconds, args.on_the_fly, return_audio=True)
    val_ds = WavPairDataset(args.data, "val", cfg, args.seconds, args.on_the_fly, return_audio=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_sequence_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_sequence_batch)
    model = TinyGRUDenoiser(cfg.feature_dim, cfg.bands, args.hidden, args.layers, args.min_gain, 1.0)
    params = count_parameters(model)
    print(f"parameters={params} int8_weight_bytes~={params}")
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            cfg,
            args.vad_loss_weight,
            args.temporal_loss_weight,
            args.high_snr_preserve_weight,
            args.high_snr_threshold,
            args.waveform_loss_weight,
            args.stft_mag_loss_weight,
            args.si_sdr_loss_weight,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            None,
            args.device,
            cfg,
            args.vad_loss_weight,
            args.temporal_loss_weight,
            args.high_snr_preserve_weight,
            args.high_snr_threshold,
            args.waveform_loss_weight,
            args.stft_mag_loss_weight,
            args.si_sdr_loss_weight,
        )
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        state = {
            "model": model.state_dict(),
            "config": {
                "sample_rate": cfg.sample_rate,
                "n_fft": cfg.n_fft,
                "hop_length": cfg.hop_length,
                "bands": cfg.bands,
                "min_gain": cfg.min_gain,
                "max_gain": cfg.max_gain,
                "mask_target": cfg.mask_target,
                "feature_dim": cfg.feature_dim,
                "spatial_features": cfg.spatial_features,
                "spatial_frontend": cfg.spatial_frontend,
                "hidden": args.hidden,
                "layers": args.layers,
                "model_type": "tiny_gru_denoiser",
                "vad_loss_weight": args.vad_loss_weight,
                "temporal_loss_weight": args.temporal_loss_weight,
                "high_snr_preserve_weight": args.high_snr_preserve_weight,
                "high_snr_threshold": args.high_snr_threshold,
                "waveform_loss_weight": args.waveform_loss_weight,
                "stft_mag_loss_weight": args.stft_mag_loss_weight,
                "si_sdr_loss_weight": args.si_sdr_loss_weight,
            },
            "epoch": epoch,
            "val_loss": val_loss,
            "params": params,
        }
        torch.save(state, out / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(state, out / "best.pt")


if __name__ == "__main__":
    main()
