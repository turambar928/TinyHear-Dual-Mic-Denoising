#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ha_denoise.dataset import WavPairDataset
from ha_denoise.features import (
    FeatureConfig,
    enhance_with_complex_mask,
    extract_features,
    pad_sequence_batch,
    stft,
    target_complex_mask,
)
from ha_denoise.metrics import si_sdr
from ha_denoise.model import TinyComplexMaskTCN, count_parameters


def complex_mask_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    frames = min(pred.shape[1], target.shape[1], valid.shape[-1])
    pred = pred[:, :frames]
    target = target[:, :frames].to(pred.device)
    valid_t = valid[:, :, :frames].transpose(1, 2).unsqueeze(-1).to(pred.device)
    return ((pred - target) ** 2 * valid_t).sum() / valid_t.sum().clamp_min(1.0)


def enhance_batch(mask: torch.Tensor, mix_refs: torch.Tensor, clean_refs: torch.Tensor, cfg: FeatureConfig):
    enhanced = []
    clean = []
    for i in range(mask.shape[0]):
        y = enhance_with_complex_mask(mix_refs[i], mask[i], cfg)
        n = min(y.numel(), clean_refs[i].numel())
        enhanced.append(y[:n])
        clean.append(clean_refs[i, :n])
    return enhanced, clean


def waveform_l1_loss(mask: torch.Tensor, mix_refs: torch.Tensor, clean_refs: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    enhanced, clean = enhance_batch(mask, mix_refs, clean_refs, cfg)
    return torch.stack([torch.mean(torch.abs(y - c)) for y, c in zip(enhanced, clean)]).mean()


def complex_stft_loss(mask: torch.Tensor, mix_refs: torch.Tensor, clean_refs: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    losses = []
    for i in range(mask.shape[0]):
        noisy_spec = stft(mix_refs[i], cfg)
        clean_spec = stft(clean_refs[i], cfg)
        frames = min(noisy_spec.shape[-1], clean_spec.shape[-1], mask.shape[1])
        bins = min(noisy_spec.shape[0], mask.shape[2])
        m = torch.complex(mask[i, :frames, :bins, 0], mask[i, :frames, :bins, 1]).transpose(0, 1)
        pred_spec = noisy_spec[:bins, :frames] * m
        target_spec = clean_spec[:bins, :frames]
        mag_loss = torch.mean(torch.abs(torch.log1p(pred_spec.abs()) - torch.log1p(target_spec.abs())))
        complex_loss = torch.mean(torch.abs(pred_spec.real - target_spec.real) + torch.abs(pred_spec.imag - target_spec.imag))
        losses.append(mag_loss + 0.25 * complex_loss)
    return torch.stack(losses).mean()


def si_sdr_train_loss(mask: torch.Tensor, mix_refs: torch.Tensor, clean_refs: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    enhanced, clean = enhance_batch(mask, mix_refs, clean_refs, cfg)
    return torch.stack([-si_sdr(y, c) / 20.0 for y, c in zip(enhanced, clean)]).mean()


def high_band_residual_loss(
    mask: torch.Tensor,
    mix_refs: torch.Tensor,
    clean_refs: torch.Tensor,
    cfg: FeatureConfig,
    start_hz: float,
) -> torch.Tensor:
    losses = []
    freqs = torch.linspace(0.0, cfg.sample_rate / 2, cfg.n_fft // 2 + 1, device=mask.device)
    high = freqs >= float(start_hz)
    for i in range(mask.shape[0]):
        y = enhance_with_complex_mask(mix_refs[i], mask[i], cfg)
        n = min(y.numel(), clean_refs[i].numel())
        pred = stft(y[:n], cfg)
        clean = stft(clean_refs[i, :n], cfg)
        frames = min(pred.shape[-1], clean.shape[-1])
        residual = pred[high, :frames] - clean[high, :frames]
        target = clean[high, :frames]
        losses.append(torch.mean(torch.abs(residual)) / torch.mean(torch.abs(target)).clamp_min(1e-4))
    return torch.stack(losses).mean()


def build_complex_targets(mix_refs: torch.Tensor, clean_refs: torch.Tensor, cfg: FeatureConfig, clip: float) -> torch.Tensor:
    targets = []
    max_frames = 0
    for i in range(mix_refs.shape[0]):
        target = target_complex_mask(mix_refs[i], clean_refs[i], cfg, clip)
        targets.append(target)
        max_frames = max(max_frames, target.shape[0])
    out = mix_refs.new_zeros((len(targets), max_frames, cfg.n_fft // 2 + 1, 2))
    for i, target in enumerate(targets):
        out[i, : target.shape[0], : target.shape[1]] = target
    return out


def run_epoch(
    model: TinyComplexMaskTCN,
    loader: DataLoader,
    optimizer,
    device: str,
    cfg: FeatureConfig,
    mask_weight: float,
    waveform_weight: float,
    stft_weight: float,
    sisdr_weight: float,
    high_band_weight: float,
    high_band_start_hz: float,
    mask_clip: float,
) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, leave=False):
            feats, _, valid, mix_refs, clean_refs, _ = batch
            feats = feats.to(device)
            valid = valid.to(device)
            mix_refs = mix_refs.to(device)
            clean_refs = clean_refs.to(device)
            pred = model(feats)
            target = build_complex_targets(mix_refs, clean_refs, cfg, mask_clip)
            loss = mask_weight * complex_mask_mse(pred, target, valid)
            if waveform_weight > 0.0:
                loss = loss + waveform_weight * waveform_l1_loss(pred, mix_refs, clean_refs, cfg)
            if stft_weight > 0.0:
                loss = loss + stft_weight * complex_stft_loss(pred, mix_refs, clean_refs, cfg)
            if sisdr_weight > 0.0:
                loss = loss + sisdr_weight * si_sdr_train_loss(pred, mix_refs, clean_refs, cfg)
            if high_band_weight > 0.0:
                loss = loss + high_band_weight * high_band_residual_loss(
                    pred,
                    mix_refs,
                    clean_refs,
                    cfg,
                    high_band_start_hz,
                )
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--on-the-fly", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--channels", type=int, default=80)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--mask-scale", type=float, default=2.0)
    parser.add_argument("--mask-clip", type=float, default=2.0)
    parser.add_argument("--mask-loss-weight", type=float, default=0.25)
    parser.add_argument("--waveform-loss-weight", type=float, default=0.5)
    parser.add_argument("--stft-loss-weight", type=float, default=0.35)
    parser.add_argument("--si-sdr-loss-weight", type=float, default=0.04)
    parser.add_argument("--high-band-loss-weight", type=float, default=0.08)
    parser.add_argument("--high-band-start-hz", type=float, default=2500.0)
    parser.add_argument("--spatial-frontend", choices=["delay_sum", "coherence_mwf"], default="coherence_mwf")
    parser.add_argument("--resume")
    parser.add_argument("--reset-best-on-resume", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = FeatureConfig(spatial_features=True, spatial_frontend=args.spatial_frontend)
    train_ds = WavPairDataset(args.data, "train", cfg, args.seconds, args.on_the_fly, return_audio=True)
    val_ds = WavPairDataset(args.data, "val", cfg, args.seconds, args.on_the_fly, return_audio=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_sequence_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_sequence_batch)

    model = TinyComplexMaskTCN(cfg.feature_dim, cfg.n_fft // 2 + 1, args.channels, args.blocks, args.kernel_size, args.mask_scale)
    best = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if ckpt.get("val_loss") is not None and not args.reset_best_on_resume:
            best = float(ckpt["val_loss"])
        print(f"resumed_from={args.resume}")
    params = count_parameters(model)
    print(f"parameters={params} int8_weight_bytes~={params}")
    print(f"spatial_frontend={cfg.spatial_frontend}")
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            cfg,
            args.mask_loss_weight,
            args.waveform_loss_weight,
            args.stft_loss_weight,
            args.si_sdr_loss_weight,
            args.high_band_loss_weight,
            args.high_band_start_hz,
            args.mask_clip,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            None,
            args.device,
            cfg,
            args.mask_loss_weight,
            args.waveform_loss_weight,
            args.stft_loss_weight,
            args.si_sdr_loss_weight,
            args.high_band_loss_weight,
            args.high_band_start_hz,
            args.mask_clip,
        )
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        state = {
            "model": model.state_dict(),
            "config": {
                "sample_rate": cfg.sample_rate,
                "n_fft": cfg.n_fft,
                "hop_length": cfg.hop_length,
                "bands": cfg.bands,
                "feature_dim": cfg.feature_dim,
                "spatial_features": cfg.spatial_features,
                "spatial_frontend": cfg.spatial_frontend,
                "freq_bins": cfg.n_fft // 2 + 1,
                "channels": args.channels,
                "blocks": args.blocks,
                "kernel_size": args.kernel_size,
                "mask_scale": args.mask_scale,
                "mask_clip": args.mask_clip,
                "model_type": "tiny_complex_mask_tcn",
                "mask_loss_weight": args.mask_loss_weight,
                "waveform_loss_weight": args.waveform_loss_weight,
                "stft_loss_weight": args.stft_loss_weight,
                "si_sdr_loss_weight": args.si_sdr_loss_weight,
                "high_band_loss_weight": args.high_band_loss_weight,
                "high_band_start_hz": args.high_band_start_hz,
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
