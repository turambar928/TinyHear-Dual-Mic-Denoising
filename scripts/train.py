#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ha_denoise.dataset import WavPairDataset
from ha_denoise.features import FeatureConfig, enhance_with_mask, pad_sequence_batch
from ha_denoise.metrics import si_sdr
from ha_denoise.model import TinyCausalTCN, count_parameters


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * valid).sum() / valid.sum().clamp_min(1.0)


def masked_band_mag_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    feats: torch.Tensor,
    valid: torch.Tensor,
    bands: int,
) -> torch.Tensor:
    noisy_mag = torch.exp(0.5 * torch.clamp(feats[:, :bands], -20.0, 20.0))
    pred_mag = pred * noisy_mag
    target_mag = target * noisy_mag
    return (torch.abs(pred_mag - target_mag) * valid).sum() / valid.sum().clamp_min(1.0)


def si_sdr_mask_loss(pred: torch.Tensor, mix_refs: torch.Tensor, clean_refs: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    losses = []
    for i in range(pred.shape[0]):
        mask = pred[i].transpose(0, 1)
        enhanced = enhance_with_mask(mix_refs[i], mask, cfg)
        n = min(enhanced.numel(), clean_refs[i].numel())
        losses.append(-si_sdr(enhanced[:n], clean_refs[i, :n]) / 20.0)
    return torch.stack(losses).mean()


def run_epoch(model, loader, optimizer, device, cfg, band_mag_loss_weight, si_sdr_loss_weight):
    train = optimizer is not None
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, leave=False):
            feats, masks, valid = batch[:3]
            feats = feats.to(device)
            masks = masks.to(device)
            valid = valid.to(device)
            pred = model(feats)
            loss = masked_mse(pred, masks, valid)
            if band_mag_loss_weight > 0.0:
                loss = loss + band_mag_loss_weight * masked_band_mag_l1(pred, masks, feats, valid, cfg.bands)
            if si_sdr_loss_weight > 0.0:
                mix_refs = batch[3].to(device)
                clean_refs = batch[4].to(device)
                loss = loss + si_sdr_loss_weight * si_sdr_mask_loss(pred, mix_refs, clean_refs, cfg)
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--on-the-fly", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", help="Optional checkpoint to initialize model weights from.")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--channels", type=int, default=112)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--spatial-features", action="store_true", help="Add IPD sin/cos and coherence band features.")
    parser.add_argument("--band-mag-loss-weight", type=float, default=0.0)
    parser.add_argument("--si-sdr-loss-weight", type=float, default=0.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = FeatureConfig(spatial_features=args.spatial_features)
    return_audio = args.si_sdr_loss_weight > 0.0
    train_ds = WavPairDataset(args.data, "train", cfg, args.seconds, args.on_the_fly, return_audio=return_audio)
    val_ds = WavPairDataset(args.data, "val", cfg, args.seconds, args.on_the_fly, return_audio=return_audio)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_sequence_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_sequence_batch)

    model = TinyCausalTCN(
        feature_dim=cfg.feature_dim,
        bands=cfg.bands,
        channels=args.channels,
        blocks=args.blocks,
        kernel_size=args.kernel_size,
    )
    best = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if ckpt.get("val_mse") is not None:
            best = float(ckpt["val_mse"])
        print(f"resumed_from={args.resume}")
    params = count_parameters(model)
    print(f"parameters={params} int8_weight_bytes~={params}")
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.start_epoch + 1, args.start_epoch + args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            cfg,
            args.band_mag_loss_weight,
            args.si_sdr_loss_weight,
        )
        val_loss = run_epoch(model, val_loader, None, args.device, cfg, args.band_mag_loss_weight, args.si_sdr_loss_weight)
        print(f"epoch={epoch} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")
        state = {
            "model": model.state_dict(),
            "config": {
                "sample_rate": cfg.sample_rate,
                "n_fft": cfg.n_fft,
                "hop_length": cfg.hop_length,
                "bands": cfg.bands,
                "feature_dim": cfg.feature_dim,
                "spatial_features": cfg.spatial_features,
                "channels": model.channels,
                "blocks": model.blocks,
                "kernel_size": model.kernel_size,
                "spatial_features": cfg.spatial_features,
                "band_mag_loss_weight": args.band_mag_loss_weight,
                "si_sdr_loss_weight": args.si_sdr_loss_weight,
            },
            "epoch": epoch,
            "val_mse": val_loss,
        }
        torch.save(state, out / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(state, out / "best.pt")


if __name__ == "__main__":
    main()
