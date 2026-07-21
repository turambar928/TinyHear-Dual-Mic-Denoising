#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ha_denoise.dataset import WavPairDataset
from ha_denoise.features import FeatureConfig, enhance_with_deep_filter, pad_sequence_batch
from ha_denoise.metrics import si_sdr
from ha_denoise.model import TinyCausalTCN, TinyDeepFilterTCN, count_parameters


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * valid).sum() / valid.sum().clamp_min(1.0)


def waveform_l1_loss(gain: torch.Tensor, coef: torch.Tensor, mix_refs: torch.Tensor, clean_refs: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    losses = []
    for i in range(gain.shape[0]):
        enhanced = enhance_with_deep_filter(mix_refs[i], gain[i].transpose(0, 1), coef[i], cfg)
        n = min(enhanced.numel(), clean_refs[i].numel())
        losses.append(torch.mean(torch.abs(enhanced[:n] - clean_refs[i, :n])))
    return torch.stack(losses).mean()


def stft_logmag_loss(gain: torch.Tensor, coef: torch.Tensor, mix_refs: torch.Tensor, clean_refs: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    window = torch.hann_window(cfg.n_fft, device=gain.device, dtype=mix_refs.dtype)
    losses = []
    for i in range(gain.shape[0]):
        enhanced = enhance_with_deep_filter(mix_refs[i], gain[i].transpose(0, 1), coef[i], cfg)
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


def si_sdr_loss(gain: torch.Tensor, coef: torch.Tensor, mix_refs: torch.Tensor, clean_refs: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    losses = []
    for i in range(gain.shape[0]):
        enhanced = enhance_with_deep_filter(mix_refs[i], gain[i].transpose(0, 1), coef[i], cfg)
        n = min(enhanced.numel(), clean_refs[i].numel())
        losses.append(-si_sdr(enhanced[:n], clean_refs[i, :n]) / 20.0)
    return torch.stack(losses).mean()


def coef_identity_loss(coef: torch.Tensor) -> torch.Tensor:
    target = torch.zeros_like(coef)
    target[:, :, :, 0, 0] = 1.0
    return torch.mean((coef - target) ** 2)


def run_epoch(
    model: TinyDeepFilterTCN,
    loader: DataLoader,
    optimizer,
    device: str,
    cfg: FeatureConfig,
    waveform_weight: float,
    stft_weight: float,
    sisdr_weight: float,
    coef_reg_weight: float,
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
            gain, coef = model(feats)
            loss = masked_mse(gain, masks, valid)
            if waveform_weight > 0.0:
                loss = loss + waveform_weight * waveform_l1_loss(gain, coef, mix_refs, clean_refs, cfg)
            if stft_weight > 0.0:
                loss = loss + stft_weight * stft_logmag_loss(gain, coef, mix_refs, clean_refs, cfg)
            if sisdr_weight > 0.0:
                loss = loss + sisdr_weight * si_sdr_loss(gain, coef, mix_refs, clean_refs, cfg)
            if coef_reg_weight > 0.0:
                loss = loss + coef_reg_weight * coef_identity_loss(coef)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total += float(loss.item())
    return total / max(1, len(loader))


def load_denoiser_backbone(path: str, device: str) -> tuple[dict[str, torch.Tensor], dict]:
    ckpt = torch.load(path, map_location=device)
    return ckpt["model"], ckpt["config"]


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
    parser.add_argument("--resume-denoiser", help="Initialize stem/tcn/gain head from a TinyCausalTCN checkpoint.")
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--df-bins", type=int, default=64)
    parser.add_argument("--df-order", type=int, default=3)
    parser.add_argument("--coef-scale", type=float, default=1.5)
    parser.add_argument("--waveform-loss-weight", type=float, default=0.5)
    parser.add_argument("--stft-mag-loss-weight", type=float, default=0.1)
    parser.add_argument("--si-sdr-loss-weight", type=float, default=0.02)
    parser.add_argument("--coef-reg-weight", type=float, default=0.01)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = FeatureConfig(spatial_features=True)
    train_ds = WavPairDataset(args.data, "train", cfg, args.seconds, args.on_the_fly, return_audio=True)
    val_ds = WavPairDataset(args.data, "val", cfg, args.seconds, args.on_the_fly, return_audio=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_sequence_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_sequence_batch)

    model = TinyDeepFilterTCN(
        cfg.feature_dim,
        cfg.bands,
        args.channels,
        args.blocks,
        args.kernel_size,
        args.df_bins,
        args.df_order,
        args.coef_scale,
    )
    if args.resume_denoiser:
        state, denoiser_cfg = load_denoiser_backbone(args.resume_denoiser, "cpu")
        if int(denoiser_cfg["feature_dim"]) != cfg.feature_dim:
            raise ValueError("resume denoiser must use spatial feature config")
        if int(denoiser_cfg["channels"]) == args.channels:
            model.load_denoiser_backbone(state)
        else:
            print("skip_backbone_init=channel_mismatch")
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
            args.waveform_loss_weight,
            args.stft_mag_loss_weight,
            args.si_sdr_loss_weight,
            args.coef_reg_weight,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            None,
            args.device,
            cfg,
            args.waveform_loss_weight,
            args.stft_mag_loss_weight,
            args.si_sdr_loss_weight,
            args.coef_reg_weight,
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
                "channels": args.channels,
                "blocks": args.blocks,
                "kernel_size": args.kernel_size,
                "df_bins": args.df_bins,
                "df_order": args.df_order,
                "coef_scale": args.coef_scale,
                "model_type": "tiny_deepfilter_tcn",
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
