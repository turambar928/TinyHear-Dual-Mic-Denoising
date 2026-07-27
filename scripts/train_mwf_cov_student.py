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
    enhance_with_mwf_masks,
    extract_features,
    pad_sequence_batch,
    stft,
)
from ha_denoise.metrics import si_sdr
from ha_denoise.model import TinyMwfMaskTCN, count_parameters


def moving_average(x: torch.Tensor, window: int, causal: bool) -> torch.Tensor:
    if window <= 1:
        return x
    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype) / float(window)
    flat = x.reshape(-1, 1, x.shape[-1])
    if causal:
        padded = torch.nn.functional.pad(flat, (window - 1, 0), mode="replicate")
    else:
        left = window // 2
        right = window - 1 - left
        padded = torch.nn.functional.pad(flat, (left, right), mode="replicate")
    return torch.nn.functional.conv1d(padded, kernel).reshape_as(x)


def moving_average_complex(x: torch.Tensor, window: int, causal: bool) -> torch.Tensor:
    return torch.complex(moving_average(x.real, window, causal), moving_average(x.imag, window, causal))


def oracle_local_mwf(
    mix: torch.Tensor,
    clean: torch.Tensor,
    cfg: FeatureConfig,
    window: int,
    causal: bool,
    diagonal_loading: float,
) -> torch.Tensor:
    mix_spec = torch.stack([stft(mix[0], cfg), stft(mix[1], cfg)], dim=0)
    clean_spec = torch.stack([stft(clean[0], cfg), stft(clean[1], cfg)], dim=0)
    frames = min(mix_spec.shape[-1], clean_spec.shape[-1])
    mix_spec = mix_spec[:, :, :frames]
    clean_spec = clean_spec[:, :, :frames]
    x0 = mix_spec[0]
    x1 = mix_spec[1]
    s0 = clean_spec[0]

    r00 = moving_average_complex(x0 * torch.conj(x0), window, causal).real
    r01 = moving_average_complex(x0 * torch.conj(x1), window, causal)
    r10 = torch.conj(r01)
    r11 = moving_average_complex(x1 * torch.conj(x1), window, causal).real
    p0 = moving_average_complex(x0 * torch.conj(s0), window, causal)
    p1 = moving_average_complex(x1 * torch.conj(s0), window, causal)

    trace = (r00 + r11).clamp_min(1e-8)
    load = float(diagonal_loading) * trace + 1e-8
    a = r00 + load
    d = r11 + load
    b = r01
    c = r10
    det = a * d - b * c + 1e-8
    w0 = (d * p0 - b * p1) / det
    w1 = (-c * p0 + a * p1) / det
    enhanced_spec = torch.conj(w0) * x0 + torch.conj(w1) * x1
    return torch.istft(
        enhanced_spec,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.n_fft,
        window=torch.hann_window(cfg.n_fft, device=enhanced_spec.device, dtype=enhanced_spec.real.dtype),
        center=True,
        length=mix.shape[-1],
    )


def waveform_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.mean(torch.abs(p[: min(p.numel(), t.numel())] - t[: min(p.numel(), t.numel())])) for p, t in zip(pred, target)]).mean()


def stft_loss(pred: torch.Tensor, target: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    losses = []
    for p, t in zip(pred, target):
        p_spec = stft(p, cfg)
        t_spec = stft(t, cfg)
        frames = min(p_spec.shape[-1], t_spec.shape[-1])
        p_spec = p_spec[:, :frames]
        t_spec = t_spec[:, :frames]
        losses.append(torch.mean(torch.abs(torch.log1p(p_spec.abs()) - torch.log1p(t_spec.abs()))))
    return torch.stack(losses).mean()


def run_epoch(
    model: TinyMwfMaskTCN,
    loader: DataLoader,
    optimizer,
    device: str,
    cfg: FeatureConfig,
    args: argparse.Namespace,
) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, leave=False):
            feats, _, valid, reference, clean_refs, _, mix_pairs, clean_pairs = batch
            feats = feats.to(device)
            mix_pairs = mix_pairs.to(device)
            clean_pairs = clean_pairs.to(device)

            pred_masks = model(feats)
            enhanced = []
            teacher = []
            clean = []
            for i in range(pred_masks.shape[0]):
                y = enhance_with_mwf_masks(
                    mix_pairs[i],
                    pred_masks[i],
                    cfg,
                    covariance_alpha=args.covariance_alpha,
                    diagonal_loading=args.diagonal_loading,
                    min_mask=args.min_mask,
                )
                t = oracle_local_mwf(
                    mix_pairs[i],
                    clean_pairs[i],
                    cfg,
                    args.oracle_window,
                    args.oracle_causal,
                    args.diagonal_loading,
                )
                n = min(y.numel(), t.numel(), clean_pairs[i][0].numel())
                enhanced.append(y[:n])
                teacher.append(t[:n])
                clean.append(clean_pairs[i][0, :n])

            enhanced = torch.stack(enhanced)
            teacher = torch.stack(teacher)
            clean = torch.stack(clean)

            loss = args.teacher_waveform_weight * waveform_l1_loss(enhanced, teacher)
            if args.teacher_stft_weight > 0.0:
                loss = loss + args.teacher_stft_weight * stft_loss(enhanced, teacher, cfg)
            if args.clean_waveform_weight > 0.0:
                loss = loss + args.clean_waveform_weight * waveform_l1_loss(enhanced, clean)
            if args.clean_stft_weight > 0.0:
                loss = loss + args.clean_stft_weight * stft_loss(enhanced, clean, cfg)
            if args.clean_sisdr_weight > 0.0:
                loss = loss + args.clean_sisdr_weight * torch.stack([-si_sdr(y, c) / 20.0 for y, c in zip(enhanced, clean)]).mean()
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--max-train-items", type=int)
    parser.add_argument("--max-val-items", type=int)
    parser.add_argument("--on-the-fly", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--channels", type=int, default=80)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--spatial-frontend", choices=["delay_sum", "coherence_mwf"], default="coherence_mwf")
    parser.add_argument("--oracle-window", type=int, default=9)
    parser.add_argument("--oracle-causal", action="store_true")
    parser.add_argument("--covariance-alpha", type=float, default=0.96)
    parser.add_argument("--diagonal-loading", type=float, default=0.01)
    parser.add_argument("--min-mask", type=float, default=0.02)
    parser.add_argument("--teacher-waveform-weight", type=float, default=0.50)
    parser.add_argument("--teacher-stft-weight", type=float, default=0.35)
    parser.add_argument("--clean-waveform-weight", type=float, default=0.20)
    parser.add_argument("--clean-stft-weight", type=float, default=0.25)
    parser.add_argument("--clean-sisdr-weight", type=float, default=0.04)
    parser.add_argument("--resume")
    parser.add_argument("--reset-best-on-resume", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = FeatureConfig(spatial_features=True, spatial_frontend=args.spatial_frontend)
    train_ds = WavPairDataset(args.data, "train", cfg, args.seconds, args.on_the_fly, return_audio=True, return_mix_pair=True)
    val_ds = WavPairDataset(args.data, "val", cfg, args.seconds, args.on_the_fly, return_audio=True, return_mix_pair=True)
    if args.max_train_items is not None:
        train_ds.items = train_ds.items[: args.max_train_items]
    if args.max_val_items is not None:
        val_ds.items = val_ds.items[: args.max_val_items]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_sequence_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_sequence_batch)

    model = TinyMwfMaskTCN(cfg.feature_dim, cfg.n_fft // 2 + 1, args.channels, args.blocks, args.kernel_size)
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
    print(f"teacher=oracle_local_mwf window={args.oracle_window} causal={args.oracle_causal}")
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, args.device, cfg, args)
        val_loss = run_epoch(model, val_loader, None, args.device, cfg, args)
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
                "model_type": "tiny_mwf_mask_tcn",
                "teacher": "oracle_local_mwf",
                "oracle_window": args.oracle_window,
                "oracle_causal": args.oracle_causal,
                "covariance_alpha": args.covariance_alpha,
                "diagonal_loading": args.diagonal_loading,
                "min_mask": args.min_mask,
                "teacher_waveform_weight": args.teacher_waveform_weight,
                "teacher_stft_weight": args.teacher_stft_weight,
                "clean_waveform_weight": args.clean_waveform_weight,
                "clean_stft_weight": args.clean_stft_weight,
                "clean_sisdr_weight": args.clean_sisdr_weight,
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
