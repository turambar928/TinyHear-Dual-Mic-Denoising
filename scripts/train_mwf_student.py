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


def moving_average(x: torch.Tensor, window: int, causal: bool) -> torch.Tensor:
    if window <= 1:
        return x
    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype) / float(window)
    flat = x.reshape(-1, 1, x.shape[-1])
    if causal:
        padded = F.pad(flat, (window - 1, 0), mode="replicate")
    else:
        left = window // 2
        right = window - 1 - left
        padded = F.pad(flat, (left, right), mode="replicate")
    out = F.conv1d(padded, kernel)
    return out.reshape_as(x)


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
    return torch.conj(w0) * x0 + torch.conj(w1) * x1


def complex_mask_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    frames = min(pred.shape[1], target.shape[1], valid.shape[-1])
    pred = pred[:, :frames]
    target = target[:, :frames].to(pred.device)
    valid_t = valid[:, :, :frames].transpose(1, 2).unsqueeze(-1).to(pred.device)
    return ((pred - target) ** 2 * valid_t).sum() / valid_t.sum().clamp_min(1.0)


def build_teacher_targets(
    mix_pairs: torch.Tensor,
    reference: torch.Tensor,
    clean_refs: torch.Tensor,
    cfg: FeatureConfig,
    oracle_window: int,
    oracle_causal: bool,
    diagonal_loading: float,
    mask_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    masks = []
    teachers = []
    max_frames = 0
    for i in range(mix_pairs.shape[0]):
        teacher_spec = oracle_local_mwf(
            mix_pairs[i],
            clean_refs[i],
            cfg,
            oracle_window,
            oracle_causal,
            diagonal_loading,
        )
        teacher = torch.istft(
            teacher_spec,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.n_fft,
            window=torch.hann_window(cfg.n_fft, device=teacher_spec.device, dtype=teacher_spec.real.dtype),
            center=True,
            length=reference[i].numel(),
        )
        target = target_complex_mask(reference[i], teacher, cfg, mask_clip)
        masks.append(target)
        teachers.append(teacher)
        max_frames = max(max_frames, target.shape[0])
    target_out = reference.new_zeros((len(masks), max_frames, cfg.n_fft // 2 + 1, 2))
    teacher_out = reference.new_zeros((len(teachers), reference.shape[1]))
    for i, target in enumerate(masks):
        target_out[i, : target.shape[0], : target.shape[1]] = target
        teacher_out[i, : teachers[i].numel()] = teachers[i]
    return target_out, teacher_out


def enhance_batch(mask: torch.Tensor, refs: torch.Tensor, targets: torch.Tensor, cfg: FeatureConfig):
    enhanced = []
    target = []
    for i in range(mask.shape[0]):
        y = enhance_with_complex_mask(refs[i], mask[i], cfg)
        n = min(y.numel(), targets[i].numel())
        enhanced.append(y[:n])
        target.append(targets[i, :n])
    return enhanced, target


def waveform_l1_loss(mask: torch.Tensor, refs: torch.Tensor, targets: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    enhanced, target = enhance_batch(mask, refs, targets, cfg)
    return torch.stack([torch.mean(torch.abs(y - t)) for y, t in zip(enhanced, target)]).mean()


def complex_stft_loss(mask: torch.Tensor, refs: torch.Tensor, targets: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    losses = []
    for i in range(mask.shape[0]):
        noisy_spec = stft(refs[i], cfg)
        target_spec = stft(targets[i], cfg)
        frames = min(noisy_spec.shape[-1], target_spec.shape[-1], mask.shape[1])
        bins = min(noisy_spec.shape[0], mask.shape[2])
        m = torch.complex(mask[i, :frames, :bins, 0], mask[i, :frames, :bins, 1]).transpose(0, 1)
        pred_spec = noisy_spec[:bins, :frames] * m
        target_spec = target_spec[:bins, :frames]
        mag = torch.mean(torch.abs(torch.log1p(pred_spec.abs()) - torch.log1p(target_spec.abs())))
        complex_l1 = torch.mean(torch.abs(pred_spec.real - target_spec.real) + torch.abs(pred_spec.imag - target_spec.imag))
        losses.append(mag + 0.25 * complex_l1)
    return torch.stack(losses).mean()


def si_sdr_train_loss(mask: torch.Tensor, refs: torch.Tensor, targets: torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    enhanced, target = enhance_batch(mask, refs, targets, cfg)
    return torch.stack([-si_sdr(y, t) / 20.0 for y, t in zip(enhanced, target)]).mean()


def run_epoch(
    model: TinyComplexMaskTCN,
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
            valid = valid.to(device)
            reference = reference.to(device)
            clean_refs = clean_refs.to(device)
            mix_pairs = mix_pairs.to(device)
            clean_pairs = clean_pairs.to(device)

            pred = model(feats)
            teacher_mask, teacher_wav = build_teacher_targets(
                mix_pairs,
                reference,
                clean_pairs,
                cfg,
                args.oracle_window,
                args.oracle_causal,
                args.diagonal_loading,
                args.mask_clip,
            )
            loss = args.teacher_mask_weight * complex_mask_mse(pred, teacher_mask, valid)
            if args.teacher_waveform_weight > 0.0:
                loss = loss + args.teacher_waveform_weight * waveform_l1_loss(pred, reference, teacher_wav, cfg)
            if args.teacher_stft_weight > 0.0:
                loss = loss + args.teacher_stft_weight * complex_stft_loss(pred, reference, teacher_wav, cfg)
            if args.clean_stft_weight > 0.0:
                loss = loss + args.clean_stft_weight * complex_stft_loss(pred, reference, clean_refs, cfg)
            if args.clean_sisdr_weight > 0.0:
                loss = loss + args.clean_sisdr_weight * si_sdr_train_loss(pred, reference, clean_refs, cfg)
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
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--on-the-fly", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--channels", type=int, default=80)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--mask-scale", type=float, default=2.0)
    parser.add_argument("--mask-clip", type=float, default=2.0)
    parser.add_argument("--spatial-frontend", choices=["delay_sum", "coherence_mwf"], default="coherence_mwf")
    parser.add_argument("--oracle-window", type=int, default=9)
    parser.add_argument("--oracle-causal", action="store_true")
    parser.add_argument("--diagonal-loading", type=float, default=0.01)
    parser.add_argument("--teacher-mask-weight", type=float, default=0.35)
    parser.add_argument("--teacher-waveform-weight", type=float, default=0.45)
    parser.add_argument("--teacher-stft-weight", type=float, default=0.45)
    parser.add_argument("--clean-stft-weight", type=float, default=0.20)
    parser.add_argument("--clean-sisdr-weight", type=float, default=0.04)
    parser.add_argument("--resume")
    parser.add_argument("--reset-best-on-resume", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = FeatureConfig(spatial_features=True, spatial_frontend=args.spatial_frontend)
    train_ds = WavPairDataset(args.data, "train", cfg, args.seconds, args.on_the_fly, return_audio=True, return_mix_pair=True)
    val_ds = WavPairDataset(args.data, "val", cfg, args.seconds, args.on_the_fly, return_audio=True, return_mix_pair=True)
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
                "mask_scale": args.mask_scale,
                "mask_clip": args.mask_clip,
                "model_type": "tiny_mwf_student_complex_mask_tcn",
                "teacher": "oracle_local_mwf",
                "oracle_window": args.oracle_window,
                "oracle_causal": args.oracle_causal,
                "diagonal_loading": args.diagonal_loading,
                "teacher_mask_weight": args.teacher_mask_weight,
                "teacher_waveform_weight": args.teacher_waveform_weight,
                "teacher_stft_weight": args.teacher_stft_weight,
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
