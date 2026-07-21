#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ha_denoise.dataset import WavPairDataset
from ha_denoise.features import FeatureConfig, enhance_with_mask, pad_sequence_batch, feature_config_from_dict
from ha_denoise.metrics import si_sdr
from ha_denoise.model import TinyCausalTCN


class TinyGate(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pooled_features(feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # feats: [B, F, T], valid: [B, 1, T]
    denom = valid.sum(dim=-1).clamp_min(1.0)
    mean = (feats * valid).sum(dim=-1) / denom
    centered = (feats - mean.unsqueeze(-1)) * valid
    var = centered.square().sum(dim=-1) / denom
    return torch.cat([mean, torch.sqrt(var.clamp_min(1e-8))], dim=-1)


def gate_targets(mix_refs: torch.Tensor, clean_refs: torch.Tensor, threshold: float) -> torch.Tensor:
    labels = []
    for i in range(mix_refs.shape[0]):
        score = si_sdr(mix_refs[i], clean_refs[i])
        labels.append((score < threshold).to(mix_refs.dtype))
    return torch.stack(labels).to(mix_refs.device)


def load_denoiser(checkpoint: str, device: str) -> tuple[TinyCausalTCN, FeatureConfig]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg_d = ckpt["config"]
    cfg = feature_config_from_dict(cfg_d)
    model = TinyCausalTCN(cfg_d["feature_dim"], cfg_d["bands"], cfg_d["channels"], cfg_d["blocks"], cfg_d["kernel_size"], float(cfg_d.get("output_min_gain", 0.0)), float(cfg_d.get("output_max_gain", 1.0)))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def oracle_blend_targets(
    denoiser: TinyCausalTCN,
    cfg: FeatureConfig,
    feats: torch.Tensor,
    mix_refs: torch.Tensor,
    clean_refs: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    alphas = torch.linspace(0.0, 1.0, steps, device=feats.device, dtype=feats.dtype)
    labels = []
    with torch.no_grad():
        pred = denoiser(feats)
        for i in range(pred.shape[0]):
            mask = pred[i].transpose(0, 1)
            enhanced = enhance_with_mask(mix_refs[i], mask, cfg)
            n = min(mix_refs[i].numel(), clean_refs[i].numel(), enhanced.numel())
            noisy = mix_refs[i, :n]
            clean = clean_refs[i, :n]
            enhanced = enhanced[:n]
            scores = []
            for alpha in alphas:
                blended = alpha * enhanced + (1.0 - alpha) * noisy
                scores.append(si_sdr(blended, clean))
            best_idx = int(torch.stack(scores).argmax().item())
            labels.append(alphas[best_idx])
    return torch.stack(labels).to(feats.device)


def run_epoch(model, loader, optimizer, device, threshold, target_mode, denoiser, denoiser_cfg, oracle_steps):
    train = optimizer is not None
    model.train(train)
    total = 0.0
    metric_total = 0.0
    count = 0
    loss_fn = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, leave=False):
            feats, _, valid, mix_refs, clean_refs, _ = batch
            feats = feats.to(device)
            valid = valid.to(device)
            mix_refs = mix_refs.to(device)
            clean_refs = clean_refs.to(device)
            x = pooled_features(feats, valid)
            logits = model(x)
            if target_mode == "oracle-blend":
                if denoiser is None or denoiser_cfg is None:
                    raise ValueError("--denoiser is required for oracle-blend gate training")
                target = oracle_blend_targets(denoiser, denoiser_cfg, feats, mix_refs, clean_refs, oracle_steps)
                pred = torch.sigmoid(logits)
                loss = mse_loss(pred, target)
                metric_total += float(torch.mean(torch.abs(pred.detach() - target)).item()) * int(target.numel())
            else:
                target = gate_targets(mix_refs, clean_refs, threshold)
                loss = loss_fn(logits, target)
                pred = (torch.sigmoid(logits) >= 0.5).to(target.dtype)
                metric_total += int((pred == target).sum().item())
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            count += int(target.numel())
            total += float(loss.item())
    return total / max(1, len(loader)), metric_total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--on-the-fly", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--spatial-features", action="store_true")
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--target-mode", choices=("threshold", "oracle-blend"), default="threshold")
    parser.add_argument("--denoiser", help="Denoiser checkpoint used to build oracle-blend soft targets.")
    parser.add_argument("--oracle-steps", type=int, default=21)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = FeatureConfig(spatial_features=args.spatial_features)
    train_ds = WavPairDataset(args.data, "train", cfg, args.seconds, args.on_the_fly, return_audio=True)
    val_ds = WavPairDataset(args.data, "val", cfg, args.seconds, args.on_the_fly, return_audio=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_sequence_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_sequence_batch)

    model = TinyGate(cfg.feature_dim * 2, args.hidden).to(args.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"gate_parameters={params}")
    denoiser = None
    denoiser_cfg = None
    if args.target_mode == "oracle-blend":
        denoiser, denoiser_cfg = load_denoiser(args.denoiser, args.device)
        if denoiser_cfg.feature_dim != cfg.feature_dim:
            raise ValueError("gate feature config must match denoiser feature config")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metric = run_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            args.threshold,
            args.target_mode,
            denoiser,
            denoiser_cfg,
            args.oracle_steps,
        )
        val_loss, val_metric = run_epoch(
            model,
            val_loader,
            None,
            args.device,
            args.threshold,
            args.target_mode,
            denoiser,
            denoiser_cfg,
            args.oracle_steps,
        )
        metric_name = "mae" if args.target_mode == "oracle-blend" else "acc"
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} train_{metric_name}={train_metric:.4f} "
            f"val_loss={val_loss:.6f} val_{metric_name}={val_metric:.4f}"
        )
        state = {
            "model": model.state_dict(),
            "config": {
                "sample_rate": cfg.sample_rate,
                "n_fft": cfg.n_fft,
                "hop_length": cfg.hop_length,
                "bands": cfg.bands,
                "feature_dim": cfg.feature_dim,
                "spatial_features": cfg.spatial_features,
                "threshold": args.threshold,
                "hidden": args.hidden,
                "input_dim": cfg.feature_dim * 2,
                "target_mode": args.target_mode,
                "oracle_steps": args.oracle_steps,
            },
            "epoch": epoch,
            "val_loss": val_loss,
            "val_metric": val_metric,
        }
        torch.save(state, out / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(state, out / "best.pt")


if __name__ == "__main__":
    main()
