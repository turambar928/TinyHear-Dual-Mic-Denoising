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
from ha_denoise.features import FeatureConfig, enhance_with_deep_filter, istft, pad_sequence_batch, stft, target_band_mask
from ha_denoise.metrics import si_sdr
from ha_denoise.model import TinyCausalTCN, TinyDeepFilterTCN, count_parameters


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * valid).sum() / valid.sum().clamp_min(1.0)


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


def oracle_local_mwf_waveform(
    mix: torch.Tensor,
    clean: torch.Tensor,
    cfg: FeatureConfig,
    window: int,
    causal: bool,
    diagonal_loading: float,
) -> torch.Tensor:
    """Oracle two-mic local MWF teacher estimating clean mic0."""
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
    return istft(enhanced_spec, length=mix.shape[-1], cfg=cfg)


def build_mwf_teacher_targets(
    mix_pairs: torch.Tensor,
    clean_pairs: torch.Tensor,
    clean_refs: torch.Tensor,
    cfg: FeatureConfig,
    oracle_window: int,
    oracle_causal: bool,
    diagonal_loading: float,
    teacher_blend: float,
) -> torch.Tensor:
    targets = torch.zeros_like(clean_refs)
    blend = float(min(max(teacher_blend, 0.0), 1.0))
    for i in range(mix_pairs.shape[0]):
        teacher = oracle_local_mwf_waveform(
            mix_pairs[i],
            clean_pairs[i],
            cfg,
            oracle_window,
            oracle_causal,
            diagonal_loading,
        )
        n = min(teacher.numel(), clean_refs.shape[1])
        targets[i, :n] = blend * teacher[:n] + (1.0 - blend) * clean_refs[i, :n]
    return targets


def build_target_masks(mix_refs: torch.Tensor, target_refs: torch.Tensor, cfg: FeatureConfig, bands: int, frames: int) -> torch.Tensor:
    masks = mix_refs.new_zeros((mix_refs.shape[0], bands, frames))
    for i in range(mix_refs.shape[0]):
        n = min(mix_refs[i].numel(), target_refs[i].numel())
        mask = target_band_mask(mix_refs[i, :n], target_refs[i, :n], cfg).transpose(0, 1)
        t = min(mask.shape[-1], frames)
        b = min(mask.shape[0], bands)
        masks[i, :b, :t] = mask[:b, :t]
    return masks


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


def coef_energy_loss(coef: torch.Tensor) -> torch.Tensor:
    """Keep the residual deep-filter branch small unless it improves audio."""
    return torch.mean(coef.square())


def residual_noise_loss(
    gain: torch.Tensor,
    coef: torch.Tensor,
    mix_refs: torch.Tensor,
    clean_refs: torch.Tensor,
    cfg: FeatureConfig,
    speech_threshold: float,
) -> torch.Tensor:
    window = torch.hann_window(cfg.n_fft, device=gain.device, dtype=mix_refs.dtype)
    losses = []
    threshold = float(max(speech_threshold, 1e-4))
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
        clean_mag = clean_spec.abs()
        frame_ref = clean_mag.amax(dim=0, keepdim=True).clamp_min(1e-5)
        quiet_weight = torch.clamp((threshold * frame_ref - clean_mag) / (threshold * frame_ref), 0.0, 1.0)
        losses.append((torch.log1p(enh_spec.abs()) * quiet_weight).sum() / quiet_weight.sum().clamp_min(1.0))
    return torch.stack(losses).mean()


def silence_floor_loss(
    gain: torch.Tensor,
    coef: torch.Tensor,
    mix_refs: torch.Tensor,
    clean_refs: torch.Tensor,
    cfg: FeatureConfig,
    silence_threshold: float,
) -> torch.Tensor:
    """Force clean-speech silence bins toward a lower output floor."""
    window = torch.hann_window(cfg.n_fft, device=gain.device, dtype=mix_refs.dtype)
    losses = []
    threshold = float(max(silence_threshold, 1e-4))
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
        clean_mag = clean_spec.abs()
        frame_ref = clean_mag.mean(dim=0, keepdim=True).clamp_min(1e-5)
        silence_weight = torch.clamp((threshold * frame_ref - clean_mag) / (threshold * frame_ref), 0.0, 1.0)
        losses.append((enh_spec.abs() * silence_weight).sum() / silence_weight.sum().clamp_min(1.0))
    return torch.stack(losses).mean()


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
    residual_noise_weight: float,
    residual_noise_threshold: float,
    silence_floor_weight: float,
    silence_threshold: float,
    teacher_mwf: bool,
    teacher_mask_weight: float,
    teacher_blend: float,
    oracle_window: int,
    oracle_causal: bool,
    diagonal_loading: float,
) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, leave=False):
            if teacher_mwf:
                feats, masks, valid, mix_refs, clean_refs, _, mix_pairs, clean_pairs = batch
            else:
                feats, masks, valid, mix_refs, clean_refs, _ = batch
            feats = feats.to(device)
            masks = masks.to(device)
            valid = valid.to(device)
            mix_refs = mix_refs.to(device)
            clean_refs = clean_refs.to(device)
            if teacher_mwf:
                mix_pairs = mix_pairs.to(device)
                clean_pairs = clean_pairs.to(device)
                with torch.no_grad():
                    target_refs = build_mwf_teacher_targets(
                        mix_pairs,
                        clean_pairs,
                        clean_refs,
                        cfg,
                        oracle_window,
                        oracle_causal,
                        diagonal_loading,
                        teacher_blend,
                    )
                    masks = build_target_masks(mix_refs, target_refs, cfg, masks.shape[1], masks.shape[2])
            else:
                target_refs = clean_refs
            gain, coef = model(feats)
            loss = teacher_mask_weight * masked_mse(gain, masks, valid)
            if waveform_weight > 0.0:
                loss = loss + waveform_weight * waveform_l1_loss(gain, coef, mix_refs, target_refs, cfg)
            if stft_weight > 0.0:
                loss = loss + stft_weight * stft_logmag_loss(gain, coef, mix_refs, target_refs, cfg)
            if sisdr_weight > 0.0:
                loss = loss + sisdr_weight * si_sdr_loss(gain, coef, mix_refs, target_refs, cfg)
            if coef_reg_weight > 0.0:
                loss = loss + coef_reg_weight * coef_energy_loss(coef)
            if residual_noise_weight > 0.0:
                loss = loss + residual_noise_weight * residual_noise_loss(
                    gain,
                    coef,
                    mix_refs,
                    target_refs,
                    cfg,
                    residual_noise_threshold,
                )
            if silence_floor_weight > 0.0:
                loss = loss + silence_floor_weight * silence_floor_loss(
                    gain,
                    coef,
                    mix_refs,
                    target_refs,
                    cfg,
                    silence_threshold,
                )
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
    parser.add_argument("--virtual-multiplier", type=int, default=1)
    parser.add_argument("--snr-min-db", type=float, default=-5.0)
    parser.add_argument("--snr-max-db", type=float, default=15.0)
    parser.add_argument("--noise-mix-prob", type=float, default=0.0)
    parser.add_argument("--mic-distance-min-m", type=float, default=0.014)
    parser.add_argument("--mic-distance-max-m", type=float, default=0.022)
    parser.add_argument("--self-noise-prob", type=float, default=0.0)
    parser.add_argument("--self-noise-db", type=float, default=-36.0)
    parser.add_argument("--wind-noise-prob", type=float, default=0.0)
    parser.add_argument("--wind-noise-db", type=float, default=-22.0)
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
    parser.add_argument("--residual-noise-loss-weight", type=float, default=0.10)
    parser.add_argument("--residual-noise-threshold", type=float, default=0.08)
    parser.add_argument("--silence-floor-weight", type=float, default=0.12)
    parser.add_argument("--silence-threshold", type=float, default=0.03)
    parser.add_argument("--teacher-mwf", action="store_true", help="Distill from an oracle local MWF teacher during training.")
    parser.add_argument("--teacher-mask-weight", type=float, default=1.0)
    parser.add_argument("--teacher-blend", type=float, default=0.85, help="Blend teacher waveform with clean mic0 target.")
    parser.add_argument("--oracle-window", type=int, default=9)
    parser.add_argument("--oracle-causal", action="store_true")
    parser.add_argument("--diagonal-loading", type=float, default=0.01)
    parser.add_argument("--resume", help="Optional checkpoint to resume DeepFilter training from.")
    parser.add_argument("--reset-best-on-resume", action="store_true")
    parser.add_argument(
        "--spatial-frontend",
        choices=["delay_sum", "coherence_mwf"],
        default="delay_sum",
        help="Mono spatial input used as the DeepFilter reference.",
    )
    parser.add_argument("--min-gain", type=float, default=0.02,
                        help="Minimum mask gain; lower values allow stronger noise suppression.")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = FeatureConfig(spatial_features=True, spatial_frontend=args.spatial_frontend, min_gain=args.min_gain)
    train_ds = WavPairDataset(
        args.data,
        "train",
        cfg,
        args.seconds,
        args.on_the_fly,
        return_audio=True,
        return_mix_pair=args.teacher_mwf,
        virtual_multiplier=args.virtual_multiplier,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        noise_mix_prob=args.noise_mix_prob,
        mic_distance_min_m=args.mic_distance_min_m,
        mic_distance_max_m=args.mic_distance_max_m,
        self_noise_prob=args.self_noise_prob,
        self_noise_db=args.self_noise_db,
        wind_noise_prob=args.wind_noise_prob,
        wind_noise_db=args.wind_noise_db,
    )
    val_ds = WavPairDataset(
        args.data,
        "val",
        cfg,
        args.seconds,
        args.on_the_fly,
        return_audio=True,
        return_mix_pair=args.teacher_mwf,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        noise_mix_prob=args.noise_mix_prob,
        mic_distance_min_m=args.mic_distance_min_m,
        mic_distance_max_m=args.mic_distance_max_m,
        self_noise_prob=args.self_noise_prob,
        self_noise_db=args.self_noise_db,
        wind_noise_prob=args.wind_noise_prob,
        wind_noise_db=args.wind_noise_db,
    )
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
    best = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if ckpt.get("val_loss") is not None and not args.reset_best_on_resume:
            best = float(ckpt["val_loss"])
        print(f"resumed_from={args.resume}")
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
    print(f"spatial_frontend={cfg.spatial_frontend}")
    if args.teacher_mwf:
        print(
            "teacher=oracle_local_mwf "
            f"window={args.oracle_window} causal={args.oracle_causal} "
            f"diagonal_loading={args.diagonal_loading} blend={args.teacher_blend}"
        )
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
            args.residual_noise_loss_weight,
            args.residual_noise_threshold,
            args.silence_floor_weight,
            args.silence_threshold,
            args.teacher_mwf,
            args.teacher_mask_weight,
            args.teacher_blend,
            args.oracle_window,
            args.oracle_causal,
            args.diagonal_loading,
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
            args.residual_noise_loss_weight,
            args.residual_noise_threshold,
            args.silence_floor_weight,
            args.silence_threshold,
            args.teacher_mwf,
            args.teacher_mask_weight,
            args.teacher_blend,
            args.oracle_window,
            args.oracle_causal,
            args.diagonal_loading,
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
                "channels": args.channels,
                "blocks": args.blocks,
                "kernel_size": args.kernel_size,
                "df_bins": args.df_bins,
                "df_order": args.df_order,
                "coef_scale": args.coef_scale,
                "model_type": "tiny_deepfilter_tcn",
                "waveform_loss_weight": args.waveform_loss_weight,
                "stft_mag_loss_weight": args.stft_mag_loss_weight,
                "si_sdr_loss_weight": args.si_sdr_loss_weight,
                "coef_reg_weight": args.coef_reg_weight,
                "residual_noise_loss_weight": args.residual_noise_loss_weight,
                "residual_noise_threshold": args.residual_noise_threshold,
                "silence_floor_weight": args.silence_floor_weight,
                "silence_threshold": args.silence_threshold,
                "teacher_mwf": args.teacher_mwf,
                "teacher_mask_weight": args.teacher_mask_weight,
                "teacher_blend": args.teacher_blend,
                "oracle_window": args.oracle_window,
                "oracle_causal": args.oracle_causal,
                "diagonal_loading": args.diagonal_loading,
                "min_gain": cfg.min_gain,
                "max_gain": cfg.max_gain,
                "virtual_multiplier": args.virtual_multiplier,
                "snr_min_db": args.snr_min_db,
                "snr_max_db": args.snr_max_db,
                "noise_mix_prob": args.noise_mix_prob,
                "mic_distance_min_m": args.mic_distance_min_m,
                "mic_distance_max_m": args.mic_distance_max_m,
                "self_noise_prob": args.self_noise_prob,
                "self_noise_db": args.self_noise_db,
                "wind_noise_prob": args.wind_noise_prob,
                "wind_noise_db": args.wind_noise_db,
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
