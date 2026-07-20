from __future__ import annotations

import torch

from ha_denoise.features import FeatureConfig, extract_features, pad_sequence_batch, target_band_mask
from ha_denoise.model import TinyCausalTCN, count_parameters
from ha_denoise.realtime import StreamingDenoiser
from ha_denoise.streaming import run_streaming_model


def test_model_size_and_shape() -> None:
    model = TinyCausalTCN()
    params = count_parameters(model)
    assert 100_000 <= params <= 150_000
    x = torch.randn(2, 96, 12)
    y = model(x)
    assert y.shape == (2, 32, 12)
    assert float(y.detach().min()) >= 0.0
    assert float(y.detach().max()) <= 1.0


def test_larger_model_stays_inside_target_size() -> None:
    model = TinyCausalTCN(channels=120)
    params = count_parameters(model)
    assert 100_000 <= params <= 150_000


def test_spatial_larger_model_stays_inside_target_size() -> None:
    model = TinyCausalTCN(feature_dim=192, channels=120)
    params = count_parameters(model)
    assert 100_000 <= params <= 150_000


def test_feature_shapes() -> None:
    cfg = FeatureConfig()
    wav = torch.randn(2, 16000)
    clean = wav[0] * 0.5
    feat = extract_features(wav, cfg)
    mask = target_band_mask(wav[0], clean, cfg)
    assert feat.shape[1] == 96
    assert mask.shape[1] == 32
    assert abs(feat.shape[0] - mask.shape[0]) <= 1


def test_spatial_feature_shapes() -> None:
    cfg = FeatureConfig(spatial_features=True)
    wav = torch.randn(2, 16000)
    feat = extract_features(wav, cfg)
    assert cfg.feature_dim == 192
    assert feat.shape[1] == 192
    assert torch.isfinite(feat).all()


def test_streaming_model_matches_batch_model() -> None:
    torch.manual_seed(2026)
    model = TinyCausalTCN()
    model.eval()
    x = torch.randn(9, 96)
    with torch.no_grad():
        batch = model(x.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        streaming = run_streaming_model(model, x)
    assert torch.allclose(streaming, batch, atol=1e-6)


def test_realtime_denoiser_shapes() -> None:
    model = TinyCausalTCN()
    model.eval()
    cfg = FeatureConfig()
    denoiser = StreamingDenoiser(model, cfg)
    hop = torch.randn(2, cfg.hop_length)
    out_hop = denoiser.process_hop(hop)
    assert out_hop.shape == (cfg.hop_length,)
    wav = torch.randn(2, cfg.sample_rate // 10)
    enhanced = denoiser.process(wav, flush=True)
    assert enhanced.numel() == wav.shape[1] + cfg.n_fft
    assert torch.isfinite(enhanced).all()


def test_collate_with_audio() -> None:
    feat = torch.randn(4, 96)
    mask = torch.rand(4, 32)
    mix = torch.randn(256)
    clean = torch.randn(256)
    batch = pad_sequence_batch([(feat, mask, mix, clean)])
    assert len(batch) == 6
    assert batch[0].shape == (1, 96, 4)
    assert batch[3].shape == (1, 256)
