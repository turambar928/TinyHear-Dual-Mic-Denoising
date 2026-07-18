from __future__ import annotations

import torch

from ha_denoise.features import FeatureConfig, extract_features, target_band_mask
from ha_denoise.model import TinyCausalTCN, count_parameters
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


def test_feature_shapes() -> None:
    cfg = FeatureConfig()
    wav = torch.randn(2, 16000)
    clean = wav[0] * 0.5
    feat = extract_features(wav, cfg)
    mask = target_band_mask(wav[0], clean, cfg)
    assert feat.shape[1] == 96
    assert mask.shape[1] == 32
    assert abs(feat.shape[0] - mask.shape[0]) <= 1


def test_streaming_model_matches_batch_model() -> None:
    torch.manual_seed(2026)
    model = TinyCausalTCN()
    model.eval()
    x = torch.randn(9, 96)
    with torch.no_grad():
        batch = model(x.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        streaming = run_streaming_model(model, x)
    assert torch.allclose(streaming, batch, atol=1e-6)
