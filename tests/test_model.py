import numpy as np
import torch
import pytest
from fz_cnn_wbilstm.model import FuzzyCNNWaveletBiLSTMAttention, morlet
from fz_cnn_wbilstm.metrics import regression_metrics


def test_morlet_function_matches_article():
    x = torch.tensor([0., 1., -1.])
    expected = torch.cos(1.75 * x) * torch.exp(-x.square() / 2)
    assert torch.allclose(morlet(x), expected)
    assert morlet(x)[0].item() == pytest.approx(1.)


def test_forward_output_and_attention_weights():
    torch.set_num_threads(2)
    net = FuzzyCNNWaveletBiLSTMAttention(8, conv_filters=(8, 12, 16),
                                         lstm_units=12, attention_units=24)
    out = net(torch.randn(3, 96, 8))
    assert out.shape == (3,)
    assert torch.isfinite(out).all()
    assert net._last_attention.shape[0] == 3
    assert torch.allclose(net._last_attention.sum(1), torch.ones(3), atol=1e-5)


def test_model_backprop():
    net = FuzzyCNNWaveletBiLSTMAttention(8, conv_filters=(8, 12, 16),
                                         lstm_units=12, attention_units=24)
    output = net(torch.randn(2, 96, 8)).square().mean()
    output.backward()
    assert any(p.grad is not None for p in net.parameters())


def test_daylight_mape_avoids_zero_singularity():
    report = regression_metrics([0, 100, 200], [0, 110, 190])
    assert report['mape_daylight_samples'] == 2
    assert report['MAPE_daylight_pct'] == pytest.approx(7.5)


def test_metrics_reject_shapes():
    with pytest.raises(ValueError):
        regression_metrics(np.ones(3), np.ones(2))


def test_published_model_dimensions():
    torch.set_num_threads(2)
    model = FuzzyCNNWaveletBiLSTMAttention(8)
    assert [block[0].out_channels for block in model.blocks] == [64, 128, 256]
    assert [block[0].kernel_size[0] for block in model.blocks] == [5, 5, 5]
    assert [block[2].kernel_size for block in model.blocks] == [3, 3, 3]
    assert model.lstm.hidden_size == 150
    assert model.attention[0].out_features == 300
    with torch.no_grad():
        y = model(torch.randn(2, 96, 8))
    assert y.shape == (2,)
