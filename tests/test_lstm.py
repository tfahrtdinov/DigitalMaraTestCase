from typing import cast

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import TensorDataset  # noqa: E402

from data.features.sequences import Sequences  # noqa: E402
from models.nets import NETS  # noqa: E402
from models.neural import MODELS  # noqa: E402
from trainer import Trainer  # noqa: E402


def sequences(n=40, hours=168):
    rng = np.random.default_rng(0)
    y = rng.normal(size=(n, hours)).astype(np.float32)
    y[:, -1] = np.nan
    future = np.dstack([y, rng.normal(size=(n, hours))]).astype(np.float32)
    return Sequences(
        pd.date_range("2025-01-01", periods=n, tz="UTC"),
        rng.normal(size=(n, 24, 3)).astype(np.float32),
        np.nan_to_num(future),
        y,
        rng.normal(size=n).astype(np.float32),
        np.zeros((n, hours), np.float32),
    )


def fit(data, hours=(0, 48), epochs=30):
    train = np.arange(len(data.y)) < 30
    train_loader, val_loader, _ = data.loaders(train, ~train, hours, 8)
    net = NETS["lstm"](3, 2, hidden=16, layers=1, dropout=0.0)
    model = MODELS["rnn"](net, lr=0.01, weight_decay=0.0)
    Trainer(epochs, grad_clip=1.0, one_cycle=True).fit(model, train_loader, val_loader)
    return model, train_loader, val_loader


def test_lstm_gives_one_value_per_forecast_hour():
    net = NETS["lstm"](3, 2, hidden=8, layers=2)
    assert net(torch.zeros(4, 24, 3), torch.zeros(4, 168, 2)).shape == (4, 168)


def test_training_learns_from_future_inputs():
    data = sequences()
    model, _, val_loader = fit(data)
    assert Trainer.validate(model, val_loader) < 0.4  # guessing zero gives about 0.8


def test_only_masked_hours_enter_the_loss():
    data = sequences()
    train = np.arange(len(data.y)) < 30
    loader, _, _ = data.loaders(train, ~train, (0, 48), 8)
    _, _, y, mask = next(iter(loader))
    assert mask[:, :48].all() and not mask[:, 48:].any()
    assert torch.isfinite(y).all()


def test_scaling_uses_training_issues_only():
    data = sequences()
    train = np.arange(len(data.y)) < 30
    loader_a, _, scale_a = data.loaders(train, ~train, (0, 48), 8)
    data.past[~train] += 1000.0
    data.future[~train] += 1000.0
    data.y[~train] += 1000.0
    loader_b, _, scale_b = data.loaders(train, ~train, (0, 48), 8)
    assert scale_a == scale_b
    tensors_a = cast(TensorDataset, loader_a.dataset).tensors
    tensors_b = cast(TensorDataset, loader_b.dataset).tensors
    for a, b in zip(tensors_a, tensors_b, strict=True):
        assert torch.equal(a, b)


def test_predict_returns_numpy_for_all_validation_issues():
    data = sequences()
    model, _, val_loader = fit(data, epochs=1)
    assert Trainer.predict(model, val_loader).shape == (10, 168)
