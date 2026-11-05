from typing import cast

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import TensorDataset  # noqa: E402

from config import TorchConfig  # noqa: E402
from data.features.sequences import Sequences  # noqa: E402
from loggers import ConsoleLogger  # noqa: E402
from neural import fit as fit_network  # noqa: E402
from neural import load, save  # noqa: E402
from neural.models import MODELS  # noqa: E402
from neural.nets import NETS  # noqa: E402
from neural.run_holdout import forecast  # noqa: E402
from neural.trainer import Trainer  # noqa: E402


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
        rng.normal(size=(n, hours)).astype(np.float32),
        np.zeros((n, hours), np.float32),
        np.nan_to_num(future),
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


def test_table_loaders_keep_issue_and_hour_order_and_drop_unknown_targets():
    data = sequences()
    train = np.arange(len(data.y)) < 30
    train_loader, val_loader, _ = data.table_loaders(train, ~train, (160, 168), 16)
    assert len(cast(TensorDataset, train_loader.dataset)) == 30 * 7
    assert len(cast(TensorDataset, val_loader.dataset)) == 10 * 8
    net = NETS["mlp"](2, hidden=[8])
    model = MODELS["regression"](net)
    Trainer(1).fit(model, train_loader, val_loader)
    assert Trainer.predict(model, val_loader).reshape(-1, 8).shape == (10, 8)


@pytest.mark.parametrize(
    ("net", "params", "model"),
    [
        ("lstm", {"hidden": 8, "layers": 1, "dropout": 0.0}, "rnn"),
        ("mlp", {"hidden": [8]}, "regression"),
    ],
)
def test_holdout_forecast_comes_from_the_saved_networks(tmp_path, net, params, model):
    data = sequences()
    train = np.arange(len(data.y)) < 30
    config = TorchConfig(
        net=net, net_params=params, model=model, epochs=1, batch_size=8
    )
    tabular = net == "mlp"
    loaders = data.table_loaders if tabular else data.loaders
    rows, val_rows, y_scale = loaders(train, ~train, (0, 48), 8)
    sizes = (data.table.shape[-1],) if tabular else (3, 2)
    scaling = data.table_scaling(train, (0, 48)) if tabular else data.scaling(train)
    fitted = fit_network(config, sizes, rows, val_rows, 0, "cpu", ConsoleLogger("t"))
    save(tmp_path / "m_seed0.pt", fitted, config, sizes, (0, 48), y_scale, "", scaling)

    table = forecast(load([tmp_path / "m_seed0.pt"]), data, "2025-01-31", "2025-02-10")
    predicted = Trainer.predict(fitted, val_rows).reshape(10, -1)[:, :48] * y_scale
    expected = np.clip(predicted + data.base[~train][:, :48], 0, None)
    assert np.allclose(table.y_hat.to_numpy(), expected.ravel(), atol=1e-3)
