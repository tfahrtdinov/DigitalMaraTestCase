"""Training of one network, its checkpoint, and the forecast from checkpoints.

checkpoints = load(sorted(Path("fitted").glob("co2_lstm_h0-48_seed*.pt")))
co2 = predict(checkpoints, past, future, base)
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from config import TorchConfig
from loggers import Logger
from neural.models import MODELS, NeuralModel
from neural.nets import NETS
from neural.trainer import Trainer


def fit(
    config: TorchConfig,
    sizes: tuple[int, ...],
    train: DataLoader,
    val: DataLoader,
    seed: int,
    device: str,
    logger: Logger,
) -> NeuralModel:
    """Trains the network of `config` with one seed; `sizes` are its input channels."""
    torch.manual_seed(seed)
    net = NETS[config.net](*sizes, **config.net_params)
    model = MODELS[config.model](net, **config.model_params)
    return Trainer(
        config.epochs,
        config.threads,
        seed,
        logger,
        device,
        config.grad_clip,
        config.one_cycle,
    ).fit(model, train, val)


def save(
    path: Path,
    model: NeuralModel,
    config: TorchConfig,
    sizes: tuple[int, ...],
    hours: tuple[int, int],
    y_scale: float,
    train_end: str,
    scaling: dict[str, NDArray],
) -> None:
    """Writes the weights with all that `predict` needs to use them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu() for k, v in model.net.state_dict().items()}
    torch.save(
        {
            "net": config.net,
            "net_params": config.net_params,
            "sizes": sizes,
            "hours": hours,
            "state_dict": state,
            "y_scale": y_scale,
            "train_end": train_end,
            **scaling,
        },
        path,
    )


def load(paths: Iterable[Path | str]) -> list[dict]:
    return [torch.load(p, weights_only=False) for p in paths]


def _scaled(values: NDArray, mean: NDArray, std: NDArray) -> torch.Tensor:
    scaled = (values - mean) / np.where(std > 0, std, 1.0)
    return torch.from_numpy(np.nan_to_num(scaled).astype(np.float32))


@torch.no_grad()
def predict(
    checkpoints: list[dict], past: NDArray, future: NDArray, base: NDArray
) -> NDArray[np.float64]:
    """CO2 factor in g/kWh for the hours of the checkpoints, mean over them, (issues, hours).

    past (issues, 336, channels) and future (issues, 168, channels) are unscaled, in the
    layout of data/derived/co2_sequences.npz. base (issues, hours) is what the target is
    relative to: the factor in the last hour before issue for hours 0-47, the 28-day
    profile for hours 48-167."""
    outputs = []
    for c in checkpoints:
        net = NETS[c["net"]](*c["sizes"], **c["net_params"])
        net.load_state_dict(c["state_dict"])
        net.eval()
        x_past = _scaled(past, c["past_mean"], c["past_std"])
        x_future = _scaled(future, c["future_mean"], c["future_std"])
        first, last = c["hours"]
        outputs.append(net(x_past, x_future)[:, first:last].numpy() * c["y_scale"])
    return np.clip(np.mean(outputs, axis=0) + base, 0, None)


@torch.no_grad()
def predict_table(
    checkpoints: list[dict], table: NDArray, base: NDArray
) -> NDArray[np.float64]:
    """As `predict` for an MLP: table is (issues, hours, channels), unscaled."""
    outputs = []
    for c in checkpoints:
        net = NETS[c["net"]](*c["sizes"], **c["net_params"])
        net.load_state_dict(c["state_dict"])
        net.eval()
        x = _scaled(table, c["table_mean"], c["table_std"])
        rows = net(x.reshape(-1, x.shape[-1])).numpy()
        outputs.append(rows.reshape(base.shape) * c["y_scale"])
    return np.clip(np.mean(outputs, axis=0) + base, 0, None)
