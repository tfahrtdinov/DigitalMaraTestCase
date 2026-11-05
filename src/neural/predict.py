"""Forecast with saved LSTM checkpoints (neural.run --save): the mean over the seeds.

    checkpoints = load(sorted(Path("models").glob("co2_lstm_h0-48_seed*.pt")))
    co2 = predict(checkpoints, past, future, base)

past (issues, 336, channels) and future (issues, 168, channels) are unscaled, in the layout of
data/derived/co2_sequences.npz. base (issues, hours) is what the target is relative to: the
factor in the last hour before issue for hours 0-47, the 28-day profile for hours 48-167.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from models.nets import NETS


def load(paths: Iterable[Path | str]) -> list[dict]:
    return [torch.load(p, weights_only=False) for p in paths]


def _scaled(values: NDArray, mean: NDArray, std: NDArray) -> torch.Tensor:
    scaled = (values - mean) / np.where(std > 0, std, 1.0)
    return torch.from_numpy(np.nan_to_num(scaled).astype(np.float32))


@torch.no_grad()
def predict(
    checkpoints: list[dict], past: NDArray, future: NDArray, base: NDArray
) -> NDArray[np.float64]:
    """CO2 factor in g/kWh for the hours of the checkpoints, (issues, hours)."""
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
