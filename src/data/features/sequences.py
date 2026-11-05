"""Tensors of the RNN models from data/derived/co2_sequences.npz. Scaling is fitted on
the training issues of each fold."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset


def _scale(values: NDArray, train: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Standardises per channel over the training issues; missing values become 0."""
    axes = (0, 1)
    mean = np.nanmean(values[train], axis=axes, keepdims=True)
    std = np.nanstd(values[train], axis=axes, keepdims=True)
    scaled = (values - mean) / np.where(std > 0, std, 1.0)
    return np.nan_to_num(scaled).astype(np.float32)


@dataclass
class Sequences:
    issue_time: pd.DatetimeIndex
    past: NDArray[np.float32]  # (issues, past hours, channels)
    future: NDArray[np.float32]  # (issues, 168, channels)
    y: NDArray[np.float32]  # (issues, 168): change of the factor from the last hour
    co2_last: NDArray[np.float32]  # (issues,)
    y_ref: NDArray[np.float32]  # (issues, 168): reference forecast of the factor

    @classmethod
    def load(cls, path: str | Path) -> "Sequences":
        z = np.load(path)
        return cls(
            pd.to_datetime(z["issue_time"], utc=True),
            z["past"],
            z["future"],
            z["y"],
            z["co2_last"],
            z["y_ref"],
        )

    def loaders(
        self,
        train: NDArray[np.bool_],
        val: NDArray[np.bool_],
        hours: tuple[int, int],
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader, float]:
        """Training and validation loaders, and the scale of the target. Only the hours in
        [hours[0], hours[1]) enter the loss, and all 168 enter the decoder."""
        selected = np.zeros(self.y.shape[1], bool)
        selected[hours[0] : hours[1]] = True
        y_scale = float(np.nanstd(self.y[train][:, selected]))
        past, future = _scale(self.past, train), _scale(self.future, train)
        y = np.nan_to_num(self.y / y_scale).astype(np.float32)
        mask = (np.isfinite(self.y) & selected).astype(np.float32)

        def loader(issues: NDArray[np.bool_], shuffle: bool) -> DataLoader:
            arrays = (past[issues], future[issues], y[issues], mask[issues])
            data = TensorDataset(*map(torch.from_numpy, arrays))
            return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

        return loader(train, True), loader(val, False), y_scale
