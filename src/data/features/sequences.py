"""Tensors of the neural models from data/derived/co2_sequences.npz. Scaling is fitted on
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
    y: NDArray[np.float32]  # (issues, 168): target, the factor relative to `base`
    base: NDArray[np.float32]  # (issues, 168)
    y_ref: NDArray[np.float32]  # (issues, 168): reference forecast of the factor
    table: NDArray[np.float32]  # (issues, 168, channels): inputs of the tabular models

    @classmethod
    def load(cls, path: str | Path) -> "Sequences":
        z = np.load(path)
        return cls(
            pd.to_datetime(z["issue_time"], utc=True),
            z["past"],
            z["future"],
            z["y"],
            z["base"],
            z["y_ref"],
            z["table"],
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

    def scaling(self, train: NDArray[np.bool_]) -> dict[str, NDArray]:
        """The statistics `loaders` standardises with, for inference after training."""
        return {
            "past_mean": np.nanmean(self.past[train], axis=(0, 1)),
            "past_std": np.nanstd(self.past[train], axis=(0, 1)),
            "future_mean": np.nanmean(self.future[train], axis=(0, 1)),
            "future_std": np.nanstd(self.future[train], axis=(0, 1)),
        }

    def table_loaders(
        self,
        train: NDArray[np.bool_],
        val: NDArray[np.bool_],
        hours: tuple[int, int],
        batch_size: int,
    ) -> tuple[DataLoader, DataLoader, float]:
        """One row per (issue, hour) for hours in [hours[0], hours[1]). Training rows
        without a target are dropped; validation rows stay in (issue, hour) order."""
        h = np.arange(*hours)
        y_scale = float(np.nanstd(self.y[train][:, h]))
        x = _scale(self.table[:, h], train)
        y = (self.y[:, h] / y_scale).astype(np.float32)

        def loader(xs: NDArray, ys: NDArray, shuffle: bool) -> DataLoader:
            data = TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))
            return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

        x_train, y_train = x[train].reshape(-1, x.shape[-1]), y[train].ravel()
        known = np.isfinite(y_train)
        x_val, y_val = x[val].reshape(-1, x.shape[-1]), np.nan_to_num(y[val]).ravel()
        return (
            loader(x_train[known], y_train[known], True),
            loader(x_val, y_val, False),
            y_scale,
        )
