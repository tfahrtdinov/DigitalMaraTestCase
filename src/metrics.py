"""Forecast metrics on a table with columns y (actual), y_hat (forecast), h (hour within the
168-hour window) and issue_time. Optional y_naive adds the improvement over a reference
forecast: 1 - MAE / MAE_naive, in percent."""

from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr


def mae(y: NDArray[np.float64], y_hat: NDArray[np.float64]) -> float:
    return float(np.mean(np.abs(y_hat - y)))


def rmse(y: NDArray[np.float64], y_hat: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean((y_hat - y) ** 2)))


def bias(y: NDArray[np.float64], y_hat: NDArray[np.float64]) -> float:
    return float(np.mean(y_hat - y))


def improvement(
    y: NDArray[np.float64], y_hat: NDArray[np.float64], y_naive: NDArray[np.float64]
) -> float:
    return float(100 * (1 - mae(y, y_hat) / mae(y, y_naive)))


def spearman_within_window(df: pd.DataFrame) -> float:
    """Rank correlation between y and y_hat inside each 168-hour window, averaged."""
    corr = [
        cast(float, spearmanr(w["y"].to_numpy(), w["y_hat"].to_numpy()).statistic)
        for _, w in df.groupby("issue_time")
    ]
    return float(np.mean(corr))


def summary(df: pd.DataFrame) -> pd.Series:
    """Overall metrics, then MAE per forecast day d1..d7 (h // 24)."""
    y, y_hat = df["y"].to_numpy(), df["y_hat"].to_numpy()
    out = {"mae": mae(y, y_hat), "rmse": rmse(y, y_hat), "bias": bias(y, y_hat)}
    if "y_naive" in df:
        out["improvement_pct"] = improvement(y, y_hat, df["y_naive"].to_numpy())
    if "issue_time" in df:
        out["spearman"] = spearman_within_window(df)
    day = df["h"].to_numpy() // 24
    for d in np.unique(day):
        out[f"mae_d{d + 1}"] = mae(y[day == d], y_hat[day == d])
    return pd.Series(out)
