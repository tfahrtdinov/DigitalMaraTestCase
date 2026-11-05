"""Onshore and offshore wind tasks. Target y = production / cap, see data/features/wind.py;
metrics are computed in kWh per hour."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from data import DERIVED
from data.features.wind import power_curve
from tasks.task import Task, lgbm_space

FEATURES = [
    "ws_gefs",
    "ws_std_gefs",
    "ws_ecmwf",
    "ws_std_ecmwf",
    "h",
    "y_last",
    "y_1d",
    "err_1d",
]


def scoring_table(df: pd.DataFrame, y_hat: NDArray[np.float64]) -> pd.DataFrame:
    """Scales model output back to kWh per hour. The naive forecast is a cubic power curve
    on the GEFS wind speed, without a model."""
    cap = df["cap"].to_numpy()
    curve = power_curve(pd.Series(df["ws_gefs"])).to_numpy()
    return pd.DataFrame(
        {
            "issue_time": df["issue_time"].to_numpy(),
            "h": df["h"].to_numpy(),
            "y": df["wind_kwh"].to_numpy(),
            "y_hat": np.clip(y_hat, 0, None) * cap,
            "y_naive": curve * cap,
        }
    )


def task(group: str) -> Task:
    return Task(
        group,
        FEATURES,
        "y",
        scoring_table,
        lambda: pd.read_parquet(DERIVED / f"{group}_dataset.parquet"),
        lgbm_space,
    )


ONSHORE, OFFSHORE = task("onshore"), task("offshore")
