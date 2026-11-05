"""Solar production task. Target y = production / (eta * clear-sky GHI), see
data/solar_features.py; metrics are computed in kWh per hour."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from data import DERIVED
from tasks.task import Task, lgbm_space

FEATURES = ["k_gefs", "k_std_gefs", "k_ecmwf", "k_std_ecmwf", "h", "cs_ghi"]


def scoring_table(df: pd.DataFrame, y_hat: NDArray[np.float64]) -> pd.DataFrame:
    """Scales model output back to kWh per hour. Night rows (y undefined) get 0. The naive
    forecast is the GEFS clearness index applied directly, without a model."""
    scale = (df["eta"] * df["cs_ghi"]).to_numpy()
    day = df["y"].notna().to_numpy()
    return pd.DataFrame(
        {
            "issue_time": df["issue_time"].to_numpy(),
            "h": df["h"].to_numpy(),
            "y": df["solar_kwh"].to_numpy(),
            "y_hat": np.where(day, np.clip(y_hat, 0, None) * scale, 0.0),
            "y_naive": np.where(day, df["k_gefs"].fillna(0).to_numpy() * scale, 0.0),
        }
    )


SOLAR = Task(
    "solar",
    FEATURES,
    "y",
    scoring_table,
    lambda: pd.read_parquet(DERIVED / "solar_dataset.parquet"),
    lgbm_space,
)
