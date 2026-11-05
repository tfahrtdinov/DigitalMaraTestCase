"""Solar production task. Target y = production / (eta * clear-sky GHI), see
data/solar_features.py; metrics are computed in kWh per hour."""

import numpy as np
import optuna
import pandas as pd
from numpy.typing import NDArray

from data import DERIVED
from models import LGBMConfig
from tasks.task import Task

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


def space(trial: optuna.trial.BaseTrial) -> LGBMConfig:
    """LightGBM search space."""
    return LGBMConfig(
        n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 300, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    )


SOLAR = Task(
    "solar",
    FEATURES,
    "y",
    scoring_table,
    lambda: pd.read_parquet(DERIVED / "solar_dataset.parquet"),
    space,
)
