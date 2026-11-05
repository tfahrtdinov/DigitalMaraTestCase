from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from numpy.typing import NDArray

from config import LGBMConfig, ModelConfig


@dataclass(frozen=True)
class Task:
    """One forecasting target: where its dataset is, which columns are features and
    target, how model output becomes a table for the metrics, and which model
    configurations hyperparameter search may try."""

    name: str
    features: list[str]
    target: str
    # (dataset rows, model output for them) -> table for metrics.summary:
    # issue_time, h, y (actual), y_hat, y_naive, all in physical units
    scoring_table: Callable[[pd.DataFrame, NDArray[np.float64]], pd.DataFrame]
    load: Callable[[], pd.DataFrame]
    space: Callable[[optuna.trial.BaseTrial], ModelConfig]

    def train_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(subset=[self.target])

    def X(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, self.features]

    def y(self, df: pd.DataFrame) -> pd.Series:
        return df.loc[:, self.target]


def lgbm_space(trial: optuna.trial.BaseTrial) -> LGBMConfig:
    """LightGBM search space shared by the renewables tasks."""
    return LGBMConfig(
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 300, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    )
