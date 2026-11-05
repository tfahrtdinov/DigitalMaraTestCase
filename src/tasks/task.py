from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from numpy.typing import NDArray

from models import ModelConfig


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
