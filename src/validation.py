"""Expanding-window folds and scoring of forecast tables."""

from dataclasses import dataclass

import pandas as pd

from loggers import Logger
from metrics import summary


@dataclass(frozen=True)
class Fold:
    train_end: pd.Timestamp  # exclusive; training uses everything before it
    val_start: pd.Timestamp  # inclusive
    val_end: pd.Timestamp  # exclusive

    def train(self, df: pd.DataFrame, column: str = "issue_time") -> pd.DataFrame:
        return df.loc[df[column] < self.train_end]

    def val(self, df: pd.DataFrame, column: str = "issue_time") -> pd.DataFrame:
        t = df[column]
        return df.loc[(t >= self.val_start) & (t < self.val_end)]


def folds(start: str, end: str, n_splits: int, gap_days: int = 7) -> list[Fold]:
    """`n_splits` equal consecutive blocks covering [start, end), each preceded by a
    `gap_days` gap; the training window grows from fold to fold."""
    edges = pd.date_range(start, end, periods=n_splits + 1, tz="UTC").floor("D")
    return [
        Fold(a - pd.Timedelta(days=gap_days), a, b)
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]


@dataclass(frozen=True)
class Periods:
    """Validation [val_start, holdout_start) for model selection; holdout [holdout_start, end)
    scored once at the end."""

    val_start: str
    holdout_start: str
    end: str
    gap_days: int = 7

    def validation_folds(self, n_splits: int) -> list[Fold]:
        return folds(self.val_start, self.holdout_start, n_splits, self.gap_days)

    def holdout_folds(self, n_splits: int) -> list[Fold]:
        return folds(self.holdout_start, self.end, n_splits, self.gap_days)


def evaluate(forecasts: pd.DataFrame, logger: Logger, prefix: str = "") -> pd.Series:
    """Scores a table with columns issue_time, h, y, y_hat (optional y_naive) and logs it."""
    scores = summary(forecasts)
    logger.log_metrics({f"{prefix}{k}": float(v) for k, v in scores.items()})
    return scores
