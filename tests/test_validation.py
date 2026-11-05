import numpy as np
import pandas as pd

import validation
from loggers import ConsoleLogger


class MemoryLogger(ConsoleLogger):
    def __init__(self):
        super().__init__("test")
        self.records = []

    def log_metrics(self, metrics, step=None):
        self.records.append((step, metrics))


def test_evaluate_logs_summary_with_prefix():
    h = np.arange(168)
    df = pd.DataFrame(
        {
            "issue_time": "2025-01-01",
            "h": h,
            "y": h,
            "y_hat": h + 2.0,
            "y_naive": h + 4.0,
        }
    )
    logger = MemoryLogger()
    scores = validation.evaluate(df, logger, prefix="solar/").to_dict()
    assert (scores["mae"], scores["improvement_pct"]) == (2.0, 50.0)
    assert logger.records == [
        (None, {f"solar/{k}": float(v) for k, v in scores.items()})
    ]


def test_expanding_folds():
    f = validation.folds("2024-01-01", "2025-01-01", n_splits=6, gap_days=7)
    assert len(f) == 6
    assert str(f[0].val_start) == "2024-01-01 00:00:00+00:00"
    assert str(f[0].train_end) == "2023-12-25 00:00:00+00:00"
    assert str(f[-1].val_end) == "2025-01-01 00:00:00+00:00"
    assert all(a.val_end == b.val_start for a, b in zip(f[:-1], f[1:], strict=True))


def test_folds_are_equal_whole_days():
    f = validation.folds("2024-01-01", "2024-03-01", n_splits=3)
    assert [str(x.val_end.date()) for x in f] == [
        "2024-01-21",
        "2024-02-10",
        "2024-03-01",
    ]
    assert all(x.val_start == x.val_start.floor("D") for x in f)


def test_split_respects_gap():
    t = pd.date_range("2023-12-01", "2024-03-31", freq="D", tz="UTC")
    df = pd.DataFrame({"issue_time": t, "h": 0})
    fold = validation.folds("2024-01-01", "2024-03-01", n_splits=1)[0]
    train, val = fold.train(df), fold.val(df)
    assert train.issue_time.max() == pd.Timestamp("2023-12-24", tz="UTC")
    assert val.issue_time.min() == pd.Timestamp("2024-01-01", tz="UTC")
    assert val.issue_time.max() == pd.Timestamp("2024-02-29", tz="UTC")


def test_periods_separate_validation_and_holdout():
    p = validation.Periods("2024-01-01", "2025-01-01", "2026-01-01")
    v, h = p.validation_folds(n_splits=6), p.holdout_folds(n_splits=12)
    assert len(v) == 6 and len(h) == 12
    assert v[-1].val_end == h[0].val_start == pd.Timestamp("2025-01-01", tz="UTC")
    assert str(h[-1].val_end.date()) == "2026-01-01"
