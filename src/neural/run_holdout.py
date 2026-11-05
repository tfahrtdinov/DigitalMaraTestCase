"""PYTHONPATH=src python -m neural.run_holdout fitted/co2_lstm_h0-48
    --start 2026-08-28 --end 2026-10-27 [--data data/derived/co2_sequences.npz] [--console]

Loads the saved networks of a path prefix (neural.train), averages their forecasts over the
issues of [start, end) and scores them. Nothing is trained. Forecasts go to
data/derived/forecasts/<name>_<start>_<end>.parquet."""

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

import neural
from data.features.sequences import Sequences
from loggers import ConsoleLogger, MLflowLogger
from validation import evaluate


def forecast(
    checkpoints: list[dict], data: Sequences, start: str, end: str
) -> pd.DataFrame:
    """Table of forecasts: issue_time, h, y, y_hat, y_naive."""
    issues = np.asarray(
        (data.issue_time >= pd.Timestamp(start, tz="UTC"))
        & (data.issue_time < pd.Timestamp(end, tz="UTC"))
    )
    h = np.arange(*checkpoints[0]["hours"])
    base = data.base[issues][:, h]
    if checkpoints[0]["net"] == "mlp":
        y_hat = neural.predict_table(checkpoints, data.table[issues][:, h], base)
    else:
        y_hat = neural.predict(
            checkpoints, data.past[issues], data.future[issues], base
        )
    table = pd.DataFrame(
        {
            "issue_time": np.repeat(data.issue_time[issues], len(h)),
            "h": np.tile(h, issues.sum()),
            "y": (data.y[issues][:, h] + base).ravel(),
            "y_hat": y_hat.ravel(),
            "y_naive": data.y_ref[issues][:, h].ravel(),
        }
    )
    return table.dropna(subset=["y"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("models", help="path prefix of the checkpoints")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--data", default="data/derived/co2_sequences.npz")
    p.add_argument("--out", default="data/derived/forecasts")
    p.add_argument("--console", action="store_true", help="print instead of MLflow")
    a = p.parse_args()

    checkpoints = neural.load(sorted(glob(f"{a.models}_seed*.pt")))
    table = forecast(checkpoints, Sequences.load(a.data), a.start, a.end)
    name = f"{Path(a.models).name}_{a.start}_{a.end}"
    logger = ConsoleLogger(name) if a.console else MLflowLogger("neural", name)
    with logger:
        evaluate(table, logger)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    table.to_parquet(Path(a.out) / f"{name}.parquet", index=False)


if __name__ == "__main__":
    main()
