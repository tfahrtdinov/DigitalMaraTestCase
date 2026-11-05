"""PYTHONPATH=src uv run python -m lgbm.run_holdout co2 fitted/co2_lgbm_h0-48.txt --hours 0 48
    --start 2026-08-28 --end 2026-10-27 [--console]

Loads a saved LightGBM model (lgbm.train), forecasts the issues of [start, end) and scores
them. Nothing is trained. Forecasts go to data/derived/forecasts/<model>_<start>_<end>.parquet.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data import DERIVED
from lgbm.run_hpo import TASKS
from loggers import ConsoleLogger, MLflowLogger
from validation import evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=TASKS)
    p.add_argument("model")
    p.add_argument("--hours", type=int, nargs=2, default=[0, 168])
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--console", action="store_true", help="print instead of MLflow")
    a = p.parse_args()

    task = TASKS[a.task]
    df = task.load()
    rows = df.loc[
        (df.h >= a.hours[0])
        & (df.h < a.hours[1])
        & (df.issue_time >= pd.Timestamp(a.start, tz="UTC"))
        & (df.issue_time < pd.Timestamp(a.end, tz="UTC"))
    ]
    booster = lgb.Booster(model_file=a.model)
    table = task.scoring_table(rows, np.asarray(booster.predict(task.X(rows))))
    name = f"{Path(a.model).stem}_{a.start}_{a.end}"
    logger = ConsoleLogger(name) if a.console else MLflowLogger("holdout", name)
    with logger:
        evaluate(table, logger)
    table.to_parquet(DERIVED / "forecasts" / f"{name}.parquet", index=False)


if __name__ == "__main__":
    main()
