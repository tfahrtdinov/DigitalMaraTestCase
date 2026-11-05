"""PYTHONPATH=src uv run python -m lgbm.run_folds solar [--start 2025-01-01 --end 2026-01-01
    --splits 12] [--hours 0 168] [--console]

Out-of-sample forecasts with the tuned LightGBM configs: [start, end) is cut into --splits
equal folds, each forecast by a model refitted on everything before it minus the 7-day gap,
one model per hour segment of --hours (edges). To keep a model use lgbm.train. The defaults are the 2025 test and write data/derived/forecasts/<task>_lgbm.parquet;
another range gets the suffix _<start>_<end>. Metrics go to MLflow (experiment "holdout")
unless --console is given.
"""

import argparse
import json

import pandas as pd

import lgbm
from config import LGBMConfig
from data import DERIVED
from lgbm.run_hpo import ROOT_CONFIGS, TASKS
from loggers import ConsoleLogger, MLflowLogger
from validation import Fold, evaluate, folds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=TASKS)
    p.add_argument("--hours", type=int, nargs="+", default=[0, 168])
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--splits", type=int, default=12)
    p.add_argument("--console", action="store_true", help="print instead of MLflow")
    a = p.parse_args()

    task = TASKS[a.task]
    df = task.load()
    name = f"{task.name}_lgbm"
    if (a.start, a.end) != ("2025-01-01", "2026-01-01"):
        name += f"_{a.start}_{a.end}"
    tables = []
    for start, end in zip(a.hours[:-1], a.hours[1:], strict=True):
        with open(
            f"{ROOT_CONFIGS}/{lgbm.config_name(task.name, start, end)}.json"
        ) as f:
            config = LGBMConfig(**json.load(f))
        rows = df.loc[(df.h >= start) & (df.h < end)]
        tables += [
            forecast(task, config, rows, fold)
            for fold in folds(a.start, a.end, a.splits)
        ]
    table = pd.concat(tables).sort_values(["issue_time", "h"])
    (DERIVED / "forecasts").mkdir(parents=True, exist_ok=True)
    table.to_parquet(DERIVED / "forecasts" / f"{name}.parquet", index=False)
    logger = ConsoleLogger(name) if a.console else MLflowLogger("holdout", name)
    with logger:
        logger.log_params(
            {"hours": a.hours, "start": a.start, "end": a.end, "splits": a.splits}
        )
        evaluate(table, logger)


def forecast(task, config: LGBMConfig, df: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    fit_rows, val_rows = task.train_rows(fold.train(df)), fold.val(df)
    booster = lgbm.fit(config, task.X(fit_rows), task.y(fit_rows))
    return task.scoring_table(val_rows, lgbm.predict(booster, task.X(val_rows)))


if __name__ == "__main__":
    main()
