"""PYTHONPATH=src uv run python -m lgbm.run_holdout solar [--start 2025-01-01 --end 2026-01-01
    --splits 12] [--hours 0 168] [--console]

Out-of-sample forecasts with the tuned LightGBM configs: [start, end) is cut into
--splits equal folds, each forecast by a model refitted on everything before it minus the
7-day gap; one model per hour segment of --hours (edges). The defaults are the 2025 holdout
and write data/derived/forecasts/<task>_lgbm.parquet; another range gets the suffix
_<start>_<end>. --save writes the model of each segment; with several folds the last stays.
Metrics go to MLflow (experiment "holdout") unless --console is given.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import lgbm
from data import DERIVED
from lgbm.run_hpo import ROOT_CONFIGS, TASKS
from loggers import ConsoleLogger, MLflowLogger
from models import LGBMConfig
from validation import Fold, evaluate, folds


def config_name(task: str, start: int, end: int, full: bool) -> str:
    return f"{task}_lgbm" if full else f"{task}_lgbm_h{start}-{end}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=TASKS)
    p.add_argument("--hours", type=int, nargs="+", default=[0, 168])
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--splits", type=int, default=12)
    p.add_argument("--save", type=Path, help="folder for the models")
    p.add_argument("--console", action="store_true", help="print instead of MLflow")
    a = p.parse_args()

    task = TASKS[a.task]
    df = task.load()
    name = f"{task.name}_lgbm"
    if (a.start, a.end) != ("2025-01-01", "2026-01-01"):
        name += f"_{a.start}_{a.end}"
    full = a.hours == [0, 168]
    tables = []
    for start, end in zip(a.hours[:-1], a.hours[1:], strict=True):
        with open(
            f"{ROOT_CONFIGS}/{config_name(task.name, start, end, full)}.json"
        ) as f:
            config = LGBMConfig(**json.load(f))
        rows = df.loc[(df.h >= start) & (df.h < end)]
        saved = config_name(task.name, start, end, full)
        target = a.save / f"{saved}.txt" if a.save else None
        tables += [
            forecast(task, config, rows, fold, target)
            for fold in folds(a.start, a.end, a.splits)
        ]
    table = pd.concat(tables).sort_values(["issue_time", "h"])
    table.to_parquet(DERIVED / "forecasts" / f"{name}.parquet", index=False)
    logger = ConsoleLogger(name) if a.console else MLflowLogger("holdout", name)
    with logger:
        logger.log_params(
            {"hours": a.hours, "start": a.start, "end": a.end, "splits": a.splits}
        )
        evaluate(table, logger)


def forecast(
    task, config: LGBMConfig, df: pd.DataFrame, fold: Fold, save: Path | None = None
) -> pd.DataFrame:
    fit_rows, val_rows = task.train_rows(fold.train(df)), fold.val(df)
    booster = lgbm.fit(config, task.X(fit_rows), task.y(fit_rows))
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        booster.booster_.save_model(save)
    return task.scoring_table(val_rows, lgbm.predict(booster, task.X(val_rows)))


if __name__ == "__main__":
    main()
