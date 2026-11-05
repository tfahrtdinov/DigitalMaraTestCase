"""PYTHONPATH=src uv run python -m lgbm.train co2 --hours 0 48 --start 2026-08-28 [--save fitted]

Fits the tuned LightGBM config of a task on the issues before start minus the 7-day gap and
saves <save>/<config name>.txt. Forecasts: lgbm.run_holdout."""

import argparse
import json
from pathlib import Path

import pandas as pd

import lgbm
from config import LGBMConfig
from lgbm.run_hpo import ROOT_CONFIGS, TASKS

GAP = pd.Timedelta(days=7)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=TASKS)
    p.add_argument("--hours", type=int, nargs=2, default=[0, 168])
    p.add_argument("--start", required=True)
    p.add_argument("--save", type=Path, default=Path("fitted"))
    a = p.parse_args()

    task = TASKS[a.task]
    first, last = a.hours
    name = lgbm.config_name(task.name, first, last)
    with open(f"{ROOT_CONFIGS}/{name}.json") as f:
        config = LGBMConfig(**json.load(f))
    df = task.load()
    train_end = pd.Timestamp(a.start, tz="UTC") - GAP
    rows = task.train_rows(
        df.loc[(df.h >= first) & (df.h < last) & (df.issue_time < train_end)]
    )
    model = lgbm.fit(config, task.X(rows), task.y(rows))
    a.save.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(a.save / f"{name}.txt")
    print(f"{len(rows)} rows up to {train_end:%Y-%m-%d} -> {a.save / name}.txt")


if __name__ == "__main__":
    main()
