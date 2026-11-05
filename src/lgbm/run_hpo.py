"""PYTHONPATH=src uv run python -m lgbm.run_hpo solar --trials 30 [--console]

Metrics go to MLflow (experiment "hpo") unless --console is given."""

import argparse
import json

import optuna

from lgbm import hpo
from loggers import ConsoleLogger, MLflowLogger
from tasks import solar
from validation import Periods

TASKS = {"solar": solar.SOLAR}
ROOT_CONFIGS = "configs"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=TASKS)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--splits", type=int, default=6)
    p.add_argument("--val-start", default="2024-01-01")
    p.add_argument("--holdout-start", default="2025-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--console", action="store_true", help="print instead of MLflow")
    a = p.parse_args()

    task = TASKS[a.task]
    folds = Periods(a.val_start, a.holdout_start, a.end).validation_folds(a.splits)
    name = f"{task.name}_lgbm"
    logger = ConsoleLogger(name) if a.console else MLflowLogger("hpo", name)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with logger:
        study = hpo.tune(task, task.load(), folds, a.trials, logger)
    best = task.space(optuna.trial.FixedTrial(study.best_params)).to_dict()
    with open(f"{ROOT_CONFIGS}/{name}.json", "w") as f:
        json.dump(best, f, indent=2)
    print(
        f"best trial {study.best_trial.number}: {study.best_trial.user_attrs['metrics']}"
    )
    print(f"config -> {ROOT_CONFIGS}/{name}.json")


if __name__ == "__main__":
    main()
