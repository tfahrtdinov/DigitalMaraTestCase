"""Optuna search over LightGBM configs. Every trial is a child run of the logger: its
config and per-fold metrics go there, the fold means also go to the study run with the
trial number as step. The study is optimised on the mean MAE."""

from typing import cast

import optuna
import pandas as pd

import lgbm
from loggers import Logger
from models import LGBMConfig
from tasks import Task
from validation import Fold, evaluate


def fold_scores(
    task: Task, config: LGBMConfig, df: pd.DataFrame, fold: Fold, logger: Logger
) -> pd.Series:
    """Fits on the fold's training rows, scores its validation rows (metrics.summary)."""
    fit_rows, val_rows = task.train_rows(fold.train(df)), fold.val(df)
    booster = lgbm.fit(config, task.X(fit_rows), task.y(fit_rows))

    table = task.scoring_table(val_rows, lgbm.predict(booster, task.X(val_rows)))
    return evaluate(table, logger, prefix=f"{fold.val_start:%Y-%m-%d}/")


def tune(
    task: Task, df: pd.DataFrame, folds: list[Fold], n_trials: int, logger: Logger
) -> optuna.Study:
    """Runs `n_trials` of task.space; the best trial's metrics are in its user_attrs."""

    def objective(trial: optuna.Trial) -> float:
        config = task.space(trial)
        assert isinstance(config, LGBMConfig)
        with logger.child(f"trial_{trial.number}") as trial_logger:
            trial_logger.log_params(config.to_dict())
            per_fold = [fold_scores(task, config, df, f, trial_logger) for f in folds]
            by_fold = pd.concat(per_fold, axis=1)  # metrics x folds
            fold_metrics_mean = cast(dict[str, float], by_fold.mean(axis=1).to_dict())
            trial_logger.log_metrics(fold_metrics_mean)
        logger.log_metrics(fold_metrics_mean, step=trial.number)

        trial.set_user_attr("metrics", fold_metrics_mean)
        return fold_metrics_mean["mae"]

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(objective, n_trials=n_trials)
    return study
