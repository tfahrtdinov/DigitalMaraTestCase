"""PYTHONPATH=src python -m neural.train configs/co2_lstm_h0-48.json --hours 0 48
    --start 2026-08-28 --end 2026-10-27 [--seeds 3] [--device cuda]
    [--data data/derived/co2_sequences.npz] [--save fitted] [--console]

Fits one network per seed on the issues before start minus the 7-day gap, with [start, end) as
the monitored validation, and saves <save>/<config name>_seed<n>.pt. Forecasts: neural.run_holdout.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import neural
from config import TorchConfig
from data.features.sequences import Sequences
from loggers import ConsoleLogger, MLflowLogger
from validation import folds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--hours", type=int, nargs=2, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data", default="data/derived/co2_sequences.npz")
    p.add_argument("--save", type=Path, default=Path("fitted"))
    p.add_argument("--console", action="store_true", help="print instead of MLflow")
    a = p.parse_args()

    with open(a.config) as f:
        config = TorchConfig(**json.load(f))
    data = Sequences.load(a.data)
    hours = (a.hours[0], a.hours[1])
    name = Path(a.config).stem
    fold = folds(a.start, a.end, 1)[0]
    train = np.asarray(data.issue_time < fold.train_end)
    val = np.asarray(
        (data.issue_time >= fold.val_start) & (data.issue_time < fold.val_end)
    )
    if config.net == "mlp":
        sizes = (data.table.shape[-1],)
        scaling = data.table_scaling(train, hours)
        rows, val_rows, y_scale = data.table_loaders(
            train, val, hours, config.batch_size
        )
    else:
        sizes = (data.past.shape[-1], data.future.shape[-1])
        scaling = data.scaling(train)
        rows, val_rows, y_scale = data.loaders(train, val, hours, config.batch_size)

    logger = ConsoleLogger(name) if a.console else MLflowLogger("neural", name)
    with logger:
        logger.log_params({**config.to_dict(), "hours": a.hours, "seeds": a.seeds})
        for seed in range(config.seed, config.seed + a.seeds):
            with logger.child(f"seed{seed}") as run_logger:
                model = neural.fit(
                    config, sizes, rows, val_rows, seed, a.device, run_logger
                )
            neural.save(
                a.save / f"{name}_seed{seed}.pt",
                model,
                config,
                sizes,
                hours,
                y_scale,
                str(fold.train_end),
                scaling,
            )


if __name__ == "__main__":
    main()
