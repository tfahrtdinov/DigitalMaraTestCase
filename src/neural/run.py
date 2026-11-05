"""PYTHONPATH=src python -m neural.run configs/co2_lstm_h0-48.json --hours 0 48
    --start 2025-01-01 --end 2026-01-01 --splits 12 [--seeds 5] [--device cuda] [--console]
    [--save models]

Trains the neural model of the config (one row per issue and hour for an MLP, whole
windows for an LSTM) on the expanding folds of [start, end), once per
seed, and averages the seeds. Metrics per fold and overall go to MLflow (experiment
"neural") unless --console is given; forecasts go to <out>/<config name>.parquet, with the suffix _<start>_<end> for any period
other than the validation half-year. --save writes one checkpoint per seed of an LSTM, for
neural.predict; with several folds the last one stays."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.features.sequences import Sequences
from loggers import ConsoleLogger, Logger, MLflowLogger
from models import TorchConfig
from models.nets import NETS
from models.neural import MODELS
from trainer import Trainer
from validation import Fold, evaluate, folds


def forecast(
    config: TorchConfig,
    data: Sequences,
    fold: Fold,
    hours: tuple[int, int],
    seeds: int,
    device: str,
    logger: Logger,
    save: Path | None = None,
) -> pd.DataFrame:
    """Table of forecasts for the fold's issues: issue_time, h, y, y_hat, y_naive."""
    train = np.asarray(data.issue_time < fold.train_end)
    val = np.asarray(
        (data.issue_time >= fold.val_start) & (data.issue_time < fold.val_end)
    )
    tabular = config.net == "mlp"
    loaders = data.table_loaders if tabular else data.loaders
    train_loader, val_loader, y_scale = loaders(train, val, hours, config.batch_size)
    sizes = (
        (data.table.shape[-1],)
        if tabular
        else (data.past.shape[-1], data.future.shape[-1])
    )
    h = np.arange(*hours)
    predictions = []
    for seed in range(config.seed, config.seed + seeds):
        torch.manual_seed(seed)
        net = NETS[config.net](*sizes, **config.net_params)
        model = MODELS[config.model](net, **config.model_params)
        with logger.child(f"{fold.val_start:%Y-%m-%d}_seed{seed}") as run_logger:
            Trainer(
                config.epochs,
                config.threads,
                seed,
                run_logger,
                device,
                config.grad_clip,
                config.one_cycle,
            ).fit(model, train_loader, val_loader)
        if save is not None:
            save.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "net": config.net,
                    "net_params": config.net_params,
                    "sizes": sizes,
                    "hours": hours,
                    "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                    "y_scale": y_scale,
                    "train_end": str(fold.train_end),
                    **data.scaling(train),
                },
                f"{save}_seed{seed}.pt",
            )
        predicted = Trainer.predict(model, val_loader)
        predictions.append(
            predicted.reshape(-1, len(h)) if tabular else predicted[:, h]
        )
    base = data.base[val][:, h]
    y_hat = np.mean(predictions, axis=0) * y_scale + base

    table = pd.DataFrame(
        {
            "issue_time": np.repeat(data.issue_time[val], len(h)),
            "h": np.tile(h, val.sum()),
            "y": (data.y[val][:, h] + base).ravel(),
            "y_hat": np.clip(y_hat, 0, None).ravel(),
            "y_naive": data.y_ref[val][:, h].ravel(),
        }
    )
    return table.dropna(subset=["y"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--hours", type=int, nargs=2, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--splits", type=int, required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data", default="data/derived/co2_sequences.npz")
    p.add_argument("--out", default="data/derived/forecasts")
    p.add_argument("--save", type=Path, help="folder for the checkpoints")
    p.add_argument("--console", action="store_true", help="print instead of MLflow")
    a = p.parse_args()

    with open(a.config) as f:
        config = TorchConfig(**json.load(f))
    name = Path(a.config).stem
    if (a.start, a.end) != ("2025-07-01", "2026-01-01"):
        name += f"_{a.start}_{a.end}"
    data = Sequences.load(a.data)
    hours = (a.hours[0], a.hours[1])
    logger = ConsoleLogger(name) if a.console else MLflowLogger("neural", name)
    tables = []
    with logger:
        logger.log_params({**config.to_dict(), "hours": a.hours, "seeds": a.seeds})
        for fold in folds(a.start, a.end, a.splits):
            target = a.save / Path(a.config).stem if a.save else None
            table = forecast(
                config, data, fold, hours, a.seeds, a.device, logger, target
            )
            evaluate(table, logger, prefix=f"{fold.val_start:%Y-%m-%d}/")
            tables.append(table)
        result = pd.concat(tables)
        evaluate(result, logger)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    result.to_parquet(Path(a.out) / f"{name}.parquet", index=False)


if __name__ == "__main__":
    main()
