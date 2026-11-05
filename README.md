# CO₂ factor forecast for electricity in the Netherlands

Forecasts the CO₂ factor for the next 168 hours at 1-hour resolution. LightGBM models first forecast renewable production; the CO₂ models then use those forecasts as inputs.

![Data flow from sources through production forecasts to the CO₂ forecast](docs/figures/pipeline.svg)

A forecast is issued daily at 06:00 UTC for hours `h = 0…167`. Inputs must be known at issue, and tables are indexed by `(issue_time, h)`.

## Run

The feature tables and model configs are committed, so the models run without raw data. `make` lists all targets.

```bash
uv sync --extra neural   # dependencies with torch (uv sync: without torch)
make mlflow              # MLflow in Docker at http://localhost:5001; without it, runs go to mlruns/
make forecasts           # production models, 2025
make co2-forecast        # CO₂ LightGBM for days 1–2 and days 3–7
make co2-lstm            # LSTM, DEVICE=cuda for a GPU
```

`make hpo co2-hpo` repeats the parameter search and overwrites the configs. `make renewables co2` rebuilds the tables as well and needs the raw data (see [Data](#data)).

### Docker

I trained the LSTM and tuned the days 3–7 LightGBM on Kaggle, with its GPU and package versions (torch 2.11, LightGBM 4.6, pandas 2.3). The image uses the versions in `uv.lock` (torch 2.15, LightGBM 4.7, pandas 3.1), so training in Docker may not match my results exactly. To train in Docker:

```bash
make mlflow
make image                    # LightGBM image, and the LSTM image with torch
make forecasts co2-forecast DOCKER=1
make co2-lstm DOCKER=1
```

Rebuild the images after changing `src/`. Exit code 137 means the container ran out of memory; raise the limit in Docker Desktop.

## Trained models

`models/` holds the models of the last test fold: trained on issues up to 2026-08-20, they forecast 2026-08-28 – 2026-10-26 with the numbers in the report.

| Files | Forecast |
|---|---|
| `co2_lstm_h0-48_seed{0,1,2}.pt` | CO₂, days 1–2, LSTM, mean over the seeds |
| `co2_lstm_h48-168_seed{0,1,2}.pt` | CO₂, days 3–7, LSTM, mean over the seeds |
| `co2_lgbm_h0-48.txt` | CO₂, days 1–2, LightGBM |
| `co2_far_lgbm_h48-168.txt` | CO₂, days 3–7, LightGBM |
| `solar_lgbm.txt`, `onshore_lgbm.txt`, `offshore_lgbm.txt` | production inputs of the CO₂ models |

Run with `PYTHONPATH=src OMP_NUM_THREADS=1`: on macOS, torch and LightGBM in one process otherwise crash with a segmentation fault.

```python
# the last issue in the tables, days 1–2
from pathlib import Path
import lightgbm as lgb
from data.features.sequences import Sequences
from neural.predict import load, predict
from tasks.co2 import CO2

data = Sequences.load("data/derived/co2_sequences.npz")
lstm = load(sorted(Path("models").glob("co2_lstm_h0-48_seed*.pt")))
co2_lstm = predict(lstm, data.past[-1:], data.future[-1:], data.base[-1:, :48])

rows = CO2.load().query("h < 48").tail(48)
booster = lgb.Booster(model_file="models/co2_lgbm_h0-48.txt")
co2_lgbm = rows["co2_last"] + booster.predict(CO2.X(rows))
```

Days 3–7 work the same way with `co2_far_sequences.npz`, `base[:, 48:168]` and `CO2_FAR`, whose base is `co2_prof`. Both results are in g CO₂/kWh.

## Models

![One model run: split by time, train, forecast, score, with what is written through the Logger protocol to the console or MLflow](docs/figures/models.svg)

| | LightGBM | LSTM |
|---|---|---|
| Targets | solar, onshore, offshore, CO₂ | CO₂ |
| Input | parquet, one row per (issue, hour) | arrays, `co2_sequences.npz` and `co2_far_sequences.npz` |
| Missing values | kept as NaN | zero plus a flag channel |
| Scaling | none | per channel, on each fold's training issues |
| Parameters | tuned with Optuna | set in JSON configs |
| Runs on | local or Docker | local or Docker |

Each branch has two CO₂ models. Days 1–2 predict the change from the last hour before issue; days 3–7 predict the deviation from the 28-day hourly profile.

### Fit LightGBM from a config

Each LightGBM file in `configs/` holds the parameters of one model. The tables are committed, so no raw data is needed.

| Config | Task | Hours |
|---|---|---|
| `solar_lgbm.json` | `tasks.solar.SOLAR` | 0–167 |
| `onshore_lgbm.json` | `tasks.wind.ONSHORE` | 0–167 |
| `offshore_lgbm.json` | `tasks.wind.OFFSHORE` | 0–167 |
| `co2_lgbm_h0-48.json` | `tasks.co2.CO2` | 0–47 |
| `co2_far_lgbm_h48-168.json` | `tasks.co2.CO2_FAR` | 48–167 |

This script fits the days 1–2 CO₂ model and scores it on the second half of 2025, with MAE 24.92 g/kWh. Run it from the repository root with `PYTHONPATH=src uv run python fit.py`.

```python
import json

import lgbm
from metrics import summary
from models import LGBMConfig
from tasks.co2 import CO2

with open("configs/co2_lgbm_h0-48.json") as f:
    config = LGBMConfig(**json.load(f))

df = CO2.load()
df = df[df.h < 48]
train = CO2.train_rows(df[df.issue_time < "2025-06-24"])  # 7-day gap before the test
test = df[(df.issue_time >= "2025-07-01") & (df.issue_time < "2026-01-01")]

model = lgbm.fit(config, CO2.X(train), CO2.y(train))
forecast = CO2.scoring_table(test, lgbm.predict(model, CO2.X(test)))  # back to g/kWh
print(summary(forecast))
```

`lgbm.fit` passes the config to `lightgbm.LGBMRegressor`, with `seed` as `random_state` and `subsample_freq=1`. The same fit as a command, with the forecasts saved to `data/derived/forecasts/` and the model to `models/`:

```bash
PYTHONPATH=src uv run python -m lgbm.run_holdout co2 --hours 0 48 \
    --start 2025-07-01 --end 2026-01-01 --splits 1 --save models --console
```

## Protocols and contracts

| Name | Defined in | Role |
|---|---|---|
| `NeuralModel` protocol | [protocol.py](src/models/neural/protocol.py) | everything `Trainer` needs from a model |
| `Logger` protocol | [loggers.py](src/loggers.py) | console or MLflow output |
| `Task` | [task.py](src/tasks/task.py) | features, target and loading of one LightGBM task |
| Forecast table | [metrics.py](src/metrics.py) | `issue_time, h, y, y_hat, y_naive`, the output of both branches |

## Data

The repository holds only the feature tables, 240 MB in `data/derived/`. Each row or array entry is one forecast issue, with inputs known at that issue.

| Committed file | Contents | Used by |
|---|---|---|
| `solar_dataset.parquet`, `onshore_dataset.parquet`, `offshore_dataset.parquet` | weather forecasts, production history and target, per (issue, hour) | production LightGBM |
| `co2_dataset.parquet` | production forecasts, measured history, day-ahead forecasts, weather, outages, targets and the TimesFM-3 reference, per (issue, hour) | CO₂ LightGBM |
| `co2_sequences.npz`, `co2_far_sequences.npz` | the same inputs as arrays over the past 336 and next 168 hours, for days 1–2 and days 3–7 | LSTM |

The raw data is 10 GB, so I do not commit it to git; I can send it on request. To rebuild the tables, put it in `data/raw/`.

[loader.py](src/data/loader.py) reads `data/raw/`, `src/data/features/` builds the tables, and training reads the results from `data/derived/`. Tables are rebuilt by running their module; only the GEFS cache refreshes itself. Column details are in [data/README.md](data/README.md).

| Source | Provides | Access | Files in `data/raw/` |
|---|---|---|---|
| NED | CO₂ factor, production by fuel, consumption | REST, key, EU VPN | `<series>_cls2_*.csv` |
| ENTSO-E | offshore wind, nuclear, prices, day-ahead forecasts, outages | REST, key | `entsoe_*.csv` |
| NOAA GEFS | ensemble wind and radiation from 2020-10 | dynamical.org, Zarr | `gefs/`, `gefs_neighbors/` |
| ECMWF | wind, radiation, temperature from 2024-04 | Open-Meteo REST | `openmeteo/` |

## Validation

Folds use an expanding window with a 7-day gap before each block.

| Period | Production models | CO₂ models |
|---|---|---|
| until 2023-12 | training | training |
| 2024 | tuning, 6 folds | training |
| 2025 | test, 12 folds | LightGBM tuning on 12 folds, model comparison on 2025-07 – 2025-12 |
| 2026 | not used | final test, once |

## Known gaps

- The TimesFM-3 reference (`data/derived/references/`) is not in the repository, and the scripts that computed it were removed. `make co2-tables` reads it, so the CO₂ table cannot be rebuilt without that folder. The table in the repository already holds it as `y_ref`.
- `lgbm.run_holdout` names its output by task and period regardless of `--hours`, so another segment overwrites the file.
- Production forecasts for 2021–2023 use parameters tuned on 2024; they enter only the CO₂ training data.
- [data/README.md](data/README.md) is in Russian and describes the previous setup.
