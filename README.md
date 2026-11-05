# CO₂ factor forecast for electricity in the Netherlands

Forecasts the CO₂ factor for the next 168 hours at 1-hour resolution. LightGBM models first forecast renewable production; the CO₂ models then use those forecasts as inputs.

![Data flow from sources through production forecasts to the CO₂ forecast](docs/figures/pipeline.svg)

A forecast is issued daily at 06:00 UTC for hours `h = 0…167`. Inputs must be known at issue, and tables are indexed by `(issue_time, h)`.

## Run

The data is not in the repository: put the source files in `data/raw/`. `make` lists all targets.

```bash
make setup-neural        # folders, dependencies with torch (make setup: without torch)
make mlflow              # results at http://localhost:5001, with or without Docker
make renewables          # production tables, tuning, forecasts
make co2                 # CO₂ table, LSTM arrays, LightGBM for days 1–2 and days 3–7
make co2-lstm            # LSTM, DEVICE=cuda for a GPU
```

### Docker

I trained the LSTM and tuned the days 3–7 LightGBM on Kaggle, with its GPU and package versions (torch 2.11, LightGBM 4.6, pandas 2.3). The image uses the versions in `uv.lock` (torch 2.15, LightGBM 4.7, pandas 3.1), so training in Docker may not match my results exactly. To train in Docker:

```bash
make dirs mlflow
make image                    # LightGBM image, and the LSTM image with torch
make renewables co2 DOCKER=1
make co2-lstm DOCKER=1
```

Rebuild the images after changing `src/`. Exit code 137 means the container ran out of memory; raise the limit in Docker Desktop.

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

## Protocols and contracts

| Name | Defined in | Role |
|---|---|---|
| `NeuralModel` protocol | [protocol.py](src/models/neural/protocol.py) | everything `Trainer` needs from a model |
| `Logger` protocol | [loggers.py](src/loggers.py) | console or MLflow output |
| `Task` | [task.py](src/tasks/task.py) | features, target and loading of one LightGBM task |
| Forecast table | [metrics.py](src/metrics.py) | `issue_time, h, y, y_hat, y_naive`, the output of both branches |

## Data

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

- `outdated/` (download scripts) and `experiments/` (TimesFM-3 reference) are in `.gitignore`, so a clean clone cannot rebuild the data, and `make co2` fails without `data/derived/forecasts/timesfm3.csv`.
- `lgbm.run_holdout` names its output by task and period regardless of `--hours`, so another segment overwrites the file.
- Production forecasts for 2021–2023 use parameters tuned on 2024; they enter only the CO₂ training data.
- [data/README.md](data/README.md) is in Russian and describes the previous setup.
