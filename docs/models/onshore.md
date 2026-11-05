# Onshore wind production, Netherlands: LightGBM predictor

Forecasts hourly production for the 168 hours after each 06 UTC issue from weather forecasts
and the production history available at issue. Metrics are in MWh per hour unless stated.

## Task

| | |
|---|---|
| production series | NED wind onshore, measured (classification 2), hourly kWh |
| target | y = production / cap, where cap is the 99th percentile of hourly production over the 60 days before issue (capacity proxy). |
| back to production | production = y × cap |
| naive reference | cubic power curve of the GEFS wind speed (cut-in 3 m/s, rated 12 m/s) × cap, no model |
| issue time | 06 UTC daily; window hours 0–167 from issue |
| model | LightGBM regressor, L1 objective, one model for all 168 hours |

## Periods

| period | renewables predictors | CO₂ models |
|---|---|---|
| 2020-10 – 2023-12 | training | training |
| 2024-01 – 2024-12 | training | 6 two-month expanding folds, 7-day gap |
| 2025-01 – 2025-12 | 12 monthly expanding folds, scored once | no decisions |

## Features

| feature | description | source | unit |
|---|---|---|---|
| ws_gefs | 100 m wind speed, ensemble mean over the park points | GEFS ensemble (31 members, 00 UTC run), 3-hourly, interpolated to hour centres | m/s |
| ws_std_gefs | ensemble standard deviation of the same speed | GEFS ensemble (31 members, 00 UTC run), 3-hourly, interpolated to hour centres | m/s |
| ws_ecmwf | 100 m wind speed at hour centre, mean over the park points | ECMWF IFS via Open-Meteo previous runs, hourly, lead ceil(h/24) days; from 2024-03 | m/s |
| ws_std_ecmwf | standard deviation of that speed over the park points | ECMWF IFS via Open-Meteo previous runs, hourly, lead ceil(h/24) days; from 2024-03 | m/s |
| h | hour within the 168-hour window (0 = issue, 06 UTC) | calendar | h |
| y_last | production of the hour before issue / cap | production history | fraction |
| y_1d | mean production of the 24 hours before issue / cap | production history | fraction |
| err_1d | mean of (y − power curve of GEFS speed) over the previous issue's hours 0–23 | production history, GEFS | fraction |

Missing values (ECMWF before 2024-03, night hours) stay NaN; LightGBM routes them.

## Hyperparameter search

Optuna TPE, 20 trials, objective: MAE averaged over the six 2024 folds. Each
trial refits the model on every fold and logs all metrics to MLflow (experiment `hpo`,
run `onshore_lgbm`, one child run per trial).

| parameter | range | scale |
|---|---|---|
| n_estimators | 100 – 500, step 100 | linear |
| learning_rate | 0.01 – 0.1 | log |
| num_leaves | 15 – 127 | log |
| min_child_samples | 20 – 300 | log |
| colsample_bytree | 0.5 – 1.0 | linear |
| reg_lambda | 0.001 – 10 | log |

Fixed: objective = l1 (MAE), subsample = 0.8 every iteration, seed = 0.

![HPO convergence](figures/onshore_hpo.png)

### Best configuration (trial 19)

| parameter | value |
|---|---|
| seed | 0 |
| objective | l1 |
| n_estimators | 200 |
| learning_rate | 0.03934 |
| num_leaves | 52 |
| min_child_samples | 40 |
| subsample | 0.8 |
| colsample_bytree | 0.9858 |
| reg_lambda | 0.001166 |

Validation metrics of this trial, mean over the 2024 folds:

| metric | value |
|---|---|
| MAE | 798 |
| RMSE | 1,130 |
| bias (forecast − actual) | -87 |
| improvement over naive, % | 25.1 |
| Spearman within window | 0.740 |
| MAE day 1 | 448 |
| MAE day 2 | 521 |
| MAE day 3 | 635 |
| MAE day 4 | 785 |
| MAE day 5 | 921 |
| MAE day 6 | 1,089 |
| MAE day 7 | 1,190 |

## Test 2025

Twelve monthly expanding folds; each month is forecast by a model fit on everything before
it minus a 7-day gap. All 365 issues of 2025 are scored once. TimesFM and Chronos-2 are
zero-shot on the same origins with 1024 hours of production history and the same weather
forecasts as covariates.

| metric | LightGBM | naive | TimesFM | Chronos-2 |
|---|---|---|---|---|
| MAE | 756 | 1,037 | 739 | 785 |
| RMSE | 1,078 | 1,488 | 1,082 | 1,112 |
| bias (forecast − actual) | -129 | -384 | -195 | -191 |
| improvement over naive, % | 27.1 | 0.0 | 28.8 | 24.3 |
| Spearman within window | 0.759 | 0.612 | 0.769 | 0.760 |
| MAE day 1 | 460 | 833 | 399 | 467 |
| MAE day 2 | 527 | 843 | 482 | 577 |
| MAE day 3 | 598 | 883 | 562 | 655 |
| MAE day 4 | 702 | 985 | 688 | 738 |
| MAE day 5 | 875 | 1,136 | 884 | 903 |
| MAE day 6 | 1,044 | 1,284 | 1,044 | 1,058 |
| MAE day 7 | 1,088 | 1,298 | 1,114 | 1,100 |

![MAE by forecast day](figures/onshore_by_day.png)

![MAE by month](figures/onshore_by_month.png)

![Example week](figures/onshore_week.png)

## Feature importance

Share of total split gain, model fit on all issues before 2025.

![Feature importance](figures/onshore_importance.png)

## Reproduce

```
PYTHONPATH=src uv run python -m data.features.wind onshore
PYTHONPATH=src uv run python -m lgbm.run_hpo onshore
PYTHONPATH=src uv run python -m lgbm.run_folds onshore
```
