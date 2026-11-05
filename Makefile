# `make` lists the targets. DOCKER=1 runs the steps in the containers from docker-compose.yml.

TASKS  := solar onshore offshore
TRIALS := 30
DEVICE := cpu

# CO2 models: search over the 2025 folds, forecasts for the second half of 2025
CO2_SEARCH := --trials 20 --splits 12 --val-start 2025-01-01 --holdout-start 2026-01-01
CO2_WINDOW := --start 2025-07-01 --end 2026-01-01 --splits 1

# Production forecasts fed to the CO2 table; the period must match RENEWABLES_PERIOD in
# src/data/features/co2.py. 30 folds refit the models about every two months.
INPUTS_START  := 2021-10-01
INPUTS_END    := 2026-11-03
INPUTS_SPLITS := 30

LOCAL := PYTHONPATH=src uv run python -m
ifdef DOCKER
RUN := docker compose run --rm train
RUN_NEURAL := docker compose run --rm train-neural
else
RUN := $(LOCAL)
RUN_NEURAL := $(LOCAL)
endif

.DEFAULT_GOAL := help
.PHONY: help image mlflow \
	tables hpo forecasts inputs renewables \
	co2-tables co2-hpo co2-forecast co2 co2-lstm

help: ## List the targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

image: ## Build the LightGBM and the LSTM (torch) images; repeat after changing src/
	docker compose build train train-neural

mlflow: ## Start MLflow at http://localhost:5001
	docker compose up -d mlflow

tables: ## Build the production tables
	$(RUN) data.features.solar
	$(RUN) data.features.wind onshore
	$(RUN) data.features.wind offshore

hpo: ## Tune the production models on 2024
	for t in $(TASKS); do $(RUN) lgbm.run_hpo $$t --trials $(TRIALS) || exit 1; done

forecasts: ## Forecast production for 2025
	for t in $(TASKS); do $(RUN) lgbm.run_holdout $$t || exit 1; done

inputs: ## Forecast production for the CO2 table
	for t in $(TASKS); do $(RUN) lgbm.run_holdout $$t --start $(INPUTS_START) --end $(INPUTS_END) --splits $(INPUTS_SPLITS) || exit 1; done

renewables: tables hpo forecasts inputs ## All production steps

co2-tables: ## Build the CO2 table and the LSTM arrays
	$(RUN) data.features.co2
	$(RUN) data.features.co2_sequences

co2-hpo: ## Tune the CO2 LightGBM models on 2025
	$(RUN) lgbm.run_hpo co2 --hours 0 48 $(CO2_SEARCH)
	$(RUN) lgbm.run_hpo co2_far --hours 48 168 $(CO2_SEARCH)

co2-forecast: ## Forecast CO2 with LightGBM
	$(RUN) lgbm.run_holdout co2 --hours 0 48 $(CO2_WINDOW)
	$(RUN) lgbm.run_holdout co2_far --hours 48 168 $(CO2_WINDOW)

co2: co2-tables co2-hpo co2-forecast ## All CO2 LightGBM steps

co2-lstm: ## Train the LSTM and forecast CO2 on DEVICE (default cpu)
	$(RUN_NEURAL) neural.run configs/co2_lstm_h0-48.json --hours 0 48 --seeds 5 --data data/derived/co2_sequences.npz $(CO2_WINDOW) --device $(DEVICE)
	$(RUN_NEURAL) neural.run configs/co2_lstm_h48-168.json --hours 48 168 --seeds 3 --data data/derived/co2_far_sequences.npz $(CO2_WINDOW) --device $(DEVICE)
