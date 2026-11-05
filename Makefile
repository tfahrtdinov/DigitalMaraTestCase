# `make` lists the targets. DOCKER=1 runs the steps in the containers from docker-compose.yml.

TASKS  := solar onshore offshore
TRIALS := 30
DEVICE := cpu

# CO2 models: search over the 2025 folds, forecasts for the second half of 2025
CO2_SEARCH := --trials 20 --splits 12 --val-start 2025-01-01 --holdout-start 2026-01-01
CO2_WINDOW := --start 2025-07-01 --end 2026-01-01 --splits 1
NEURAL_WINDOW := --start 2025-07-01 --end 2026-01-01
# outside fitted/, which holds the models of the last test fold
NEURAL_FIT := data/derived/lstm_validation

# Test: five folds of 2026, refit before each; the last of them gives the models in fitted/
TEST_WINDOW := --start 2026-01-01 --end 2026-10-27
TEST_SPLITS := 5
NEURAL_TEST := data/derived/lstm_test
LAST_FOLD   := --start 2026-08-28 --end 2026-10-27

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
	co2-tables co2-hpo co2-forecast co2 co2-lstm \
	train train-test co2-lgbm-test co2-lstm-test train-final

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

co2-lstm: ## Train the LSTM on DEVICE (default cpu) and forecast the second half of 2025
	$(RUN_NEURAL) neural.train configs/co2_lstm_h0-48.json --hours 0 48 --seeds 5 --data data/derived/co2_sequences.npz $(NEURAL_WINDOW) --save $(NEURAL_FIT) --device $(DEVICE)
	$(RUN_NEURAL) neural.train configs/co2_lstm_h48-168.json --hours 48 168 --seeds 3 --data data/derived/co2_far_sequences.npz $(NEURAL_WINDOW) --save $(NEURAL_FIT) --device $(DEVICE)
	$(RUN_NEURAL) neural.run_holdout $(NEURAL_FIT)/co2_lstm_h0-48 --data data/derived/co2_sequences.npz $(NEURAL_WINDOW)
	$(RUN_NEURAL) neural.run_holdout $(NEURAL_FIT)/co2_lstm_h48-168 --data data/derived/co2_far_sequences.npz $(NEURAL_WINDOW)

train: forecasts co2-forecast co2-lstm ## Train all models on the validation periods

co2-lgbm-test: ## Refit the CO2 LightGBM on the 2026 test folds and forecast them
	$(RUN) lgbm.run_holdout co2 --hours 0 48 $(TEST_WINDOW) --splits $(TEST_SPLITS)
	$(RUN) lgbm.run_holdout co2_far --hours 48 168 $(TEST_WINDOW) --splits $(TEST_SPLITS)

co2-lstm-test: ## Refit the LSTM on the 2026 test folds on DEVICE and forecast them
	$(RUN_NEURAL) neural.train configs/co2_lstm_h0-48.json --hours 0 48 --seeds 3 --data data/derived/co2_sequences.npz $(TEST_WINDOW) --splits $(TEST_SPLITS) --save $(NEURAL_TEST) --device $(DEVICE)
	$(RUN_NEURAL) neural.train configs/co2_lstm_h48-168.json --hours 48 168 --seeds 3 --data data/derived/co2_far_sequences.npz $(TEST_WINDOW) --splits $(TEST_SPLITS) --save $(NEURAL_TEST) --device $(DEVICE)
	$(RUN_NEURAL) neural.run_holdout $(NEURAL_TEST)/co2_lstm_h0-48 --data data/derived/co2_sequences.npz $(TEST_WINDOW)
	$(RUN_NEURAL) neural.run_holdout $(NEURAL_TEST)/co2_lstm_h48-168 --data data/derived/co2_far_sequences.npz $(TEST_WINDOW)

train-test: co2-lgbm-test co2-lstm-test ## Refit the CO2 models on the 2026 test folds

train-final: ## Train the models of the last test fold; overwrites fitted/
	for t in $(TASKS); do $(RUN) lgbm.run_holdout $$t $(LAST_FOLD) --splits 1 --save fitted || exit 1; done
	$(RUN) lgbm.run_holdout co2 --hours 0 48 $(LAST_FOLD) --splits 1 --save fitted
	$(RUN) lgbm.run_holdout co2_far --hours 48 168 $(LAST_FOLD) --splits 1 --save fitted
	$(RUN_NEURAL) neural.train configs/co2_lstm_h0-48.json --hours 0 48 --seeds 3 --data data/derived/co2_sequences.npz $(LAST_FOLD) --save fitted --device $(DEVICE)
	$(RUN_NEURAL) neural.train configs/co2_lstm_h48-168.json --hours 48 168 --seeds 3 --data data/derived/co2_far_sequences.npz $(LAST_FOLD) --save fitted --device $(DEVICE)
