#!/bin/zsh
# Copies the finished kernel's forecasts to data/derived/forecasts and its log to data/logs/kaggle.
set -e
cd "$(dirname "$0")/.."
export KAGGLE_API_TOKEN=$(cat "${KAGGLE_TOKEN_FILE:-$HOME/.kaggle/access_token}")
K=${KAGGLE_BIN:-uvx --with six kaggle}
OWNER=${KAGGLE_OWNER:-timurfahrtdinov}
O=data/logs/kaggle; mkdir -p "$O" data/derived/forecasts
${=K} kernels output "$OWNER/co2-lstm" -p "$O"
mv "$O"/co2_lstm*.parquet data/derived/forecasts/
