#!/bin/zsh
# Uploads src/, the LSTM configs and data/derived/co2_sequences.npz as the private dataset
# co2-lstm-input; with --run also starts the kernel. No .env and no raw data leave the machine.
#   kaggle/push.sh "message" [--run]
# Account: KAGGLE_TOKEN_FILE (default ~/.kaggle/access_token); the owner must match kernel-metadata.json.
set -e
cd "$(dirname "$0")/.."
export KAGGLE_API_TOKEN=$(cat "${KAGGLE_TOKEN_FILE:-$HOME/.kaggle/access_token}")
K=${KAGGLE_BIN:-uvx --with six kaggle}
OWNER=${KAGGLE_OWNER:-timurfahrtdinov}
S=data/kaggle_build/co2-lstm-input; rm -rf "$S"; mkdir -p "$S/configs"
rsync -a --exclude __pycache__ src "$S/"
cp configs/co2_lstm*.json "$S/configs/"
cp data/derived/co2_sequences.npz "$S/"
print -r -- "{\"title\": \"co2-lstm-input\", \"id\": \"$OWNER/co2-lstm-input\", \"licenses\": [{\"name\": \"other\"}]}" > "$S/dataset-metadata.json"
if ${=K} datasets status "$OWNER/co2-lstm-input" >/dev/null 2>&1; then
  ${=K} datasets version -p "$S" -m "${1:-update}" --dir-mode zip
else
  ${=K} datasets create -p "$S" --dir-mode zip
fi
[[ "$2" == "--run" ]] && { sleep 60; ${=K} kernels push -p "${KDIR:-kaggle}"; }
