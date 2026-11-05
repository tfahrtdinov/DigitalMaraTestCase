"""Kaggle script kernel: runs src/lstm/run.py from the private dataset co2-lstm-input on the
session's GPU. Forecasts and logs land in /kaggle/working. Edit RUNS to change what runs.
"""

import glob
import importlib.util
import os
import subprocess
import sys

# config, first and last hour, first and last issue, number of folds
RUNS = [("co2_lstm_h0-48", 0, 48, "2025-01-01", "2026-01-01", 12)]
SEEDS = 5

subprocess.run(["nvidia-smi"], check=True)
if importlib.util.find_spec("dotenv") is None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "python-dotenv"], check=True
    )
code = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            glob.glob("/kaggle/input/**/src/lstm/run.py", recursive=True)[0]
        )
    )
)
subprocess.run(
    [sys.executable, "-m", "pip", "freeze"],
    stdout=open("/kaggle/working/pip_freeze.txt", "w"),
    check=True,
)
env = {**os.environ, "PYTHONPATH": f"{code}/src"}
for config, first, last, start, end, splits in RUNS:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "lstm.run",
            f"{code}/configs/{config}.json",
            "--hours",
            str(first),
            str(last),
            "--start",
            start,
            "--end",
            end,
            "--splits",
            str(splits),
            "--seeds",
            str(SEEDS),
            "--device",
            "cuda",
            "--data",
            f"{code}/co2_sequences.npz",
            "--out",
            "/kaggle/working",
            "--console",
        ],
        env=env,
        check=True,
    )
