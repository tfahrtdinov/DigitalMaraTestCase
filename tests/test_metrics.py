import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import metrics  # noqa: E402


def test_basic_metrics():
    y, y_hat = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 1.0])
    assert metrics.mae(y, y_hat) == 1.0
    assert metrics.rmse(y, y_hat) == np.sqrt(5 / 3)
    assert metrics.bias(y, y_hat) == -1 / 3
    assert metrics.improvement(y, y_hat, np.array([3.0, 4.0, 5.0])) == 50.0


def test_summary_per_day_and_spearman():
    h = np.arange(168)
    df = pd.DataFrame({"issue_time": "2025-01-01", "h": h, "y": h, "y_hat": h + 1.0})
    df.loc[df.h >= 24, "y_hat"] = df.y + 3.0
    s = metrics.summary(df).to_dict()
    assert (s["mae_d1"], s["mae_d2"], s["mae_d7"]) == (1.0, 3.0, 3.0)
    assert s["spearman"] == 1.0 and s["bias"] > 0
    assert "improvement_%" not in s
