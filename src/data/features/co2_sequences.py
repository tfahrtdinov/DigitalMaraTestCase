"""Arrays of the RNN models from the CO2 table, one file that needs no raw data:
data/derived/co2_sequences.npz. Past: 336 hours of every measured series before issue and
the calendar. Future: the inputs of the CO2 table that change within a forecast window and
the calendar. Channels with gaps get a flag channel; the flag carries the missing values.

Run: PYTHONPATH=src uv run python -m data.features.co2_sequences
"""

from typing import cast

import numpy as np
import pandas as pd

from data.features.co2 import measured
from data.loader import DERIVED
from tasks.co2 import DAY_AHEAD, FORECAST, OUTAGES, WEATHER_COLUMNS

PAST_HOURS = 336
N_HOURS = 168
FUTURE = [
    *FORECAST,
    "co2_prof",
    "ren_prof",
    "gas_prof",
    "coal_prof",
    "total_prof",
    "load_prof",
    "co2_same_hour",
    *DAY_AHEAD,
    *WEATHER_COLUMNS,
    *OUTAGES,
]


def calendar(times: pd.DatetimeIndex) -> pd.DataFrame:
    local = times.tz_convert("Europe/Amsterdam")
    hour = 2 * np.pi * times.hour / 24
    month = 2 * np.pi * (local.month - 1) / 12
    return pd.DataFrame(
        {
            "hour_sin": np.sin(hour),
            "hour_cos": np.cos(hour),
            "month_sin": np.sin(month),
            "month_cos": np.cos(month),
            "weekend": (local.day_of_week >= 5).astype(float),
        },
        index=times,
    )


def with_gap_flags(values: pd.DataFrame) -> pd.DataFrame:
    flags = values.loc[:, values.isna().any()].isna().astype(float).add_prefix("gap_")
    return pd.concat([values, flags], axis=1)


def build(table: pd.DataFrame, hourly: pd.DataFrame) -> dict:
    table = table.sort_values(["issue_time", "h"]).reset_index(drop=True)
    issues = pd.DatetimeIndex(table.issue_time.unique())
    assert len(table) == len(issues) * N_HOURS

    hours = pd.date_range(
        hourly.index.min(), issues.max(), freq="h", tz=None, name="time"
    )
    past = with_gap_flags(pd.concat([hourly.reindex(hours), calendar(hours)], axis=1))
    position = hours.get_indexer(issues)
    keep = position >= PAST_HOURS
    windows = position[keep][:, None] + np.arange(-PAST_HOURS, 0)

    future = pd.concat(
        [
            table[FUTURE],
            calendar(pd.DatetimeIndex(table.valid_time)).reset_index(drop=True),
            (table.h / N_HOURS).rename("h"),
        ],
        axis=1,
    )
    future = with_gap_flags(cast(pd.DataFrame, future))
    n = len(issues)
    shape = (n, N_HOURS)
    return {
        "issue_time": issues[keep].tz_localize(None).to_numpy("datetime64[ns]"),
        "past": past.to_numpy(np.float32)[windows],
        "future": future.to_numpy(np.float32).reshape(*shape, -1)[keep],
        "y": table.y.to_numpy(np.float32).reshape(shape)[keep],
        "co2_last": table.co2_last.to_numpy(np.float32).reshape(shape)[keep, 0],
        "y_ref": table.y_ref.to_numpy(np.float32).reshape(shape)[keep],
        "past_columns": np.array(past.columns),
        "future_columns": np.array(future.columns),
    }


def main() -> None:
    table = pd.read_parquet(DERIVED / "co2_dataset.parquet")
    arrays = build(table, measured())
    out = DERIVED / "co2_sequences.npz"
    np.savez_compressed(out, **arrays)
    print(
        f"{out}: {len(arrays['issue_time'])} issues, past {arrays['past'].shape}, "
        f"future {arrays['future'].shape}"
    )


if __name__ == "__main__":
    main()
