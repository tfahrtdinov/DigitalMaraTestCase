"""Arrays of the neural models from the CO2 table, one file that needs no raw data:
data/derived/co2_sequences.npz (days 1-2) and co2_far_sequences.npz (days 3-7). Past: 336 hours of every measured series before issue and
the calendar. Future: the inputs of the CO2 table that change within a forecast window and
the calendar. Table: every input of the LightGBM model, for the MLP. Channels with gaps get
a flag channel; the flag carries the missing values.

Run: PYTHONPATH=src uv run python -m data.features.co2_sequences
"""

from typing import cast

import numpy as np
import pandas as pd

from data.features.co2 import measured
from data.loader import DERIVED
from tasks.co2 import (
    CALENDAR,
    DAY_AHEAD,
    FAR_FEATURES,
    FEATURES,
    FORECAST,
    OUTAGES,
    WEATHER_COLUMNS,
)

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
# file -> columns of the future channels and of the MLP table, target, what it is relative to
SETS = {
    "co2_sequences": (FUTURE, FEATURES, "y", "co2_last"),
    "co2_far_sequences": (
        [c for c in FAR_FEATURES if c not in CALENDAR],
        FAR_FEATURES,
        "y_far",
        "co2_prof",
    ),
}


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


def build(
    table: pd.DataFrame,
    hourly: pd.DataFrame,
    future_columns: list[str],
    table_columns: list[str],
    target: str,
    base: str,
) -> dict:
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
            table[future_columns],
            calendar(pd.DatetimeIndex(table.valid_time)).reset_index(drop=True),
            (table.h / N_HOURS).rename("h"),
        ],
        axis=1,
    )
    future = with_gap_flags(cast(pd.DataFrame, future))
    rows = with_gap_flags(cast(pd.DataFrame, table[table_columns]))
    n = len(issues)
    shape = (n, N_HOURS)
    return {
        "issue_time": issues[keep].tz_localize(None).to_numpy("datetime64[ns]"),
        "past": past.to_numpy(np.float32)[windows],
        "future": future.to_numpy(np.float32).reshape(*shape, -1)[keep],
        "table": rows.to_numpy(np.float32).reshape(*shape, -1)[keep],
        "y": table[target].to_numpy(np.float32).reshape(shape)[keep],
        "base": table[base].to_numpy(np.float32).reshape(shape)[keep],
        "y_ref": table.y_ref.to_numpy(np.float32).reshape(shape)[keep],
        "past_columns": np.array(past.columns),
        "future_columns": np.array(future.columns),
    }


def main() -> None:
    table = pd.read_parquet(DERIVED / "co2_dataset.parquet")
    hourly = measured()
    for name, (future, rows, target, base) in SETS.items():
        arrays = build(table, hourly, future, rows, target, base)
        out = DERIVED / f"{name}.npz"
        np.savez_compressed(out, **arrays)
        print(
            f"{out}: {len(arrays['issue_time'])} issues, past {arrays['past'].shape}, "
            f"future {arrays['future'].shape}, table {arrays['table'].shape}"
        )


if __name__ == "__main__":
    main()
