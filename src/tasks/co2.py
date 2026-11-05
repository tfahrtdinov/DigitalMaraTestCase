"""CO2 factor tasks, see data/features/co2.py. Days 1-2: target is the change from the last
hour before issue, all inputs. Days 3-7: target is the deviation from the 28-day profile,
without the state at issue and the day-ahead values, with smoothed renewables forecasts.
Metrics are computed on the factor in g/kWh, the reference forecast is TimesFM-3."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from data import DERIVED
from data.features.co2 import NEIGHBORS, STATE, WEATHER
from tasks.task import Task, lgbm_space

FORECAST = [
    "solar_f",
    "onshore_f",
    "onshore_f_next",
    "offshore_f",
    "ren_f",
    "ren_share_f",
    "ren_step_f",
    "ren_dev_f",
    "co2_prior",
]
HISTORY = [
    *(f"{name}_{stat}" for name, stats in STATE.items() for stat in stats),
    "co2_t2",
    "co2_t3",
    "co2_same_hour",
    "co2_dev_last",
    "co2_dev_m24",
]
CALENDAR = ["h", "hour", "dow", "month"]
DAY_AHEAD = [
    "da_load",
    "da_gen",
    "da_solar",
    "da_onshore",
    "da_offshore",
    "da_ren",
    "da_resid",
    "da_price",
]
WEATHER_COLUMNS = [
    *WEATHER["solar"],
    *(f"{g}_{c}" for g in ("onshore", "offshore") for c in WEATHER[g]),
    "temp_f",
    *(f"{g}_ws_{stat}" for g in NEIGHBORS for stat in ("mean", "std")),
]
OUTAGES = [
    "out_gas",
    "out_hard_coal",
    "out_nuclear",
    "out_biomass",
    "out_wind_offshore",
]
FEATURES = [*FORECAST, *HISTORY, *CALENDAR, *DAY_AHEAD, *WEATHER_COLUMNS, *OUTAGES]


FAR_FEATURES = [
    "solar_f",
    "onshore_f",
    "onshore_f_next",
    "offshore_f",
    "ren_f",
    "ren_share_f",
    "ren_dev_f",
    *(f"{c}_{w}" for c in ("ren_f", "ren_share_f") for w in ("w3", "w12", "day")),
    *WEATHER_COLUMNS,
    "co2_m168",
    "co2_prof",
    "intensity_m168",
    "co2_prior",
    "gas_m168",
    "gas_prof",
    "coal_m168",
    "coal_prof",
    "total_prof",
    "load_prof",
    *OUTAGES,
    *CALENDAR,
]


def scoring(base: str) -> Callable[[pd.DataFrame, NDArray[np.float64]], pd.DataFrame]:
    """Adds the column `base` back to the predicted target."""

    def table(df: pd.DataFrame, y_hat: NDArray[np.float64]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "issue_time": df["issue_time"].to_numpy(),
                "h": df["h"].to_numpy(),
                "y": df["co2"].to_numpy(),
                "y_hat": np.clip(df[base].to_numpy() + y_hat, 0, None),
                "y_naive": df["y_ref"].to_numpy(),
            }
        )

    return table


def load() -> pd.DataFrame:
    return pd.read_parquet(DERIVED / "co2_dataset.parquet")


CO2 = Task("co2", FEATURES, "y", scoring("co2_last"), load, lgbm_space)
CO2_FAR = Task("co2_far", FAR_FEATURES, "y_far", scoring("co2_prof"), load, lgbm_space)
