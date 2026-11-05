"""Readers for data/raw. Each function returns one source with UTC timestamps; hourly timestamps mark the start of
the hour. Provenance and caveats: data/README.md.

    ned(name, column)   NED hourly series: data/raw/<name>_cls2_*.csv, e.g. ned("solar"), ned("electricity_mix", "emissionfactor")
    entsoe(name)        ENTSO-E hourly series: data/raw/entsoe_<name>_*.csv, e.g. entsoe("actual_wind_offshore")
    openmeteo(group)    ECMWF forecasts with known lead (Open-Meteo Previous Runs), long table: time, point, lead, value
    gefs_stores(box)    paths of the monthly NOAA GEFS ensemble stores; gefs(store) opens one lazily
    outages()           ENTSO-E unavailability announcements (A80, production units)
"""

from pathlib import Path

import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"


def _concat_csv(pattern: str, **kwargs) -> pd.DataFrame:
    files = sorted(RAW.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files match data/raw/{pattern}")
    return pd.concat((pd.read_csv(f, **kwargs) for f in files), ignore_index=True)


def ned(name: str, column: str = "volume") -> pd.Series:
    """Hourly NED series (measured, classification 2). `volume` is kWh per hour, `emissionfactor` kg CO2 / kWh.
    Files overlap at their edges; the later file wins (NED revises history)."""
    df = _concat_csv(f"{name}_cls2_*.csv", usecols=["validfrom", column])
    df["validfrom"] = pd.to_datetime(df["validfrom"], utc=True)
    df = (
        df.drop_duplicates("validfrom", keep="last").set_index("validfrom").sort_index()
    )
    return pd.Series(df[column].to_numpy(), index=df.index.rename("time"), name=name)


def entsoe(name: str) -> pd.Series:
    """ENTSO-E series in MW (EUR/MWh for dayahead_price), 15-minute values averaged to hours."""
    df = _concat_csv(f"entsoe_{name}_*.csv")
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    s = (
        df.drop_duplicates("time_utc", keep="last")
        .set_index("time_utc")
        .iloc[:, 0]
        .sort_index()
    )
    s = s.resample("h").mean()
    return pd.Series(s.to_numpy(), index=s.index.rename("time"), name=name)


def openmeteo(group: str) -> pd.DataFrame:
    """ECMWF IFS 0.25 forecasts from the Open-Meteo Previous Runs API, available from 2024-03.
    group: offshore | onshore (wind_speed_100m, m/s, instantaneous at `time`), solar (shortwave_radiation, W/m2,
    mean over the hour ending at `time`), temperature (temperature_2m, degC).
    lead = N: value from the run made about 24*N hours before `time`; N = 0 is the latest run (near analysis).
    """
    files = sorted((RAW / "openmeteo").glob(f"{group}_*.csv"))
    df = pd.concat(
        (
            pd.read_csv(
                f, usecols=["time", "point", "lead", "value"], parse_dates=["time"]
            )
            for f in files
        ),
        ignore_index=True,
    )
    return df


def gefs_stores(box: str = "nl") -> list[Path]:
    """Monthly Zarr stores of the NOAA GEFS ensemble (dynamical.org), daily 00 UTC runs from 2020-10-01, 31 members,
    leads 0-189 h in 3 h steps, 0.25 deg grid.
    box="nl": wind u/v 100 m and surface shortwave radiation over 54.5-50.75 N, 2.75-7.25 E.
    box="wind": wind only, wider box 56-50.75 N, 2.75-11.75 E (data/raw/gefs_neighbors).
    Radiation is the mean since the previous 6-hourly step (6 h at 00/06/12/18 UTC, 3 h at 03/09/15/21).
    """
    folder, prefix = {
        "nl": ("gefs", "gefs_nl"),
        "wind": ("gefs_neighbors", "gefs_wind"),
    }[box]
    return sorted((RAW / folder).glob(f"{prefix}_*.zarr"))


def gefs(store: Path) -> xr.Dataset:
    """One monthly GEFS store, lazy (dims init_time, ensemble_member, lead_time, latitude, longitude).
    Values are float16; cast to float32 before averaging to avoid overflow."""
    return xr.open_zarr(store, consolidated=False, chunks=None)


def outages() -> pd.DataFrame:
    """ENTSO-E A80 production unavailability: one row per announcement, revisions included (`mrid`, `revision`).
    Columns: unit_name, fuel, nominal_mw, available_mw, outage_start_utc, outage_end_utc, created_utc, doc_status
    (A05 active, A09 cancelled, A13 withdrawn)."""
    df = _concat_csv("entsoe_outages_A80_*.csv")
    for c in [
        "created_utc",
        "outage_start_utc",
        "outage_end_utc",
        "avail_start_utc",
        "avail_end_utc",
    ]:
        df[c] = pd.to_datetime(df[c], utc=True)
    return df
