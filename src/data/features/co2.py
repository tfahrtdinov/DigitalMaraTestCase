"""Training table for the CO2 models, one row per (issue day, hour 0-167).

Targets in g/kWh: y = CO2 factor at the valid hour minus the factor at the last hour before
issue (days 1-2), y_far = the factor minus its 28-day profile at that hour (days 3-7). Inputs: LightGBM renewables forecasts (data/derived/forecasts/<task>_lgbm_<period>.parquet),
GEFS and ECMWF weather from the renewables tables, GEFS wind over Germany and Belgium,
ENTSO-E day-ahead values for the hours of the issue's local day, outages announced before
issue, and every measured series up to the last hour before issue. y_ref is the TimesFM-3
forecast of the factor, 2025-01 .. 2026-10.

Run: PYTHONPATH=src uv run python -m data.features.co2
"""

from typing import cast

import numpy as np
import pandas as pd

from data.features.gefs import POINTS, gefs_neighbor_wind
from data.loader import DERIVED, entsoe, ned, openmeteo, outages

N_HOURS = 168
FORECASTS = DERIVED / "forecasts"
RENEWABLES_PERIOD = "2021-10-01_2026-11-03"  # file suffix of lgbm.run_holdout
NEIGHBORS = [g for g in POINTS if g[:3] in ("de_", "be_")]
FIRST_ISSUE = "2021-01-29"  # 28 days after the factor becomes usable
LOCAL_TZ = "Europe/Amsterdam"
MIN_NON_RENEWABLE_MW = 1000  # below this the intensity is noise
OUTAGE_FUELS = ["gas", "hard_coal", "nuclear", "biomass", "wind_offshore"]
WEATHER = {
    "solar": ["k_gefs", "k_std_gefs", "k_ecmwf", "k_std_ecmwf", "cs_ghi"],
    "onshore": ["ws_gefs", "ws_std_gefs", "ws_ecmwf", "ws_std_ecmwf"],
    "offshore": ["ws_gefs", "ws_std_gefs", "ws_ecmwf", "ws_std_ecmwf"],
}
# series -> statistics at issue: last hour, 24 h and 168 h means, 28-day hourly profile
STATE = {
    "co2": ["last", "m24", "m168", "prof"],
    "ren": ["last", "m24", "prof"],
    "solar": ["last"],
    "onshore": ["last"],
    "offshore": ["last"],
    "gas": ["last", "m24", "m168", "prof"],
    "coal": ["last", "m24", "m168", "prof"],
    "chp": ["last", "m24"],
    "waste": ["last", "m24"],
    "biomass": ["last", "m24"],
    "nuclear": ["last", "m24"],
    "other": ["last", "m24"],
    "total": ["last", "m24", "prof"],
    "load": ["last", "m24", "prof"],
    "intensity": ["m24", "m168"],
    "price": ["m24", "m168"],
}


def measured() -> pd.DataFrame:
    """Hourly measured series: CO2 in g/kWh, price in EUR/MWh, the others in MW."""
    s = {
        "co2": ned("electricity_mix", "emissionfactor") * 1000,
        "total": ned("electricity_mix") / 1000,
        "solar": ned("solar") / 1000,
        "onshore": ned("wind_onshore") / 1000,
        "offshore": entsoe("actual_wind_offshore"),
        "gas": ned("fossil_gas") / 1000,
        "coal": ned("fossil_coal") / 1000,
        "chp": ned("type35") / 1000,
        "waste": ned("type21") / 1000,
        "biomass": ned("type25") / 1000,
        "other": ned("type26") / 1000,
        "nuclear": entsoe("actual_nuclear"),
        "load": ned("electricity_load_consuming") / 1000,
        "price": entsoe("dayahead_price"),
    }
    df = pd.DataFrame(s).loc["2021-01-01":]
    df["ren"] = df.solar + df.onshore + df.offshore
    non_renewable = (df.total - df.ren).where(lambda x: x > MIN_NON_RENEWABLE_MW)
    df["intensity"] = df.co2 * df.total / non_renewable
    return df


def hourly_profile(s: pd.Series) -> pd.Series:
    """Mean of the same UTC hour over the 28 days ending at each timestamp."""
    return s.groupby(pd.DatetimeIndex(s.index).hour).transform(
        lambda x: x.rolling(28, min_periods=14).mean()
    )


def announced_outages(issue_times: pd.DatetimeIndex) -> pd.DataFrame:
    """Unavailable MW per fuel and hour as announced before each issue: for every
    announcement its latest version published before issue, unless cancelled or withdrawn.
    """
    o = pd.concat([outages("A80"), outages("A77")], ignore_index=True)
    o = cast(pd.DataFrame, o[o["fuel"].isin(OUTAGE_FUELS)]).dropna(
        subset=["nominal_mw"]
    )
    o["mw"] = o.nominal_mw - o.available_mw
    arrays = {f: np.zeros((len(issue_times), N_HOURS)) for f in OUTAGE_FUELS}
    hour = pd.Timedelta(hours=1)
    for i, t in enumerate(issue_times):
        known = cast(pd.DataFrame, o[o["created_utc"] < t])
        known = known[
            known["revision"] == known.groupby("mrid")["revision"].transform("max")
        ]
        active = cast(
            pd.DataFrame,
            known[
                (known.doc_status == "A05")
                & (known.avail_end_utc > t)
                & (known.avail_start_utc < t + N_HOURS * hour)
            ],
        )
        start = cast(pd.Series, np.ceil((active.avail_start_utc - t) / hour))
        end = cast(pd.Series, np.ceil((active.avail_end_utc - t) / hour))
        first = start.clip(lower=0).astype(int)
        last = end.clip(upper=N_HOURS).astype(int)
        for fuel, a, b, mw in zip(active.fuel, first, last, active.mw, strict=True):
            arrays[fuel][i, a:b] += mw
    return pd.DataFrame({f"out_{f}": v.ravel() for f, v in arrays.items()})


def renewable_forecasts() -> pd.DataFrame:
    """Issue, hour and forecast in kWh of the three renewables, where all three exist."""
    tables = []
    for task in ["solar", "onshore", "offshore"]:
        t = pd.read_parquet(
            FORECASTS / f"{task}_lgbm_{RENEWABLES_PERIOD}.parquet",
            columns=["issue_time", "h", "y_hat"],
        )
        t["issue_time"] = pd.to_datetime(t["issue_time"], utc=True)
        tables.append(t.set_index(["issue_time", "h"]).rename(columns={"y_hat": task}))
    return pd.concat(tables, axis=1, join="inner").reset_index()


def build() -> pd.DataFrame:
    df = renewable_forecasts()
    df = cast(pd.DataFrame, df[df["issue_time"] >= FIRST_ISSUE])
    df = df.sort_values(["issue_time", "h"])
    df = df.reset_index(drop=True)
    for c in ["solar", "onshore", "offshore"]:
        df[f"{c}_f"] = df.pop(c).clip(lower=0) / 1000
    df["ren_f"] = df.solar_f + df.onshore_f + df.offshore_f
    next_hour = df.groupby("issue_time").onshore_f.shift(-1)
    df["onshore_f_next"] = next_hour.fillna(df.onshore_f)
    df["valid_time"] = df.issue_time + pd.to_timedelta(df.h, unit="h")

    m = measured()
    before = df.issue_time - pd.Timedelta(hours=1)
    # latest timestamp before issue with the hour of day of the valid hour
    same_hour = df.valid_time - pd.to_timedelta(24 * ((df.h + 24) // 24), unit="h")
    means = {k: m.rolling(k, min_periods=k // 2).mean() for k in (24, 168)}
    profiles = m.apply(hourly_profile)
    for name, stats in STATE.items():
        for stat in stats:
            if stat == "prof":
                values = pd.Series(profiles[name]).reindex(same_hour)
            elif stat == "last":
                values = pd.Series(m[name]).reindex(before)
            else:
                values = pd.Series(means[int(stat[1:])][name]).reindex(before)
            df[f"{name}_{stat}"] = values.to_numpy()

    df["co2_t2"] = (
        pd.Series(m["co2"]).reindex(df.issue_time - pd.Timedelta(hours=2)).to_numpy()
    )
    df["co2_t3"] = m.co2.reindex(df.issue_time - pd.Timedelta(hours=3)).to_numpy()
    df["co2_same_hour"] = m.co2.reindex(same_hour).to_numpy()
    deviation = m.co2 - profiles.co2
    df["co2_dev_last"] = deviation.reindex(before).to_numpy()
    df["co2_dev_m24"] = (
        deviation.rolling(24, min_periods=12).mean().reindex(before).to_numpy()
    )

    df["ren_share_f"] = df.ren_f / df.total_prof
    df["ren_step_f"] = df.ren_f - df.ren_last
    df["ren_dev_f"] = df.ren_f - df.ren_prof
    df["co2_prior"] = df.intensity_m168 * (1 - df.ren_share_f.clip(0, 1))

    valid_local = df.valid_time.dt.tz_convert(LOCAL_TZ)
    # forecast errors in timing grow with lead: means around the hour and over its day
    day = valid_local.dt.date
    for c in ["ren_f", "ren_share_f"]:
        for half in (3, 12):
            df[f"{c}_w{half}"] = df.groupby("issue_time")[c].transform(
                lambda x, n=2 * half + 1: x.rolling(
                    n, center=True, min_periods=1
                ).mean()
            )
        df[f"{c}_day"] = df.groupby(["issue_time", day])[c].transform("mean")

    df["hour"] = df.valid_time.dt.hour
    df["dow"] = valid_local.dt.day_of_week
    df["month"] = valid_local.dt.month

    # day-ahead values of the issue's local day were published the day before
    today = valid_local.dt.date == df.issue_time.dt.tz_convert(LOCAL_TZ).dt.date
    for name, source in {
        "da_load": "dayahead_load",
        "da_gen": "dayahead_total_generation",
        "da_solar": "dayahead_solar",
        "da_onshore": "dayahead_wind_onshore",
        "da_offshore": "dayahead_wind_offshore",
        "da_price": "dayahead_price",
    }.items():
        df[name] = (
            entsoe(source).reindex(df.valid_time).where(today.to_numpy()).to_numpy()
        )
    df["da_ren"] = df.da_solar + df.da_onshore + df.da_offshore
    df["da_resid"] = df.da_gen - df.da_ren

    for group, columns in WEATHER.items():
        w = pd.read_parquet(
            DERIVED / f"{group}_dataset.parquet", columns=["issue_time", "h", *columns]
        )
        if group != "solar":
            w = w.rename(columns={c: f"{group}_{c}" for c in columns})
        df = df.merge(w, on=["issue_time", "h"], how="left")

    df = df.merge(gefs_neighbor_wind(NEIGHBORS), on=["issue_time", "h"], how="left")

    lead = np.maximum(1, np.ceil(df.h / 24)).astype(int)
    t = openmeteo("temperature")
    temp = t[t.lead >= 1].groupby(["time", "lead"]).value.mean()
    df["temp_f"] = temp.reindex(
        pd.MultiIndex.from_arrays([df.valid_time, lead])
    ).to_numpy()

    issues = pd.DatetimeIndex(df.issue_time.unique())
    df = pd.concat([df, announced_outages(issues)], axis=1)

    ref = pd.read_parquet(
        DERIVED / "references" / "co2.parquet", columns=["issue_time", "h", "timesfm3"]
    ).rename(columns={"timesfm3": "y_ref"})
    df = df.merge(ref, on=["issue_time", "h"], how="left")

    df["co2"] = m.co2.reindex(df.valid_time).to_numpy()
    df["y"] = df.co2 - df.co2_last
    df["y_far"] = df.co2 - df.co2_prof
    return df


def main() -> None:
    df = build()
    out = DERIVED / "co2_dataset.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    print(
        f"{out}: {len(df)} rows, {df.shape[1]} columns, issues "
        f"{df.issue_time.min():%Y-%m-%d} .. {df.issue_time.max():%Y-%m-%d}; "
        f"missing target {df.y.isna().mean():.1%}"
    )


if __name__ == "__main__":
    main()
