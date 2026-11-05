"""Training tables for the wind tasks, one row per (issue day, hour 0-167), like solar.

Inputs are 100 m wind speeds averaged over the group's points: GEFS ensemble mean and
spread per 3 h step, interpolated to hours; ECMWF (Open-Meteo, from 2024-03) per hour with
lead ceil(h/24), mean and spread over points. Target y = production / cap, where cap is the
99th percentile of hourly production over the 60 days before issue (a capacity proxy).
History features at issue: y_last (production of the last hour / cap), y_1d (mean of the
last 24 hours / cap), err_1d (mean of y minus the power curve of the GEFS wind over the
previous issue's first 24 hours). Onshore production is the NED series, offshore the
measured ENTSO-E series (MW -> kWh).

Run: PYTHONPATH=src uv run python -m data.features.wind onshore|offshore
"""

import sys

import numpy as np
import pandas as pd

from data.features.gefs import gefs_3h, to_hours
from data.loader import DERIVED, entsoe, ned, openmeteo

ISSUE_HOUR = 6
CUT_IN, RATED = 3.0, 12.0  # m/s, naive power curve
N_HOURS = 168
CAP_WINDOW_H = 60 * 24


def production(group: str) -> pd.Series:
    """Hourly production in kWh."""
    if group == "onshore":
        return ned("wind_onshore")
    return entsoe("actual_wind_offshore") * 1000.0


def gefs_wind(group: str) -> pd.DataFrame:
    """Per run and 3 h step: wind speed mean and spread, keyed by hours since issue."""
    g = gefs_3h()
    return pd.DataFrame(
        {
            "issue_time": (g.init_time + pd.Timedelta(hours=ISSUE_HOUR)).to_numpy(),
            "t_rel": (g.lead_h - ISSUE_HOUR).to_numpy(),
            "ws": g[f"{group}_ws_mean"].to_numpy(),
            "ws_std": g[f"{group}_ws_std"].to_numpy(),
        }
    )


def ecmwf_wind(group: str) -> pd.DataFrame:
    """Mean and spread over points of the hour-centre speed ((v(t) + v(t+1)) / 2), indexed by
    (hour start, lead)."""
    d = openmeteo(group).loc[lambda d: d.lead >= 1]
    wide = d.pivot_table(index="time", columns=["lead", "point"], values="value")
    centre = (wide + wide.shift(-1)) / 2
    out = pd.concat(
        {
            "ws_ecmwf": centre.T.groupby(level="lead").mean().T.stack(),
            "ws_std_ecmwf": centre.T.groupby(level="lead").std().T.stack(),
        },
        axis=1,
    )
    return out.rename_axis(["time", "lead"])


def power_curve(ws: pd.Series) -> pd.Series:
    """Cubic ramp from CUT_IN to RATED, as a fraction of capacity."""
    return ((ws - CUT_IN) / (RATED - CUT_IN)).clip(0, 1) ** 3


def build(group: str) -> pd.DataFrame:
    target = production(group)
    cap = target.rolling(CAP_WINDOW_H, min_periods=24 * 14).quantile(0.99)

    gefs = gefs_wind(group)
    issues = np.sort(gefs.issue_time.unique())
    h = np.arange(N_HOURS)
    df = pd.DataFrame(
        {"issue_time": np.repeat(issues, N_HOURS), "h": np.tile(h, len(issues))}
    )
    df["valid_time"] = df.issue_time + pd.to_timedelta(df.h, unit="h")
    df["ws_gefs"] = to_hours(gefs, "ws", N_HOURS)
    df["ws_std_gefs"] = to_hours(gefs, "ws_std", N_HOURS)
    df["lead_ecmwf"] = np.maximum(1, np.ceil(df.h / 24)).astype(int)
    df["cap"] = cap.reindex(df.issue_time - pd.Timedelta(hours=1)).to_numpy()

    ecmwf = ecmwf_wind(group).reindex(
        pd.MultiIndex.from_arrays([df.valid_time, df.lead_ecmwf])
    )
    df["ws_ecmwf"] = ecmwf["ws_ecmwf"].to_numpy()
    df["ws_std_ecmwf"] = ecmwf["ws_std_ecmwf"].to_numpy()

    df["wind_kwh"] = target.reindex(df.valid_time).to_numpy()
    df["y"] = df.wind_kwh / df.cap

    before_issue = df.issue_time - pd.Timedelta(hours=1)
    df["y_last"] = target.reindex(before_issue).to_numpy() / df.cap
    mean_1d = pd.Series(target.rolling(24, min_periods=12).mean(), index=target.index)
    df["y_1d"] = mean_1d.reindex(before_issue).to_numpy() / df.cap
    err = (df["y"] - power_curve(pd.Series(df["ws_gefs"]))).where(df["h"] < 24)
    err_by_issue = pd.Series(err.groupby(df["issue_time"]).mean())
    df["err_1d"] = err_by_issue.reindex(df.issue_time - pd.Timedelta(days=1)).to_numpy()
    return df


def main() -> None:
    group = sys.argv[1]
    df = build(group)
    out = DERIVED / f"{group}_dataset.parquet"
    df.to_parquet(out, index=False)
    print(
        f"{out}: {len(df)} rows, issues {df.issue_time.min():%Y-%m-%d} .. "
        f"{df.issue_time.max():%Y-%m-%d}; missing: target {df.wind_kwh.isna().mean():.1%}, "
        f"ws_ecmwf {df.ws_ecmwf.isna().mean():.1%}"
    )


if __name__ == "__main__":
    main()
