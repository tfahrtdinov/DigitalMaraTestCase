"""Training table for the solar task: one row per (issue day, hour 0-167).

Issue time is 06 UTC; the GEFS run of the same day 00 UTC gives leads 6-173 h. Radiation is
expressed as a clearness index k = radiation / clear-sky GHI: from GEFS per 3 h window,
interpolated to hours; from ECMWF (Open-Meteo, from 2024-03) per hour, lead ceil(h/24).
Target y = production / cs_ghi (kWh per W/m2 of clear sky). Night rows (clear sky < CS_MIN)
have NaN. Capacity and recent history enter as features: eta, the 95th percentile of
production per W/m2 of clear sky over the 28 days before issue; eta_1d, the same ratio over
the 24 hours before issue; k_err_1d, the mean error of the GEFS clearness index over the
previous issue's first 24 hours (actual minus forecast, daytime hours).

Run: PYTHONPATH=src uv run python -m data.features.solar
"""

from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from data.features.gefs import gefs_3h, to_hours
from data.loader import DERIVED, ned, openmeteo

OUT = DERIVED / "solar_dataset.parquet"
ISSUE_HOUR = 6
N_HOURS = 168
CS_MIN = 20.0  # W/m2: below this clear-sky level ratios are undefined
ETA_CS_MIN = 50.0  # W/m2: hours used to estimate eta
ETA_WINDOW_H = 28 * 24
LAT, LON = 52.2, 5.3  # clear-sky point
# Haurwitz (1945) clear-sky model: GHI = A cos(z) exp(-B / cos(z))
HAURWITZ_A, HAURWITZ_B = 1098.0, 0.059
# Spencer (1971) Fourier series in the fractional year, as in the NOAA solar calculator:
# (a0, cos coefficients for k = 1..n, sin coefficients for k = 1..n)
SPENCER_DECLINATION = (
    0.006918,
    (-0.399912, -0.006758, -0.002697),
    (0.070257, 0.000907, 0.00148),
)
SPENCER_EQUATION_OF_TIME = (0.000075, (0.001868, -0.014615), (-0.032077, -0.040849))


def _fourier(
    angle: NDArray[np.float64],
    constant: float,
    cos_coefs: tuple[float, ...],
    sin_coefs: tuple[float, ...],
) -> NDArray[np.float64]:
    """constant + sum_k cos_coefs[k] cos(k angle) + sin_coefs[k] sin(k angle), k = 1..n."""
    k = np.arange(1, len(cos_coefs) + 1)[:, None]
    return (
        constant
        + np.matmul(cos_coefs, np.cos(k * angle))
        + np.matmul(sin_coefs, np.sin(k * angle))
    )


def clear_sky_ghi(times: pd.DatetimeIndex) -> NDArray[np.float64]:
    """Clear-sky GHI (W/m2) at UTC instants: Haurwitz (1945) model on the NOAA solar
    position approximation (Spencer series for declination and equation of time)."""
    hour, minute = np.asarray(times.hour), np.asarray(times.minute)
    doy = np.asarray(times.day_of_year) + hour / 24 + minute / 1440
    year_angle = 2 * np.pi / 365 * (doy - 1)  # fractional year, radians
    decl = _fourier(year_angle, *SPENCER_DECLINATION)
    eqt = 229.18 * _fourier(year_angle, *SPENCER_EQUATION_OF_TIME)  # minutes
    solar_min = hour * 60 + minute + eqt + 4 * LON
    ha = np.deg2rad(solar_min / 4 - 180)
    lat = np.deg2rad(LAT)
    cosz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, 0, None)
    safe = np.where(cosz > 0, cosz, 1)
    return np.where(cosz > 0, HAURWITZ_A * cosz * np.exp(-HAURWITZ_B / safe), 0.0)


def hourly_clear_sky(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Mean clear-sky GHI over each hour [t, t+1), sampled every 10 minutes."""
    hours = pd.date_range(start, end, freq="h", tz="UTC")
    vals = np.zeros(len(hours))
    for m in range(5, 60, 10):
        vals += clear_sky_ghi(hours + pd.Timedelta(minutes=m))
    return pd.Series(vals / 6, index=hours, name="cs_ghi")


def gefs_clearness(cs: pd.Series) -> pd.DataFrame:
    """Per run and 3 h window: clearness index and its ensemble spread, keyed by the window
    centre in hours since issue (`t_rel`)."""
    g = gefs_3h().loc[lambda d: d.lead_h > 0]  # lead 0 has no radiation window
    cs_window = pd.Series(  # mean clear sky over the 3 h window ending at valid_time
        np.asarray(cs.rolling(3).mean()),
        index=pd.DatetimeIndex(cs.index) + pd.Timedelta(hours=1),
    )
    cs_w = cs_window.reindex(g.valid_time).to_numpy()
    ok = cs_w >= CS_MIN
    return pd.DataFrame(
        {
            "issue_time": (g.init_time + pd.Timedelta(hours=ISSUE_HOUR)).to_numpy(),
            "t_rel": (g.lead_h - 1.5 - ISSUE_HOUR).to_numpy(),
            "k": np.where(ok, g.solar_rad_mean / cs_w, np.nan),
            "k_std": np.where(ok, g.solar_rad_std / cs_w, np.nan),
        }
    )


def ecmwf_radiation() -> pd.DataFrame:
    """Mean and spread over the 10 points, indexed by (hour start, lead)."""
    d = openmeteo("solar").loc[lambda d: d.lead >= 1]
    d["time"] = d["time"] - pd.Timedelta(
        hours=1
    )  # value at t is the mean over (t-1, t]
    g = d.groupby(["time", "lead"])["value"].agg(["mean", "std"])
    return pd.DataFrame({"rad_ecmwf": g["mean"], "rad_std_ecmwf": g["std"]})


def build() -> pd.DataFrame:
    target = ned("solar")
    first = cast(pd.Timestamp, target.index.min()) - pd.Timedelta(hours=3)
    cs = hourly_clear_sky(
        cast(pd.Timestamp, first),
        cast(pd.Timestamp, pd.Timestamp("2027-01-01", tz="UTC")),
    )
    cs_target = cs.reindex(target.index)
    eta = (target / cs_target).where(cs_target >= ETA_CS_MIN)
    eta = eta.rolling(ETA_WINDOW_H, min_periods=100).quantile(0.95)

    gefs = gefs_clearness(cs)
    issues = np.sort(gefs.issue_time.unique())
    h = np.arange(N_HOURS)
    df = pd.DataFrame(
        {"issue_time": np.repeat(issues, N_HOURS), "h": np.tile(h, len(issues))}
    )
    df["valid_time"] = df.issue_time + pd.to_timedelta(df.h, unit="h")
    df["cs_ghi"] = cs.reindex(df.valid_time).to_numpy()
    df["k_gefs"] = to_hours(gefs, "k", N_HOURS)
    df["k_std_gefs"] = to_hours(gefs, "k_std", N_HOURS)
    df["lead_ecmwf"] = np.maximum(1, np.ceil(df.h / 24)).astype(int)
    df["eta"] = eta.reindex(df.issue_time - pd.Timedelta(hours=1)).to_numpy()

    ecmwf = ecmwf_radiation().reindex(
        pd.MultiIndex.from_arrays([df.valid_time, df.lead_ecmwf])
    )
    df["k_ecmwf"] = ecmwf["rad_ecmwf"].to_numpy() / df.cs_ghi
    df["k_std_ecmwf"] = ecmwf["rad_std_ecmwf"].to_numpy() / df.cs_ghi

    night = df.cs_ghi < CS_MIN
    df.loc[night, ["k_gefs", "k_std_gefs", "k_ecmwf", "k_std_ecmwf"]] = np.nan
    df["k_gefs"] = df["k_gefs"].clip(0, 1.3)
    df["k_ecmwf"] = df["k_ecmwf"].clip(0, 1.3)
    df["solar_kwh"] = target.reindex(df.valid_time).to_numpy()
    df["y"] = (df.solar_kwh / df.cs_ghi).where(~night)

    day = cs_target >= ETA_CS_MIN
    kwh_1d = target.where(day).rolling(24, min_periods=1).sum()
    cs_1d = cs_target.where(day).rolling(24, min_periods=1).sum()
    eta_1d = pd.Series(np.asarray(kwh_1d) / np.asarray(cs_1d), index=target.index)
    df["eta_1d"] = eta_1d.reindex(df.issue_time - pd.Timedelta(hours=1)).to_numpy()
    k_err = (df["solar_kwh"] / (df["eta"] * df["cs_ghi"]) - df["k_gefs"]).where(
        df["h"] < 24
    )
    k_err_by_issue = pd.Series(k_err.groupby(df["issue_time"]).mean())
    df["k_err_1d"] = k_err_by_issue.reindex(
        df.issue_time - pd.Timedelta(days=1)
    ).to_numpy()
    return df


def main() -> None:
    df = build()
    df.to_parquet(OUT, index=False)
    night = df.cs_ghi < CS_MIN
    print(
        f"{OUT}: {len(df)} rows, issues {df.issue_time.min():%Y-%m-%d} .. "
        f"{df.issue_time.max():%Y-%m-%d}; missing: target {df.solar_kwh.isna().mean():.1%}, "
        f"k_ecmwf daytime {df.k_ecmwf[~night].isna().mean():.1%}"
    )


if __name__ == "__main__":
    main()
