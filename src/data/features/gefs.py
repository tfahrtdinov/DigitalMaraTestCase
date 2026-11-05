"""GEFS ensemble reduced to a table per run and lead time (3 h steps), cached per month in
data/derived/gefs_3h/. A month is rebuilt when its Zarr store is newer than the cache.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from data.loader import DERIVED, gefs, gefs_stores

CACHE = DERIVED / "gefs_3h"
# de_* and be_* lie outside the Dutch box: approximate centres of wind regions,
# read from the wider store (box="wind")
POINTS: dict[str, dict[str, tuple[float, float]]] = {
    "de_offshore": {
        "borkum": (54.00, 6.50),
        "helgoland": (54.40, 7.60),
        "sylt": (55.00, 7.20),
    },
    "de_onshore": {
        "emsland": (52.70, 7.40),
        "lower_saxony": (53.20, 8.50),
        "schleswig_holstein": (54.50, 9.00),
        "mecklenburg": (53.80, 11.50),
    },
    "be_offshore": {"belgian_zone": (51.60, 2.90)},
    "be_onshore": {"flanders": (51.10, 3.80), "hainaut": (50.80, 4.20)},
    "offshore": {
        "borssele": (51.70, 3.00),
        "hk_zuid": (52.30, 4.05),
        "hk_noord": (52.72, 4.25),
        "ijmuiden_ver": (52.85, 3.75),
        "gemini": (54.04, 5.97),
    },
    "onshore": {
        "flevoland": (52.53, 5.55),
        "eemshaven": (53.40, 6.80),
        "zeeland": (51.55, 3.90),
        "wieringermeer": (52.85, 4.98),
        "friesland": (53.05, 5.35),
        "drenthe": (52.90, 6.95),
        "maasvlakte": (51.95, 4.05),
    },
    "solar": {
        "brabant": (51.55, 5.10),
        "gelderland": (52.05, 5.90),
        "zuid_holland": (52.00, 4.50),
        "groningen": (53.20, 6.60),
        "overijssel": (52.40, 6.45),
        "limburg": (51.20, 5.90),
        "friesland": (53.10, 5.80),
        "noord_holland": (52.60, 4.85),
        "zeeland": (51.50, 3.85),
        "utrecht": (52.08, 5.12),
    },
}


def _cells(ds: xr.Dataset, group: str) -> dict[str, xr.DataArray]:
    """Nearest grid cells of a point group, as indexers along a new dimension `p`."""
    lat, lon = ds.latitude.values, ds.longitude.values
    ii = [int(np.abs(lat - a).argmin()) for a, _ in POINTS[group].values()]
    jj = [int(np.abs(lon - b).argmin()) for _, b in POINTS[group].values()]
    return {
        "latitude": xr.DataArray(ii, dims="p"),
        "longitude": xr.DataArray(jj, dims="p"),
    }


def _wind_speed(ds: xr.Dataset, group: str) -> NDArray[np.float64]:
    """Mean 100 m wind speed over the group's cells, per (init, member, lead)."""
    cells = _cells(ds, group)
    u = ds.wind_u_100m.isel(**cells).values.astype(np.float32)
    v = ds.wind_v_100m.isel(**cells).values.astype(np.float32)
    return np.nanmean(np.sqrt(u**2 + v**2), axis=-1)


def _radiation_3h(ds: xr.Dataset, leads: NDArray[np.integer]) -> NDArray[np.float64]:
    """Mean radiation over the solar cells as 3 h means ending at each lead. The source
    averages since the previous 6-hourly step, so at 6-hourly leads R3 = 2 R6 - R3(prev).
    """
    r = ds.downward_short_wave_radiation_flux_surface.isel(**_cells(ds, "solar"))
    r = np.nanmean(r.values.astype(np.float32), axis=-1)  # (init, member, lead)
    r3 = r.copy()
    for k in np.where((leads % 6 == 0) & (leads > 0))[0]:
        if k > 0 and leads[k - 1] == leads[k] - 3:
            r3[:, :, k] = 2 * r[:, :, k] - r[:, :, k - 1]
    return np.clip(r3, 0, None)


def month_table(store: Path) -> pd.DataFrame:
    ds = gefs(store)
    leads = (ds.lead_time.values / np.timedelta64(1, "h")).astype(int)
    inits = pd.DatetimeIndex(ds.init_time.values).tz_localize("UTC")
    members = {  # (init, member, lead) arrays reduced over members below
        "solar_rad": _radiation_3h(ds, leads),
        "offshore_ws": _wind_speed(ds, "offshore"),
        "onshore_ws": _wind_speed(ds, "onshore"),
    }
    frames = []
    for i, init in enumerate(inits):
        cols = {
            "init_time": init,
            "lead_h": leads,
            "valid_time": init + pd.to_timedelta(leads, "h"),
        }
        for name, arr in members.items():
            cols[f"{name}_mean"] = np.nanmean(arr[i], axis=0)
            cols[f"{name}_std"] = np.nanstd(arr[i], axis=0)
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


def gefs_3h(box: str = "nl") -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    parts = []
    for store in gefs_stores(box):
        cached = CACHE / f"{store.stem}.parquet"
        if not cached.exists() or cached.stat().st_mtime < store.stat().st_mtime:
            month_table(store).to_parquet(cached, index=False)
        parts.append(pd.read_parquet(cached))
    return pd.concat(parts, ignore_index=True)


def to_hours(table: pd.DataFrame, column: str, n_hours: int) -> NDArray[np.float64]:
    """Linear interpolation of a 3 h column to hour centres h + 0.5, for every issue at
    once. `table` has issue_time, t_rel (hours since issue) and the column; the result is
    in (issue, h) order."""
    wide = table.pivot(index="t_rel", columns="issue_time", values=column)
    mid = pd.Index(np.arange(n_hours) + 0.5)
    hourly = (
        wide.reindex(wide.index.union(mid)).interpolate(method="index").reindex(mid)
    )
    return hourly.T.to_numpy().ravel()


def gefs_neighbor_wind(groups: list[str], n_hours: int = 168) -> pd.DataFrame:
    """Wind speed mean and spread over the members for point groups outside the Dutch box,
    per (issue at 06 UTC of the run's day, hour), interpolated to hour centres. The 3 h table
    is cached in data/derived/gefs_neighbors_3h.parquet."""
    cached = DERIVED / "gefs_neighbors_3h.parquet"
    if not cached.exists():
        parts = []
        for store in gefs_stores("wind"):
            ds = gefs(store)
            leads = (ds.lead_time.values / np.timedelta64(1, "h")).astype(int)
            inits = pd.DatetimeIndex(ds.init_time.values).tz_localize("UTC")
            speed = {g: _wind_speed(ds, g) for g in groups}
            parts.append(
                pd.DataFrame(
                    {
                        "issue_time": (inits + pd.Timedelta(hours=6)).repeat(
                            len(leads)
                        ),
                        "t_rel": np.tile(leads - 6, len(inits)),
                        **{
                            f"{g}_ws_mean": np.nanmean(a, axis=1).ravel()
                            for g, a in speed.items()
                        },
                        **{
                            f"{g}_ws_std": np.nanstd(a, axis=1).ravel()
                            for g, a in speed.items()
                        },
                    }
                )
            )
        pd.concat(parts, ignore_index=True).to_parquet(cached, index=False)
    table = pd.read_parquet(cached)
    issues = np.sort(table.issue_time.unique())
    out = pd.DataFrame(
        {
            "issue_time": np.repeat(issues, n_hours),
            "h": np.tile(np.arange(n_hours), len(issues)),
        }
    )
    for c in table.columns.drop(["issue_time", "t_rel"]):
        out[c] = to_hours(table, c, n_hours)
    return out
