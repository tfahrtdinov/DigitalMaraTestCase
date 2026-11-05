"""Readers in src/data/loader.py against small synthetic files. `raw` points data.RAW at a temporary folder, so every test
reads from disk: a changed file must change the result (no caching)."""

from pathlib import Path

import pandas as pd
import pytest

import data.loader as data


def write_ned(
    folder: Path, name: str, start: str, hours: int, volume: list[int]
) -> None:
    t = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    pd.DataFrame(
        {
            "validfrom": t.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "validto": (t + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "volume": volume,
            "emissionfactor": [0.3] * hours,
        }
    ).to_csv(folder / name, index=False)


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "RAW", tmp_path)
    (tmp_path / "openmeteo").mkdir()
    return tmp_path


def test_ned_concatenates_and_later_file_wins(raw):
    write_ned(
        raw, "solar_cls2_2024-01-01_2024-01-01.csv", "2024-01-01", 4, [1, 2, 3, 4]
    )
    write_ned(
        raw,
        "solar_cls2_2024-01-01_2024-01-02.csv",
        "2024-01-01 02:00",
        4,
        [30, 40, 5, 6],
    )
    s = data.ned("solar")
    assert isinstance(s.index, pd.DatetimeIndex) and str(s.index.tz) == "UTC"
    assert s.index.is_unique and s.index.is_monotonic_increasing
    assert s.tolist() == [1, 2, 30, 40, 5, 6]
    assert s.name == "solar" and s.index.name == "time"


def test_ned_column_and_missing_pattern(raw):
    write_ned(
        raw, "electricity_mix_cls2_2024-01-01_2024-01-01.csv", "2024-01-01", 2, [1, 1]
    )
    assert data.ned("electricity_mix", "emissionfactor").tolist() == [0.3, 0.3]
    with pytest.raises(FileNotFoundError):
        data.ned("wind_onshore")


def test_reads_disk_every_call(raw):
    write_ned(raw, "solar_cls2_a.csv", "2024-01-01", 2, [1, 2])
    assert data.ned("solar").tolist() == [1, 2]
    write_ned(raw, "solar_cls2_a.csv", "2024-01-01", 2, [7, 8])
    assert data.ned("solar").tolist() == [7, 8]


def test_entsoe_quarter_hours_are_averaged(raw):
    t = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
    pd.DataFrame({"time_utc": t, "value_mw": [1, 2, 3, 4, 10, 10, 10, 10]}).to_csv(
        raw / "entsoe_actual_solar_x.csv", index=False
    )
    s = data.entsoe("actual_solar")
    assert s.tolist() == [2.5, 10.0]
    assert isinstance(s.index, pd.DatetimeIndex) and s.index.freq == "h"


def test_openmeteo_long_table(raw):
    t = pd.date_range("2024-04-01", periods=2, freq="h", tz="UTC")
    pd.DataFrame(
        {
            "time": t,
            "point": "utrecht",
            "var": "shortwave_radiation",
            "lead": 1,
            "value": [0.0, 5.0],
        }
    ).to_csv(raw / "openmeteo" / "solar_x.csv", index=False)
    df = data.openmeteo("solar")
    assert (
        list(df.columns) == ["time", "point", "lead", "value"]
        and str(df.time.dt.tz) == "UTC"
    )


def test_gefs_stores_sorted_by_month(raw):
    for m in ["2021-02", "2020-10", "2020-11"]:
        (raw / "gefs" / f"gefs_nl_{m}.zarr").mkdir(parents=True)
    assert [p.name for p in data.gefs_stores("nl")] == [
        "gefs_nl_2020-10.zarr",
        "gefs_nl_2020-11.zarr",
        "gefs_nl_2021-02.zarr",
    ]


def test_outages_dates_are_utc(raw):
    cols = [
        "created_utc",
        "outage_start_utc",
        "outage_end_utc",
        "avail_start_utc",
        "avail_end_utc",
    ]
    row = {c: "2024-01-01 00:00:00+00:00" for c in cols} | {
        "unit_name": "Amer 9",
        "available_mw": 531.0,
    }
    pd.DataFrame([row]).to_csv(raw / "entsoe_outages_A80_x.csv", index=False)
    df = data.outages()
    assert (
        all(str(df[c].dt.tz) == "UTC" for c in cols)
        and df.available_mw.iloc[0] == 531.0
    )
