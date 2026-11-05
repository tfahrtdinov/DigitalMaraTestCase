import numpy as np
import pandas as pd

from data.features import co2


def utc(s):
    return pd.Timestamp(s, tz="UTC")


def announcement(revision, created, start, end, status="A05"):
    return {
        "mrid": "a",
        "revision": revision,
        "created_utc": utc(created),
        "doc_status": status,
        "fuel": "gas",
        "nominal_mw": 100.0,
        "available_mw": 40.0,
        "avail_start_utc": utc(start),
        "avail_end_utc": utc(end),
    }


def run(rows, monkeypatch):
    frame = pd.DataFrame(rows)
    empty = frame.iloc[:0]
    monkeypatch.setattr(co2, "outages", lambda d: frame if d == "A80" else empty)
    table = co2.announced_outages(pd.DatetimeIndex([utc("2025-01-02 06:00")]))
    return table["out_gas"].to_numpy()


def test_outage_uses_the_latest_version_known_at_issue(monkeypatch):
    rows = [
        announcement(1, "2025-01-01", "2025-01-02 08:00", "2025-01-02 10:00"),
        announcement(2, "2025-01-02 07:00", "2025-01-03", "2025-01-04"),
    ]
    out = run(rows, monkeypatch)
    assert out[2:4].tolist() == [60.0, 60.0]
    assert out.sum() == 120.0


def test_cancelled_outage_is_dropped(monkeypatch):
    rows = [
        announcement(1, "2025-01-01", "2025-01-02 08:00", "2025-01-02 10:00"),
        announcement(
            2, "2025-01-01 12:00", "2025-01-02 08:00", "2025-01-02 10:00", "A09"
        ),
    ]
    assert np.all(run(rows, monkeypatch) == 0)


def test_profile_uses_only_the_same_hour_up_to_each_timestamp():
    index = pd.date_range("2025-01-01", periods=24 * 20, freq="h", tz="UTC")
    s = pd.Series(np.arange(len(index), dtype=float), index=index)
    profile = co2.hourly_profile(s)
    t = index[-1]
    same_hour = s.to_numpy()[index.hour == t.hour]
    assert profile.iloc[-1] == same_hour.mean()
