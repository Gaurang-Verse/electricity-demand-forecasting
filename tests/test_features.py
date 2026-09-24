"""Basic correctness tests for engineered features.

The leakage property itself is tested in test_features_leakage.py.
"""

import numpy as np
import pandas as pd
import pytest

from elec_forecast.features import (
    FEATURE_COLUMNS,
    HORIZON_HOURS,
    LAG_HOURS,
    ROLLING_WINDOWS,
    TARGET_COL,
    build_features,
)


def _make_hourly(n_hours: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    return pd.DataFrame({"datetime": idx, TARGET_COL: rng.uniform(0.1, 5.0, n_hours)})


def test_build_features_returns_expected_columns():
    out = build_features(_make_hourly(400))
    assert list(out.columns) == ["datetime", TARGET_COL, *FEATURE_COLUMNS]


def test_build_features_drops_exactly_the_rows_without_enough_history():
    hourly = _make_hourly(400)
    out = build_features(hourly)
    # A rolling window of w over a series shifted by the horizon first has
    # a full window at position horizon + w - 1.
    first_valid = max(max(LAG_HOURS), HORIZON_HOURS + max(ROLLING_WINDOWS) - 1)
    assert out["datetime"].iloc[0] == hourly["datetime"].iloc[first_valid]
    assert len(out) == len(hourly) - first_valid


def test_lag_feature_matches_the_value_that_many_hours_earlier():
    hourly = _make_hourly(400)
    out = build_features(hourly).set_index("datetime")
    series = hourly.set_index("datetime")[TARGET_COL]

    ts = out.index[10]
    for h in LAG_HOURS:
        assert out.loc[ts, f"lag_{h}h"] == series.loc[ts - pd.Timedelta(hours=h)]


def test_rolling_window_ends_a_full_horizon_before_the_row():
    hourly = _make_hourly(400)
    out = build_features(hourly).set_index("datetime")
    series = hourly.set_index("datetime")[TARGET_COL]

    ts = out.index[20]
    window_end = ts - pd.Timedelta(hours=HORIZON_HOURS)
    window_start = window_end - pd.Timedelta(hours=23)
    window = series.loc[window_start:window_end]
    assert len(window) == 24
    assert np.isclose(out.loc[ts, "roll_mean_24h"], window.mean())
    assert np.isclose(out.loc[ts, "roll_std_24h"], window.std())


def test_calendar_features_match_the_timestamp():
    out = build_features(_make_hourly(400)).set_index("datetime")
    ts = out.index[5]
    assert out.loc[ts, "hour"] == ts.hour
    assert out.loc[ts, "dayofweek"] == ts.dayofweek
    assert out.loc[ts, "month"] == ts.month
    assert out.loc[ts, "is_weekend"] == int(ts.dayofweek >= 5)


def test_build_features_rejects_a_series_with_a_missing_hour():
    hourly = _make_hourly(400).drop(index=250)
    with pytest.raises(ValueError, match="not contiguous"):
        build_features(hourly)


def test_build_features_rejects_a_horizon_longer_than_the_shortest_lag():
    with pytest.raises(ValueError, match="would leak"):
        build_features(_make_hourly(400), horizon=min(LAG_HOURS) + 1)
