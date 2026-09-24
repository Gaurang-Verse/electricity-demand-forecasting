"""Unit tests for the basic correctness of engineered features (not the
leakage property itself -- see test_features_leakage.py for that)."""

import numpy as np
import pandas as pd

from elec_forecast.features import FEATURE_COLUMNS, TARGET_COL, build_features


def _make_hourly(n_hours: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    return pd.DataFrame({"datetime": idx, TARGET_COL: rng.uniform(0.1, 5.0, n_hours)})


def test_build_features_returns_expected_columns():
    hourly = _make_hourly(300)
    out = build_features(hourly)
    for col in [TARGET_COL, *FEATURE_COLUMNS]:
        assert col in out.columns


def test_build_features_drops_rows_without_enough_history():
    hourly = _make_hourly(300)
    out = build_features(hourly)
    # The longest lookback is 168h (lag) and 168h (rolling), both of which
    # need 168 prior hours to exist, so the first 168 rows can't have a
    # complete feature row.
    assert out["datetime"].min() == hourly["datetime"].iloc[168]


def test_lag_feature_matches_the_actual_value_that_many_hours_earlier():
    hourly = _make_hourly(300)
    out = build_features(hourly).set_index("datetime")
    hourly_indexed = hourly.set_index("datetime")[TARGET_COL]

    sample_ts = out.index[10]
    expected_lag_24h = hourly_indexed.loc[sample_ts - pd.Timedelta(hours=24)]
    assert out.loc[sample_ts, "lag_24h"] == expected_lag_24h


def test_rolling_mean_excludes_the_current_hour():
    hourly = _make_hourly(300)
    out = build_features(hourly).set_index("datetime")
    hourly_indexed = hourly.set_index("datetime")[TARGET_COL]

    sample_ts = out.index[20]
    window_start = sample_ts - pd.Timedelta(hours=24)
    window_end = sample_ts - pd.Timedelta(hours=1)
    window = hourly_indexed.loc[window_start:window_end]
    assert len(window) == 24
    assert np.isclose(out.loc[sample_ts, "roll_mean_24h"], window.mean())


def test_calendar_features_match_the_timestamp():
    hourly = _make_hourly(300)
    out = build_features(hourly).set_index("datetime")
    sample_ts = out.index[5]
    assert out.loc[sample_ts, "hour"] == sample_ts.hour
    assert out.loc[sample_ts, "dayofweek"] == sample_ts.dayofweek
    assert out.loc[sample_ts, "is_weekend"] == int(sample_ts.dayofweek >= 5)
