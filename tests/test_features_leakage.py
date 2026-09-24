"""Leakage tests for build_features.

The forecasting task is: at a forecast origin T (the first hour being
predicted), using only data observed before T, predict hours T .. T+23.
So the property that matters is not just "the feature row for hour t
doesn't use hour t", but "every feature row inside the forecast window
[T, T+23] uses only data from before T".

These tests check that property directly instead of asserting it. They
build features on a normal series, then again on a copy where every value
from T onward is replaced with an extreme outlier. Any feature row in the
forecast window that changes has read data that wouldn't exist yet at
forecast time.

The detector is also run against a deliberately leaky builder: rolling
windows shifted by only 1 hour, which is the version this project
originally shipped. That shows the test can catch the leak, and isn't
just passing because nothing changes.
"""

import numpy as np
import pandas as pd

from elec_forecast.features import (
    FEATURE_COLUMNS,
    HORIZON_HOURS,
    LAG_HOURS,
    ROLLING_WINDOWS,
    TARGET_COL,
    build_features,
)

N_HOURS = 600
ORIGIN_IDX = 400  # comfortably past the longest lookback, comfortably before the end
OUTLIER = 999_999.0


def _make_hourly(n_hours: int = N_HOURS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-06-01", periods=n_hours, freq="h")
    return pd.DataFrame({"datetime": idx, TARGET_COL: rng.uniform(0.1, 5.0, n_hours)})


def _leaky_build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """The original (buggy) feature builder: rolling windows shifted by 1h.

    Kept here only to prove the detector below catches this class of bug.
    """
    df = hourly.sort_values("datetime").set_index("datetime").copy()
    for h in LAG_HOURS:
        df[f"lag_{h}h"] = df[TARGET_COL].shift(h)
    shifted = df[TARGET_COL].shift(1)
    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}h"] = shifted.rolling(window=w).mean()
        df[f"roll_std_{w}h"] = shifted.rolling(window=w).std()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df.dropna().reset_index()


def _leaking_columns(builder, horizon: int = HORIZON_HOURS) -> list[str]:
    """Return feature columns whose values in [origin, origin + horizon)
    change when data from the origin onward is corrupted."""
    hourly = _make_hourly()
    origin = hourly["datetime"].iloc[ORIGIN_IDX]

    corrupted = hourly.copy()
    corrupted.loc[corrupted["datetime"] >= origin, TARGET_COL] = OUTLIER

    clean = builder(hourly).set_index("datetime")
    dirty = builder(corrupted).set_index("datetime")

    window = pd.date_range(origin, periods=horizon, freq="h")
    assert window.isin(clean.index).all(), "test setup: forecast window missing rows"

    leaking = []
    for col in FEATURE_COLUMNS:
        if not np.allclose(clean.loc[window, col], dirty.loc[window, col]):
            leaking.append(col)
    return leaking


def test_no_feature_in_the_forecast_window_uses_data_from_the_origin_onward():
    assert _leaking_columns(build_features) == []


def test_detector_catches_the_original_one_hour_shifted_rolling_features():
    """If this ever fails, the detector has stopped working, and the test
    above passing would mean nothing."""
    leaking = _leaking_columns(_leaky_build_features)
    assert set(leaking) == {f"roll_{s}_{w}h" for s in ("mean", "std") for w in ROLLING_WINDOWS}


def test_features_before_the_origin_are_unaffected_by_corrupting_the_future():
    hourly = _make_hourly()
    origin = hourly["datetime"].iloc[ORIGIN_IDX]

    corrupted = hourly.copy()
    corrupted.loc[corrupted["datetime"] >= origin, TARGET_COL] = OUTLIER

    clean = build_features(hourly).set_index("datetime")
    dirty = build_features(corrupted).set_index("datetime")
    before = clean.index[clean.index < origin]

    pd.testing.assert_frame_equal(clean.loc[before], dirty.loc[before])


def test_sanity_corrupting_the_past_does_change_later_features():
    """Rules out the tests above passing because build_features ignores its input."""
    hourly = _make_hourly()
    origin = hourly["datetime"].iloc[ORIGIN_IDX]

    corrupted = hourly.copy()
    corrupted.loc[corrupted["datetime"] < origin, TARGET_COL] = OUTLIER

    clean = build_features(hourly).set_index("datetime")
    dirty = build_features(corrupted).set_index("datetime")

    assert clean.loc[origin, "lag_168h"] != dirty.loc[origin, "lag_168h"]
