"""Tests for build_forecast_features, focused on the property that makes
inference possible at all: the forecast window's features must be
identical to what training would have computed, using nothing the
forecast wouldn't actually have available."""

import numpy as np
import pandas as pd
import pytest

from elec_forecast.features import (
    FEATURE_COLUMNS,
    HORIZON_HOURS,
    ROLLING_WINDOWS,
    TARGET_COL,
    build_features,
    build_forecast_features,
)

N_HOURS = 1000
ORIGIN_IDX = 700  # comfortably past the lookback, comfortably before the end


def _make_hourly(n_hours: int = N_HOURS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n_hours, freq="h")
    return pd.DataFrame({"datetime": idx, TARGET_COL: rng.uniform(0.1, 5.0, n_hours)})


def test_forecast_features_match_what_training_would_compute_with_full_hindsight():
    """The whole point of horizon-safety: the forecast for [origin, origin+23]
    depends only on data before origin, so it must be computable two ways --
    from the full series after the fact (build_features), or from history
    alone before the fact (build_forecast_features) -- and get exactly the
    same answer either way."""
    full = _make_hourly()
    origin = full["datetime"].iloc[ORIGIN_IDX]
    history = full[full["datetime"] < origin]

    with_hindsight = build_features(full).set_index("datetime")
    window = pd.date_range(origin, periods=HORIZON_HOURS, freq="h")
    assert window.isin(with_hindsight.index).all()

    forecast = build_forecast_features(history, origin).set_index("datetime")
    pd.testing.assert_frame_equal(
        with_hindsight.loc[window, FEATURE_COLUMNS], forecast.loc[window, FEATURE_COLUMNS]
    )


def test_forecast_features_are_unaffected_by_what_actually_happens_at_the_origin():
    """A second way of stating the same property: swapping in nonsense values
    for the forecast window itself (which build_forecast_features never even
    sees) must not change the computed features."""
    full = _make_hourly()
    origin = full["datetime"].iloc[ORIGIN_IDX]
    history = full[full["datetime"] < origin]

    forecast_a = build_forecast_features(history, origin)

    corrupted = full.copy()
    corrupted.loc[corrupted["datetime"] >= origin, TARGET_COL] = 999_999.0
    corrupted_history = corrupted[corrupted["datetime"] < origin]
    forecast_b = build_forecast_features(corrupted_history, origin)

    pd.testing.assert_frame_equal(forecast_a, forecast_b)


def test_forecast_features_returns_one_row_per_horizon_hour():
    full = _make_hourly()
    origin = full["datetime"].iloc[ORIGIN_IDX]
    history = full[full["datetime"] < origin]

    out = build_forecast_features(history, origin)
    assert list(out["datetime"]) == list(pd.date_range(origin, periods=HORIZON_HOURS, freq="h"))
    assert list(out.columns) == ["datetime", *FEATURE_COLUMNS]
    assert not out[FEATURE_COLUMNS].isna().any().any()


def test_forecast_features_rejects_history_that_does_not_end_right_before_origin():
    full = _make_hourly()
    origin = full["datetime"].iloc[ORIGIN_IDX]
    history = full[full["datetime"] < origin - pd.Timedelta(hours=1)]  # gap before origin

    with pytest.raises(ValueError, match="must end exactly at"):
        build_forecast_features(history, origin)


def test_forecast_features_rejects_insufficient_lookback():
    full = _make_hourly()
    origin = full["datetime"].iloc[ORIGIN_IDX]
    lookback_needed = HORIZON_HOURS + max(ROLLING_WINDOWS) - 1
    earliest = origin - pd.Timedelta(hours=lookback_needed - 10)
    too_short_history = full[(full["datetime"] < origin) & (full["datetime"] >= earliest)]

    with pytest.raises(ValueError, match="not enough history"):
        build_forecast_features(too_short_history, origin)


def test_forecast_features_rejects_a_gap_in_history():
    full = _make_hourly()
    origin = full["datetime"].iloc[ORIGIN_IDX]
    history = full[full["datetime"] < origin].drop(index=ORIGIN_IDX - 50)

    with pytest.raises(ValueError, match="not contiguous"):
        build_forecast_features(history, origin)
