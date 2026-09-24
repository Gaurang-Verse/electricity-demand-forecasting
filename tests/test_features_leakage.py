"""Leakage test for build_features.

This doesn't just assert "no leakage" in a docstring -- it actively tries
to catch a leak. The method: build features once on a normal series, then
build features again on a version of the SAME series where every value
strictly after a chosen cutoff hour has been replaced with an extreme,
unmistakable outlier. If any feature at or before the cutoff used
information from after the cutoff, it would change when that future data
changes. It must not.

A sanity check runs alongside it: corrupting data BEFORE the cutoff DOES
change features after the cutoff (lag/rolling features are supposed to
depend on the past). This rules out the test passing merely because
build_features ignores its input.
"""

import numpy as np
import pandas as pd

from elec_forecast.features import FEATURE_COLUMNS, TARGET_COL, build_features

N_HOURS = 400
CUTOFF_IDX = 250  # comfortably past the 168h lookback, comfortably before the end
OUTLIER = 999_999.0


def _make_hourly(n_hours: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-06-01", periods=n_hours, freq="h")
    return pd.DataFrame({"datetime": idx, TARGET_COL: rng.uniform(0.1, 5.0, n_hours)})


def test_features_at_or_before_cutoff_are_unaffected_by_corrupting_the_future():
    hourly = _make_hourly(N_HOURS)
    cutoff_ts = hourly["datetime"].iloc[CUTOFF_IDX]

    corrupted = hourly.copy()
    future_mask = corrupted["datetime"] > cutoff_ts
    assert future_mask.sum() > 0
    corrupted.loc[future_mask, TARGET_COL] = OUTLIER

    original_features = build_features(hourly).set_index("datetime")
    corrupted_features = build_features(corrupted).set_index("datetime")

    shared_ts = original_features.index[original_features.index <= cutoff_ts]
    assert len(shared_ts) > 0, "test setup produced no rows to compare -- fix CUTOFF_IDX"

    for col in [TARGET_COL, *FEATURE_COLUMNS]:
        pd.testing.assert_series_equal(
            original_features.loc[shared_ts, col],
            corrupted_features.loc[shared_ts, col],
            check_names=False,
            obj=f"column '{col}' changed when future data was corrupted -- this is a leak",
        )


def test_sanity_corrupting_the_past_does_change_later_features():
    """Confirms the test above is actually exercising the lag/rolling logic,
    not just comparing two runs that happen to be identical for any input."""
    hourly = _make_hourly(N_HOURS)
    cutoff_ts = hourly["datetime"].iloc[CUTOFF_IDX]

    corrupted = hourly.copy()
    past_mask = corrupted["datetime"] <= cutoff_ts
    corrupted.loc[past_mask, TARGET_COL] = OUTLIER

    original_features = build_features(hourly).set_index("datetime")
    corrupted_features = build_features(corrupted).set_index("datetime")

    # A lag_168h feature well after the cutoff should now differ, since its
    # source value was corrupted.
    check_ts = original_features.index[original_features.index > cutoff_ts][0]
    assert (
        original_features.loc[check_ts, "lag_168h"]
        != corrupted_features.loc[check_ts, "lag_168h"]
    )
