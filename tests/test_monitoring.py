"""Tests for drift detection and rolling forecast-error tracking.

The drift tests are validated two ways, deliberately: a null test proving
the detector stays quiet on data drawn from the same distribution as the
reference, and a positive control proving it actually fires when the
distribution really has shifted.
"""

import numpy as np
import pandas as pd
import pytest

from elec_forecast.monitoring import (
    compute_reference_stats,
    drift_report,
    rolling_forecast_error,
)

COLUMNS = ["a", "b"]


def _make_reference(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(10, 2, n), "b": rng.normal(-5, 1, n)})


def test_compute_reference_stats_matches_hand_computed_mean_and_std():
    reference = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    stats = compute_reference_stats(reference, ["a"])
    assert stats.means["a"] == pytest.approx(3.0)
    assert stats.stds["a"] == pytest.approx(reference["a"].std())


def test_compute_reference_stats_rejects_zero_variance_columns():
    reference = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
    with pytest.raises(ValueError, match="zero variance"):
        compute_reference_stats(reference, ["a"])


def test_drift_report_null_stays_quiet_on_a_fresh_sample_from_the_same_distribution():
    """The null test: a second batch from the *same* generating
    distribution as the reference should not be flagged, at the default
    threshold, with a fixed seed so this isn't flaky."""
    reference = _make_reference(seed=0)
    stats = compute_reference_stats(reference, COLUMNS)

    fresh_sample = _make_reference(n=500, seed=1)
    report = drift_report(fresh_sample, stats)

    assert report["any_drifted"] is False
    for feature in COLUMNS:
        assert report["per_feature"][feature]["drifted"] is False


def test_drift_report_positive_control_fires_on_a_genuinely_shifted_batch():
    """The positive control: without this, a detector that always says
    'no drift' would pass the null test above trivially."""
    reference = _make_reference(seed=0)
    stats = compute_reference_stats(reference, COLUMNS)

    rng = np.random.default_rng(2)
    shifted = pd.DataFrame(
        {
            "a": rng.normal(10 + 8, 2, 500),  # 4 reference std devs away
            "b": rng.normal(-5, 1, 500),  # left unshifted, for contrast
        }
    )
    report = drift_report(shifted, stats)

    assert report["any_drifted"] is True
    assert report["per_feature"]["a"]["drifted"] is True
    assert report["per_feature"]["b"]["drifted"] is False


def test_drift_report_rejects_an_empty_current_batch():
    reference = _make_reference()
    stats = compute_reference_stats(reference, COLUMNS)
    with pytest.raises(ValueError, match="empty"):
        drift_report(pd.DataFrame({"a": [], "b": []}), stats)


def test_rolling_forecast_error_matches_hand_computed_values_on_a_simple_case():
    predictions = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=4, freq="h"),
            "actual": [10.0, 10.0, 10.0, 10.0],
            "predicted": [10.0, 12.0, 10.0, 8.0],
        }
    )
    result = rolling_forecast_error(predictions, window=2)

    # errors are [0, -2, 0, 2]; a window of 2 means the first valid value
    # is at index 1.
    assert pd.isna(result["rolling_mae"].iloc[0])
    assert result["rolling_mae"].iloc[1] == pytest.approx(1.0)  # mean(|0|, |-2|)
    assert result["rolling_rmse"].iloc[1] == pytest.approx(np.sqrt((0**2 + 2**2) / 2))
    assert result["rolling_mae"].iloc[3] == pytest.approx(1.0)  # mean(|0|, |2|)


def test_rolling_forecast_error_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        rolling_forecast_error(pd.DataFrame({"datetime": [], "actual": []}))


def test_rolling_forecast_error_output_is_chronological_regardless_of_input_order():
    shuffled = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=3, freq="h")[::-1],
            "actual": [10.0, 10.0, 10.0],
            "predicted": [10.0, 10.0, 10.0],
        }
    )
    result = rolling_forecast_error(shuffled, window=2)
    assert list(result["datetime"]) == sorted(result["datetime"])