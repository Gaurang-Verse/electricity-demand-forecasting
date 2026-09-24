"""Unit tests for forecast error metrics, checked against hand-computed values."""

import math

import pytest

from elec_forecast.metrics import regression_metrics


def test_metrics_match_hand_computed_values():
    m = regression_metrics(actual=[1.0, 2.0, 4.0], predicted=[2.0, 2.0, 2.0])
    # errors (pred - actual): +1, 0, -2
    assert m["rmse"] == pytest.approx(math.sqrt((1 + 0 + 4) / 3))
    assert m["mae"] == pytest.approx(1.0)
    assert m["mape"] == pytest.approx((1 / 1 + 0 / 2 + 2 / 4) / 3 * 100)
    assert m["bias"] == pytest.approx(-1 / 3)
    assert m["n"] == 3


def test_perfect_forecast_has_zero_error():
    m = regression_metrics([1.5, 2.5], [1.5, 2.5])
    assert m["rmse"] == m["mae"] == m["mape"] == m["bias"] == 0


def test_positive_bias_means_over_forecasting():
    assert regression_metrics([1.0, 1.0], [1.5, 1.5])["bias"] > 0


def test_mape_refuses_non_positive_actuals():
    with pytest.raises(ValueError, match="MAPE"):
        regression_metrics([0.0, 1.0], [0.5, 1.0])


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        regression_metrics([1.0, 2.0], [1.0])
