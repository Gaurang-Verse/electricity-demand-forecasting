"""Unit tests for the model factory."""

import numpy as np
import pandas as pd
import pytest

from elec_forecast.features import FEATURE_COLUMNS, TARGET_COL, build_features
from elec_forecast.models import make_model


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 1000
    hourly = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=n, freq="h"),
        TARGET_COL: rng.uniform(0.2, 4.0, n),
    })
    return build_features(hourly)


def test_seasonal_naive_predicts_last_weeks_value(frame):
    model = make_model("seasonal_naive").fit(frame[FEATURE_COLUMNS], frame[TARGET_COL])
    np.testing.assert_array_equal(model.predict(frame[FEATURE_COLUMNS]), frame["lag_168h"])


@pytest.mark.parametrize("name", ["linear", "lightgbm"])
def test_learned_models_fit_and_predict_one_value_per_row(frame, name):
    params = {"n_estimators": 20, "random_state": 0} if name == "lightgbm" else None
    model = make_model(name, params)
    model.fit(frame[FEATURE_COLUMNS], frame[TARGET_COL])
    pred = model.predict(frame[FEATURE_COLUMNS])
    assert pred.shape == (len(frame),)
    assert np.isfinite(pred).all()


def test_lightgbm_is_reproducible_with_a_fixed_seed(frame):
    X, y = frame[FEATURE_COLUMNS], frame[TARGET_COL]
    params = {"n_estimators": 50, "random_state": 42}
    a = make_model("lightgbm", params).fit(X, y).predict(X)
    b = make_model("lightgbm", params).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


def test_make_model_returns_a_fresh_instance_each_call():
    assert make_model("linear") is not make_model("linear")


def test_make_model_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown model"):
        make_model("prophet")
