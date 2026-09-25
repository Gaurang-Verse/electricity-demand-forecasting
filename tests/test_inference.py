"""Tests for the Predictor class."""

import numpy as np
import pandas as pd
import pytest

from elec_forecast.artifact import ModelMetadata, save_model
from elec_forecast.features import FEATURE_COLUMNS, HORIZON_HOURS, TARGET_COL, build_features
from elec_forecast.inference import Predictor
from elec_forecast.models import make_model


def _make_hourly(n_hours: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n_hours, freq="h")
    return pd.DataFrame({"datetime": idx, TARGET_COL: rng.uniform(0.1, 5.0, n_hours)})


def _save_fitted_model(tmp_path, metadata_overrides=None):
    hourly = _make_hourly()
    features = build_features(hourly)
    model = make_model("lightgbm", {"n_estimators": 20, "random_state": 0})
    model.fit(features[FEATURE_COLUMNS], features[TARGET_COL])

    metadata = ModelMetadata(
        model_name="lightgbm",
        feature_columns=list(FEATURE_COLUMNS),
        target_col=TARGET_COL,
        horizon_hours=HORIZON_HOURS,
        trained_on="test fixture",
    )
    if metadata_overrides:
        metadata = ModelMetadata(**{**vars(metadata), **metadata_overrides})

    model_dir = tmp_path / "model"
    save_model(model, metadata, model_dir)
    return model_dir, hourly, model


def test_predict_returns_one_row_per_horizon_hour(tmp_path):
    model_dir, hourly, _ = _save_fitted_model(tmp_path)
    origin = hourly["datetime"].iloc[700]
    history = hourly[hourly["datetime"] < origin]

    predictor = Predictor(str(model_dir))
    forecast = predictor.predict(history, origin=origin)

    assert list(forecast.columns) == ["datetime", "predicted_kw"]
    assert len(forecast) == HORIZON_HOURS
    expected_index = pd.date_range(origin, periods=HORIZON_HOURS, freq="h")
    assert list(forecast["datetime"]) == list(expected_index)
    assert np.isfinite(forecast["predicted_kw"]).all()


def test_predict_matches_calling_the_model_directly(tmp_path):
    """No hidden transformation between Predictor.predict and the model:
    same features in, same predictions out."""
    model_dir, hourly, model = _save_fitted_model(tmp_path)
    origin = hourly["datetime"].iloc[700]
    history = hourly[hourly["datetime"] < origin]

    from elec_forecast.features import build_forecast_features
    expected_features = build_forecast_features(history, origin)
    expected = model.predict(expected_features[FEATURE_COLUMNS])

    predictor = Predictor(str(model_dir))
    actual = predictor.predict(history, origin=origin)["predicted_kw"].to_numpy()
    np.testing.assert_array_equal(actual, expected)


def test_predict_defaults_origin_to_right_after_the_history(tmp_path):
    model_dir, hourly, _ = _save_fitted_model(tmp_path)
    cutoff = hourly["datetime"].iloc[700]
    history = hourly[hourly["datetime"] < cutoff]

    predictor = Predictor(str(model_dir))
    forecast = predictor.predict(history)
    assert forecast["datetime"].iloc[0] == cutoff


def test_predictor_rejects_a_model_trained_on_different_features(tmp_path):
    model_dir, _, _ = _save_fitted_model(
        tmp_path, metadata_overrides={"feature_columns": ["lag_24h"]}
    )
    with pytest.raises(ValueError, match="drifted apart"):
        Predictor(str(model_dir))


def test_predictor_rejects_a_model_trained_for_a_different_horizon(tmp_path):
    model_dir, _, _ = _save_fitted_model(tmp_path, metadata_overrides={"horizon_hours": 1})
    with pytest.raises(ValueError, match="1h horizon"):
        Predictor(str(model_dir))


def test_predictor_rejects_a_model_trained_on_a_different_target(tmp_path):
    model_dir, _, _ = _save_fitted_model(tmp_path, metadata_overrides={"target_col": "other"})
    with pytest.raises(ValueError, match="target column"):
        Predictor(str(model_dir))
