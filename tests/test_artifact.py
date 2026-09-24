"""Round-trip tests for saving and loading a model artifact."""

import numpy as np
import pandas as pd
import pytest

from elec_forecast.artifact import ModelMetadata, load_model, save_model
from elec_forecast.features import FEATURE_COLUMNS, TARGET_COL, build_features
from elec_forecast.models import make_model


@pytest.fixture
def fitted_model_and_data():
    rng = np.random.default_rng(0)
    hourly = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=1000, freq="h"),
        TARGET_COL: rng.uniform(0.2, 4.0, 1000),
    })
    features = build_features(hourly)
    model = make_model("lightgbm", {"n_estimators": 20, "random_state": 0})
    model.fit(features[FEATURE_COLUMNS], features[TARGET_COL])
    return model, features


def test_loaded_model_predicts_identically_to_the_original(tmp_path, fitted_model_and_data):
    model, features = fitted_model_and_data
    metadata = ModelMetadata(
        model_name="lightgbm",
        feature_columns=list(FEATURE_COLUMNS),
        target_col=TARGET_COL,
        horizon_hours=24,
        trained_on="test fixture",
    )
    save_model(model, metadata, tmp_path / "model")
    loaded_model, _ = load_model(tmp_path / "model")

    original_pred = model.predict(features[FEATURE_COLUMNS])
    loaded_pred = loaded_model.predict(features[FEATURE_COLUMNS])
    np.testing.assert_array_equal(original_pred, loaded_pred)


def test_loaded_metadata_matches_what_was_saved(tmp_path, fitted_model_and_data):
    model, _ = fitted_model_and_data
    metadata = ModelMetadata(
        model_name="lightgbm",
        feature_columns=list(FEATURE_COLUMNS),
        target_col=TARGET_COL,
        horizon_hours=24,
        trained_on="test fixture",
    )
    save_model(model, metadata, tmp_path / "model")
    _, loaded_metadata = load_model(tmp_path / "model")
    assert loaded_metadata == metadata


def test_save_model_creates_the_output_directory(tmp_path, fitted_model_and_data):
    model, _ = fitted_model_and_data
    out_dir = tmp_path / "nested" / "model_dir"
    metadata = ModelMetadata("lightgbm", list(FEATURE_COLUMNS), TARGET_COL, 24, "x")
    save_model(model, metadata, out_dir)
    assert (out_dir / "model.joblib").exists()
    assert (out_dir / "metadata.json").exists()
