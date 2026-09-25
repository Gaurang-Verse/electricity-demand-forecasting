"""End-to-end integration test: raw hourly data -> features -> walk-forward
backtest -> final model -> saved artifact -> Predictor -> live FastAPI
request, all through the real functions each script actually calls.

Every other test file exercises one module in isolation with hand-built
inputs. This one's only job is to catch what those miss: two
independently-correct pieces that don't actually fit together when wired
up for real -- a config key one function expects that another doesn't
provide, a DataFrame shape one stage produces that the next doesn't
accept, tests that mock past exactly the seam that's broken.

It uses a small, fast backtest config (not the real configs/backtest.yaml,
which needs a full year of data) so the whole file runs in a couple of
seconds. That's a deliberate trade: this proves the pipeline's plumbing
is correct, not that the real numbers in docs/backtest_results.md and
docs/holdout_evaluation.md are reproducible from here -- those come from
scripts/run_backtest.py and scripts/evaluate_holdout.py against the real
dataset, and stay the authority for measured results.
"""

import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from elec_forecast.api import app
from elec_forecast.artifact import ModelMetadata, load_model, save_model
from elec_forecast.backtest import run_backtest, summarize
from elec_forecast.features import FEATURE_COLUMNS, HORIZON_HOURS, TARGET_COL
from elec_forecast.inference import Predictor
from elec_forecast.metrics import regression_metrics
from elec_forecast.models import MODEL_NAMES, make_model
from elec_forecast.pipeline import prepare_backtest

# Deliberately small and fast -- see module docstring for why this isn't
# configs/backtest.yaml. Verified by hand to satisfy walk_forward_folds's
# min_train_hours requirement for every fold with room to spare.
SMALL_BACKTEST_CONFIG = {
    "holdout_hours": 200,
    "n_folds": 3,
    "fold_test_hours": 100,
    "min_train_hours": 500,
}
N_HOURS = 1500


def _write_synthetic_hourly(path) -> None:
    """Same shape load_hourly() expects from scripts/preprocess.py's real
    output: datetime, target_mean_kw, and n_minutes (used to drop partial
    hours -- none are partial here, which itself exercises that
    load_hourly doesn't drop anything it shouldn't when nothing is
    partial)."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=N_HOURS, freq="h")
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    base = 1.0 + 0.5 * np.sin(2 * np.pi * hour / 24) + 0.2 * (dow >= 5)
    values = np.clip(base + rng.normal(0, 0.1, N_HOURS), 0.05, None)
    hourly = pd.DataFrame({"datetime": idx, TARGET_COL: values, "n_minutes": 60})
    hourly.to_parquet(path)


@pytest.fixture
def backtest_config(tmp_path) -> dict:
    hourly_path = tmp_path / "hourly.parquet"
    _write_synthetic_hourly(hourly_path)
    return {**SMALL_BACKTEST_CONFIG, "hourly_path": str(hourly_path)}


def test_prepare_backtest_produces_data_every_model_can_train_on(backtest_config):
    data = prepare_backtest(backtest_config)

    assert len(data.dev) > 0
    assert len(data.holdout) == SMALL_BACKTEST_CONFIG["holdout_hours"]
    assert len(data.folds) == SMALL_BACKTEST_CONFIG["n_folds"]
    # the contract every downstream stage relies on: dev and holdout never
    # overlap, and dev is entirely before holdout in time.
    assert data.dev["datetime"].max() < data.holdout["datetime"].min()


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_backtests_end_to_end(backtest_config, model_name):
    """Runs the same call sequence scripts/run_backtest.py makes, for
    every registered model, and checks the summary has real numbers in
    it -- not just that it runs without raising."""
    data = prepare_backtest(backtest_config)
    predictions = run_backtest(data, lambda: make_model(model_name))
    summary = summarize(predictions)

    assert summary["overall"]["n"] == len(predictions)
    assert np.isfinite(summary["overall"]["rmse"])
    assert len(summary["per_fold"]) == SMALL_BACKTEST_CONFIG["n_folds"]
    # every scored prediction must come from dev, never holdout -- the same
    # property tests/test_backtest.py checks in isolation, re-checked here
    # against data that went through prepare_backtest for real.
    assert predictions["datetime"].max() < data.holdout["datetime"].min()


def test_full_lifecycle_train_save_load_predict_and_serve(
    tmp_path, backtest_config, monkeypatch
):
    """The path a real deployment takes: prepare data, train the model
    that ships (on all of dev, like evaluate_holdout.py does), save it,
    load it back through Predictor, then confirm a running API returns
    the same forecast Predictor does directly. If any stage's output
    stopped matching the next stage's input, this is where it would
    show up."""
    data = prepare_backtest(backtest_config)

    model = make_model("lightgbm", {"n_estimators": 20, "random_state": 0})
    model.fit(data.dev[FEATURE_COLUMNS], data.dev[TARGET_COL])

    holdout_predicted = model.predict(data.holdout[FEATURE_COLUMNS])
    holdout_metrics = regression_metrics(data.holdout[TARGET_COL], holdout_predicted)
    assert np.isfinite(holdout_metrics["rmse"])

    model_dir = tmp_path / "model"
    save_model(
        model,
        ModelMetadata(
            model_name="lightgbm",
            feature_columns=list(FEATURE_COLUMNS),
            target_col=TARGET_COL,
            horizon_hours=HORIZON_HOURS,
            trained_on=f"integration test, {len(data.dev)} dev hours",
        ),
        model_dir,
    )

    loaded_model, loaded_metadata = load_model(model_dir)
    assert loaded_metadata.feature_columns == list(FEATURE_COLUMNS)
    np.testing.assert_array_equal(
        loaded_model.predict(data.holdout[FEATURE_COLUMNS]), holdout_predicted
    )

    predictor = Predictor(str(model_dir))
    origin = data.holdout["datetime"].iloc[0]
    history = data.features.loc[data.features["datetime"] < origin, ["datetime", TARGET_COL]]
    direct_forecast = predictor.predict(history, origin=origin)
    assert len(direct_forecast) == HORIZON_HOURS
    assert np.isfinite(direct_forecast["predicted_kw"]).all()

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.json() == {"status": "ok", "model": "lightgbm"}

        payload = {
            "history": [
                {
                    "datetime": row.datetime.isoformat(),
                    "target_mean_kw": getattr(row, TARGET_COL),
                }
                for row in history.itertuples()
            ],
            "origin": origin.isoformat(),
        }
        response = client.post("/forecast", json=payload)

    assert response.status_code == 200
    api_forecast = [point["predicted_kw"] for point in response.json()]
    assert api_forecast == pytest.approx(direct_forecast["predicted_kw"].tolist())
    # sanity: the running process's env var actually took effect, not a
    # leftover MODEL_DIR from an earlier test
    assert os.environ["MODEL_DIR"] == str(model_dir)