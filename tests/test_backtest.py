"""Integration tests for the backtest runner on small synthetic data."""

import numpy as np
import pandas as pd
import pytest

from elec_forecast.backtest import run_backtest, summarize
from elec_forecast.features import TARGET_COL
from elec_forecast.models import make_model
from elec_forecast.pipeline import prepare_backtest

CONFIG = {"holdout_hours": 200, "n_folds": 3, "fold_test_hours": 100, "min_train_hours": 500}


def _write(path, values: np.ndarray) -> None:
    pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=len(values), freq="h"),
        TARGET_COL: values,
        "n_minutes": 60,
    }).to_parquet(path, index=False)


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "hourly.parquet"
    _write(path, np.random.default_rng(0).uniform(0.2, 4.0, 1500))
    return prepare_backtest({"hourly_path": str(path), **CONFIG})


class _SpyModel:
    """Records which rows it was trained on; always predicts 1.0."""

    def __init__(self, seen: list):
        self.seen = seen

    def fit(self, X, y):
        self.seen.append(X.index)
        return self

    def predict(self, X):
        return np.ones(len(X))


def test_each_fold_is_trained_only_on_hours_before_its_origin(data):
    seen = []
    run_backtest(data, lambda: _SpyModel(seen))

    assert len(seen) == len(data.folds)
    for fold, idx in zip(data.folds, seen, strict=True):
        assert data.dev.loc[idx, "datetime"].max() < fold.origin


def test_predictions_cover_every_fold_window_exactly_once_and_never_the_holdout(data):
    preds = run_backtest(data, lambda: make_model("seasonal_naive"))

    assert len(preds) == CONFIG["n_folds"] * CONFIG["fold_test_hours"]
    assert preds["datetime"].is_unique
    assert preds["datetime"].max() < data.holdout["datetime"].min()


def test_seasonal_naive_is_exact_on_a_perfectly_weekly_series(tmp_path):
    """End-to-end check: if the series repeats every 168h, last week's value
    is always right, so the whole pipeline must report zero error."""
    week = 1.0 + np.random.default_rng(1).uniform(0, 3, 168)
    path = tmp_path / "hourly.parquet"
    _write(path, np.tile(week, 10))
    data = prepare_backtest({"hourly_path": str(path), **CONFIG})

    summary = summarize(run_backtest(data, lambda: make_model("seasonal_naive")))
    assert summary["overall"]["rmse"] == pytest.approx(0.0, abs=1e-12)


def test_summary_has_every_breakdown(data):
    summary = summarize(run_backtest(data, lambda: make_model("seasonal_naive")))
    assert set(summary) == {"overall", "per_fold", "by_season", "by_hour"}
    assert len(summary["per_fold"]) == CONFIG["n_folds"]
    assert sum(m["n"] for m in summary["by_hour"].values()) == summary["overall"]["n"]
