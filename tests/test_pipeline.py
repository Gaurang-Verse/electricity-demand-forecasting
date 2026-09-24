"""Integration test: prepare_backtest end to end on a small synthetic parquet."""

import numpy as np
import pandas as pd

from elec_forecast.features import FEATURE_COLUMNS, TARGET_COL
from elec_forecast.pipeline import load_hourly, prepare_backtest


def _write_hourly(path, n_hours: int, partial_edges: bool = True) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=n_hours, freq="h"),
        TARGET_COL: rng.uniform(0.1, 5.0, n_hours),
        "n_minutes": 60,
    })
    if partial_edges:  # mimic the real series, which starts and ends mid-hour
        df.loc[0, "n_minutes"] = 36
        df.loc[n_hours - 1, "n_minutes"] = 3
    df.to_parquet(path, index=False)


def _config(path) -> dict:
    return {
        "hourly_path": str(path),
        "holdout_hours": 200,
        "n_folds": 3,
        "fold_test_hours": 100,
        "min_train_hours": 500,
    }


def test_load_hourly_drops_partial_hours(tmp_path):
    path = tmp_path / "hourly.parquet"
    _write_hourly(path, 50)
    assert len(load_hourly(path)) == 48


def test_prepare_backtest_keeps_holdout_out_of_every_fold(tmp_path):
    path = tmp_path / "hourly.parquet"
    _write_hourly(path, 1500)
    data = prepare_backtest(_config(path))

    assert len(data.holdout) == 200
    assert data.dev["datetime"].max() < data.holdout["datetime"].min()

    dev_ts = data.dev["datetime"]
    for fold in data.folds:
        scored = dev_ts[fold.test_mask(dev_ts)]
        assert scored.max() < data.holdout["datetime"].min()
    assert data.folds[-1].test_end == data.holdout["datetime"].min()


def test_prepare_backtest_returns_complete_feature_rows(tmp_path):
    path = tmp_path / "hourly.parquet"
    _write_hourly(path, 1500)
    data = prepare_backtest(_config(path))
    assert not data.features[[TARGET_COL, *FEATURE_COLUMNS]].isna().any().any()
