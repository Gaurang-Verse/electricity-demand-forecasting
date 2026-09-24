"""Single entry point for the data every model trains and is scored on.

Every training and evaluation script calls prepare_backtest() rather than
loading and splitting data itself, so the baselines and LightGBM are always
compared on identical rows, features and folds.

Features are built once on the full series, holdout included. That's safe
because every feature is causal (it only reads data at least 24h older than
its own row, verified in tests/test_features_leakage.py), so holdout values
can never flow into a dev row's features.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from elec_forecast.features import TARGET_COL, build_features
from elec_forecast.splitting import Fold, holdout_start, walk_forward_folds


@dataclass
class BacktestData:
    features: pd.DataFrame  # datetime, target and feature columns, every usable hour
    dev: pd.DataFrame  # everything before the holdout
    holdout: pd.DataFrame  # the final block, scored once at the end
    folds: list[Fold]  # walk-forward folds over dev


def load_hourly(path: str | Path) -> pd.DataFrame:
    """Load the hourly series written by scripts/preprocess.py.

    Hours built from fewer than 60 minutes are dropped. In the real data
    these are only the first and last hours of the series (it starts at
    17:24 and ends at 21:02). If a partial hour ever appeared mid-series,
    dropping it would leave a gap, and build_features would refuse to run.
    """
    hourly = pd.read_parquet(path)
    hourly = hourly[hourly["n_minutes"] == 60]
    return hourly[["datetime", TARGET_COL]].reset_index(drop=True)


def prepare_backtest(config: dict) -> BacktestData:
    features = build_features(load_hourly(config["hourly_path"]))

    cutoff = holdout_start(features["datetime"], config["holdout_hours"])
    dev = features[features["datetime"] < cutoff].reset_index(drop=True)
    holdout = features[features["datetime"] >= cutoff].reset_index(drop=True)

    folds = walk_forward_folds(
        dev["datetime"],
        n_folds=config["n_folds"],
        fold_test_hours=config["fold_test_hours"],
        min_train_hours=config["min_train_hours"],
    )
    return BacktestData(features=features, dev=dev, holdout=holdout, folds=folds)
