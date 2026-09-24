"""Walk-forward backtest runner and result summaries.

run_backtest() fits a fresh model per fold on that fold's training hours
only, then predicts its 30-day test window. It only ever looks at dev
data; the holdout is scored separately, once, at the very end.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from elec_forecast.features import FEATURE_COLUMNS, TARGET_COL
from elec_forecast.metrics import regression_metrics
from elec_forecast.pipeline import BacktestData

SEASONS = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


def run_backtest(data: BacktestData, model_factory: Callable[[], object]) -> pd.DataFrame:
    """Return one row per scored hour: datetime, fold, actual, predicted."""
    dev = data.dev
    ts = dev["datetime"]
    frames = []
    for fold in data.folds:
        train = dev[fold.train_mask(ts)]
        test = dev[fold.test_mask(ts)]

        model = model_factory()
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COL])
        predicted = np.asarray(model.predict(test[FEATURE_COLUMNS]), dtype=float)

        frames.append(pd.DataFrame({
            "datetime": test["datetime"].to_numpy(),
            "fold": fold.index,
            "actual": test[TARGET_COL].to_numpy(),
            "predicted": predicted,
        }))
    return pd.concat(frames, ignore_index=True)


def summarize(predictions: pd.DataFrame) -> dict:
    """Overall metrics, plus breakdowns by fold, season and hour of day."""
    def _m(group: pd.DataFrame) -> dict:
        return regression_metrics(group["actual"], group["predicted"])

    per_fold = [
        {"fold": int(k), "start": str(g["datetime"].min()), **_m(g)}
        for k, g in predictions.groupby("fold")
    ]
    season = predictions["datetime"].dt.month.map(SEASONS)
    hour = predictions["datetime"].dt.hour
    return {
        "overall": _m(predictions),
        "per_fold": per_fold,
        "by_season": {s: _m(g) for s, g in predictions.groupby(season)},
        "by_hour": {int(h): _m(g) for h, g in predictions.groupby(hour)},
    }
