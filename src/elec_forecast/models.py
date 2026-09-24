"""The three models compared in the backtest.

All of them expose fit(X, y) / predict(X) over FEATURE_COLUMNS, so the
backtest treats them identically.

- seasonal_naive: predicts the value from the same hour one week earlier
  (the lag_168h feature). Nothing to fit. This is the bar the other two
  have to clear; household load repeats weekly, so it's a strong one.
- linear: ordinary least squares on the lag/rolling features, with hour,
  day of week and month one-hot encoded. Treating hour as a number would
  force a straight-line relationship between "hour 3" and "hour 20", which
  makes no sense. is_weekend is dropped here because it's fully determined
  by the one-hot day of week.
- lightgbm: gradient-boosted trees on the same features. Trees split on
  hour/month thresholds directly, so no encoding is needed.
"""

import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MODEL_NAMES = ("seasonal_naive", "linear", "lightgbm")


class SeasonalNaive:
    """Same hour last week. Needs no training data."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaive":
        return self

    def predict(self, X: pd.DataFrame):
        return X["lag_168h"].to_numpy()


def _linear() -> Pipeline:
    encode = ColumnTransformer(
        [
            ("calendar", OneHotEncoder(handle_unknown="ignore"), ["hour", "dayofweek", "month"]),
            ("redundant", "drop", ["is_weekend"]),
        ],
        remainder="passthrough",
    )
    return Pipeline([("encode", encode), ("regress", LinearRegression())])


def make_model(name: str, params: dict | None = None):
    """Return a fresh, unfitted model. `params` only applies to lightgbm."""
    params = params or {}
    if name == "seasonal_naive":
        return SeasonalNaive()
    if name == "linear":
        return _linear()
    if name == "lightgbm":
        return lgb.LGBMRegressor(verbose=-1, **params)
    raise ValueError(f"unknown model '{name}', expected one of {MODEL_NAMES}")
