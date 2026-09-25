"""Loads the shipped model and turns a window of history into a 24h forecast.

Predictor is the one thing the API (Phase 12) depends on. It owns the
"call this with plain data, get a forecast back" contract; the API layer
should never call build_forecast_features or the model directly.
"""

import pandas as pd

from elec_forecast.artifact import load_model
from elec_forecast.features import (
    FEATURE_COLUMNS,
    HORIZON_HOURS,
    TARGET_COL,
    build_forecast_features,
)


class Predictor:
    """Loads a saved model artifact once and serves forecasts from it.

    Construction verifies the artifact's metadata matches what this code
    expects (feature list, horizon) rather than trusting it silently --
    the same discipline as Project A's Predictor checking num_labels
    against the label space at startup. A model trained by a different
    version of features.py should fail loudly here, not produce quietly
    wrong forecasts.
    """

    def __init__(self, model_dir: str):
        self.model, self.metadata = load_model(model_dir)

        if self.metadata.feature_columns != list(FEATURE_COLUMNS):
            raise ValueError(
                f"model was trained on features {self.metadata.feature_columns}, "
                f"but this code expects {list(FEATURE_COLUMNS)}; the model artifact "
                f"and the installed code have drifted apart"
            )
        if self.metadata.horizon_hours != HORIZON_HOURS:
            raise ValueError(
                f"model was trained for a {self.metadata.horizon_hours}h horizon, "
                f"code expects {HORIZON_HOURS}h"
            )
        if self.metadata.target_col != TARGET_COL:
            raise ValueError(
                f"model was trained on target column '{self.metadata.target_col}', "
                f"code expects '{TARGET_COL}'"
            )

    def predict(self, history: pd.DataFrame, origin: pd.Timestamp | None = None) -> pd.DataFrame:
        """Forecast HORIZON_HOURS hours starting at `origin`.

        `history` needs `datetime` and TARGET_COL columns, containing only
        observed hourly readings ending at `origin - 1h`. If `origin` is
        omitted, it's the hour right after the last row in `history`.

        Returns a DataFrame with `datetime` and `predicted_kw`, one row per
        forecast hour.
        """
        if origin is None:
            origin = history["datetime"].max() + pd.Timedelta(hours=1)

        features = build_forecast_features(history, origin, target_col=TARGET_COL)
        predicted = self.model.predict(features[FEATURE_COLUMNS])
        return pd.DataFrame({"datetime": features["datetime"], "predicted_kw": predicted})
