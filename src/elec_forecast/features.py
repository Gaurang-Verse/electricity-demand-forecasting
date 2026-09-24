"""Phase 5 — Feature engineering.

Every feature here is built so that the row for hour t uses only
information strictly before t. This is the single most important property
of this module, and it's directly verified by
tests/test_features_leakage.py, not just asserted in this docstring.

Two feature families:

- Lag features: the target value exactly 24h, 48h, and 168h (1 week)
  before t. `shift(k)` on a value already excludes t itself for k >= 1, so
  these are leakage-safe by construction.

- Rolling statistics: mean/std over trailing windows. The naive
  `series.rolling(24).mean()` at row t INCLUDES t itself, which would leak
  the very thing being predicted into its own feature. To avoid this, the
  series is shifted by 1 hour *before* rolling, so the window at row t
  covers hours [t-24, t-1], never t.

Calendar features (hour of day, day of week, month, weekend flag) are
derived purely from the timestamp, which is always known in advance, so
there's no leakage risk there.
"""

import pandas as pd

TARGET_COL = "target_mean_kw"

LAG_HOURS = [24, 48, 168]
ROLLING_WINDOWS = [24, 168]

FEATURE_COLUMNS = (
    [f"lag_{h}h" for h in LAG_HOURS]
    + [f"roll_mean_{w}h" for w in ROLLING_WINDOWS]
    + [f"roll_std_{w}h" for w in ROLLING_WINDOWS]
    + ["hour", "dayofweek", "month", "is_weekend"]
)


def build_features(hourly: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Build the leakage-safe feature set from an hourly series.

    `hourly` must have a `datetime` column and be gap-free at hourly
    resolution (as produced by scripts/preprocess.py). Returns a DataFrame
    indexed by datetime with the target column, all engineered features,
    and rows dropped wherever a lag/rolling feature couldn't be computed
    (the first `max(LAG_HOURS)` hours of the series).
    """
    df = hourly.sort_values("datetime").set_index("datetime").copy()

    for h in LAG_HOURS:
        df[f"lag_{h}h"] = df[target_col].shift(h)

    shifted = df[target_col].shift(1)  # excludes the current hour from any rolling window
    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}h"] = shifted.rolling(window=w).mean()
        df[f"roll_std_{w}h"] = shifted.rolling(window=w).std()

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    required = [target_col, *FEATURE_COLUMNS]
    return df.dropna(subset=required)[required].reset_index()
