"""Feature engineering.

The task is a direct 24-hour-ahead forecast: at a forecast origin T, using
only data observed before T, predict hours T .. T+23 in one go. So every
feature for a row inside that window must be computable from data before
T. For the last hour of the window (T+23), the newest usable value is at
T-1, which is 24 hours earlier. That makes the rule simple: **no feature
may use a value newer than HORIZON_HOURS before its own row.**

Feature families:

- Lags: the target exactly 24h, 48h and 168h (1 week) earlier. Every lag is
  >= HORIZON_HOURS, and build_features refuses to run otherwise.

- Rolling mean/std over 24h and 168h windows. The series is shifted by
  HORIZON_HOURS before rolling, so the window for row t covers
  [t - 24 - w + 1, t - 24], never anything newer.

  The first version of this module shifted by 1 hour instead. That is
  correct for a 1-hour-ahead forecast but leaks for a 24-hour-ahead one: the
  feature for T+5 would average in T+4, which hasn't been observed at
  forecast time. The original leakage test didn't catch it because it
  checked the wrong property ("does row t use data after t?" instead of
  "does any row in the forecast window use data from after the origin?").
  tests/test_features_leakage.py now checks the right property, and keeps
  the old builder around to prove the check catches it.

- Calendar: hour of day, day of week, month, weekend flag. These come from
  the timestamp alone, which is known in advance, so they can't leak.

The trade-off: shifting everything by the full horizon means the forecast
for T+0 ignores the 23 most recent hours it could legitimately have used.
The alternatives (one model per horizon step, or recursive forecasting
that feeds predictions back in) recover some of that at a real cost in
complexity. One horizon-safe model is the simplest correct option, and the
seasonal-naive baseline makes its cost visible.
"""

import pandas as pd

TARGET_COL = "target_mean_kw"

HORIZON_HOURS = 24
LAG_HOURS = [24, 48, 168]
ROLLING_WINDOWS = [24, 168]

FEATURE_COLUMNS = (
    [f"lag_{h}h" for h in LAG_HOURS]
    + [f"roll_mean_{w}h" for w in ROLLING_WINDOWS]
    + [f"roll_std_{w}h" for w in ROLLING_WINDOWS]
    + ["hour", "dayofweek", "month", "is_weekend"]
)


def _check_contiguous_hourly(index: pd.DatetimeIndex) -> None:
    """Lags and rolling windows use positional shift(), which silently
    misaligns if an hour is missing or duplicated. Refuse to run instead."""
    expected = pd.date_range(index.min(), index.max(), freq="h")
    if not index.equals(expected):
        raise ValueError(
            f"hourly series is not contiguous: {len(index):,} rows, expected "
            f"{len(expected):,} consecutive hours between {index.min()} and {index.max()}"
        )


def build_features(
    hourly: pd.DataFrame,
    target_col: str = TARGET_COL,
    horizon: int = HORIZON_HOURS,
) -> pd.DataFrame:
    """Build the horizon-safe feature set from a contiguous hourly series.

    `hourly` needs a `datetime` column and the target column. Returns one
    row per hour with `datetime`, the target and FEATURE_COLUMNS. Leading
    hours without enough history for every feature are dropped.
    """
    if min(LAG_HOURS) < horizon:
        raise ValueError(
            f"lag of {min(LAG_HOURS)}h is shorter than the {horizon}h horizon and would leak"
        )

    df = hourly.sort_values("datetime").set_index("datetime")[[target_col]].copy()
    _check_contiguous_hourly(df.index)

    for h in LAG_HOURS:
        df[f"lag_{h}h"] = df[target_col].shift(h)

    shifted = df[target_col].shift(horizon)
    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}h"] = shifted.rolling(window=w).mean()
        df[f"roll_std_{w}h"] = shifted.rolling(window=w).std()

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    required = [target_col, *FEATURE_COLUMNS]
    return df.dropna(subset=required)[required].reset_index()
