"""Feature engineering.

The task is a direct 24-hour-ahead forecast: at a forecast origin T, using
only data observed before T, predict hourly average power for T .. T+23.
Every design decision below follows from taking that sentence literally.

Two entry points share one feature computation:

- build_features(hourly): training features. Every row's target value is
  already known (it's historical data), so this also returns the target
  column and drops any row missing a feature.

- build_forecast_features(history, origin): inference features for a
  24-hour window starting at `origin`, computed from `history` alone
  (which must end at origin - 1h, i.e. contain no data at or after the
  forecast). These rows have no target value yet -- that's the whole
  point, it's what gets predicted.

Both call _compute_features, which does the actual lag/rolling/calendar
math and never looks at a row's own target value, only at values strictly
older than HORIZON_HOURS. That's what makes the second entry point
possible at all: since no feature for hour t depends on anything from
[t - HORIZON_HOURS + 1, t], the 24 rows T .. T+23 can all be computed from
data available at forecast time, in one batch, with no recursion and no
predictions feeding into later predictions.

Feature families:

- Lags: the target exactly 24h, 48h and 168h (1 week) earlier. Every lag is
  >= HORIZON_HOURS, and build_features refuses to run otherwise.

- Rolling mean/std over 24h and 168h windows. Computed on the raw series,
  then looked up at (row's timestamp - HORIZON_HOURS), so the window for
  row t covers [t - 24 - w + 1, t - 24], never anything newer. (An earlier
  version of this shifted the series by the horizon before rolling instead
  of shifting the lookup; arithmetically equivalent when the series already
  spans both history and the rows being computed, but wrong for
  build_forecast_features, whose series contains only history and nothing
  at the forecast dates themselves -- see the bug note in
  docs/features_and_backtesting.md.)

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


def _compute_features(
    series: pd.Series,
    index: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    """Lag/rolling/calendar features for every position in `index`.

    `series` must be indexed by a contiguous hourly DatetimeIndex covering
    at least `index` and everything the lookback needs before it. Positions
    in `index` are never read from `series` at their own timestamp or
    later -- only calendar fields (derived from the index itself) do.
    """
    if min(LAG_HOURS) < horizon:
        raise ValueError(
            f"lag of {min(LAG_HOURS)}h is shorter than the {horizon}h horizon and would leak"
        )

    out = pd.DataFrame(index=index)
    for h in LAG_HOURS:
        out[f"lag_{h}h"] = series.reindex(index - pd.Timedelta(hours=h)).to_numpy()

    # Rolling stats are computed on the unshifted series (so they only ever
    # need dates that are actually in `series`), then looked up at
    # index - horizon, the same "how far back did this window end" logic as
    # the lag features above. This is arithmetically identical to shifting
    # the series by `horizon` first and rolling that -- shift-then-roll at
    # date d reads the same underlying values as roll-then-lookup-at
    # (d - horizon) -- but it doesn't require `series` to have any values at
    # `index` itself, which build_forecast_features's history-only series
    # never does.
    lookup = index - pd.Timedelta(hours=horizon)
    for w in ROLLING_WINDOWS:
        rolled_mean = series.rolling(window=w).mean()
        rolled_std = series.rolling(window=w).std()
        out[f"roll_mean_{w}h"] = rolled_mean.reindex(lookup).to_numpy()
        out[f"roll_std_{w}h"] = rolled_std.reindex(lookup).to_numpy()

    out["hour"] = index.hour
    out["dayofweek"] = index.dayofweek
    out["month"] = index.month
    out["is_weekend"] = (index.dayofweek >= 5).astype(int)
    return out


def build_features(
    hourly: pd.DataFrame,
    target_col: str = TARGET_COL,
    horizon: int = HORIZON_HOURS,
) -> pd.DataFrame:
    """Build the horizon-safe training feature set from a contiguous hourly series.

    `hourly` needs a `datetime` column and the target column. Returns one
    row per hour with `datetime`, the target and FEATURE_COLUMNS. Leading
    hours without enough history for every feature are dropped.
    """
    series = hourly.sort_values("datetime").set_index("datetime")[target_col]
    _check_contiguous_hourly(series.index)

    features = _compute_features(series, series.index, horizon)
    features[target_col] = series

    required = [target_col, *FEATURE_COLUMNS]
    return features.dropna(subset=required)[required].reset_index(names="datetime")


def build_forecast_features(
    history: pd.DataFrame,
    origin: pd.Timestamp,
    target_col: str = TARGET_COL,
    horizon: int = HORIZON_HOURS,
) -> pd.DataFrame:
    """Build features for the `horizon`-hour forecast window starting at `origin`.

    `history` needs a `datetime` column and the target column, containing
    only observed data strictly before `origin` (the most recent row must
    be at origin - 1h). Uses no data at or after `origin` -- there is none
    to use, since these are the hours being forecast. Returns one row per
    forecast hour with `datetime` and FEATURE_COLUMNS (no target column;
    it doesn't exist yet).

    Raises if `history` doesn't reach far enough back to fill every lag and
    rolling feature (rather than silently returning NaN features that a
    model would score against anyway).
    """
    series = history.sort_values("datetime").set_index("datetime")[target_col]
    _check_contiguous_hourly(series.index)

    last_observed = series.index.max()
    if last_observed != origin - pd.Timedelta(hours=1):
        raise ValueError(
            f"history must end exactly at origin - 1h ({origin - pd.Timedelta(hours=1)}), "
            f"got {last_observed}"
        )

    forecast_index = pd.date_range(origin, periods=horizon, freq="h")
    features = _compute_features(series, forecast_index, horizon)

    missing = features[FEATURE_COLUMNS].isna().any(axis=1)
    if missing.any():
        raise ValueError(
            f"not enough history to compute features for {int(missing.sum())} forecast "
            f"hour(s); history must extend back to at least "
            f"{origin - pd.Timedelta(hours=horizon + max(ROLLING_WINDOWS) - 1)}"
        )
    return features[FEATURE_COLUMNS].reset_index(names="datetime")
