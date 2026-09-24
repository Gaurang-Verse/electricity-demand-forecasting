# Features and Backtesting

## The forecasting task, precisely

At a forecast origin T (the first hour being predicted), using only data
observed before T, predict hourly average power for T .. T+23. Every design
decision below follows from taking that sentence literally.

## Features (`src/elec_forecast/features.py`)

| Feature | Definition | Newest data used, for row t |
|---|---|---|
| `lag_24h`, `lag_48h`, `lag_168h` | target 24h / 48h / 1 week earlier | t-24 |
| `roll_mean_24h`, `roll_std_24h` | mean/std over the 24 hours ending at t-24 | t-24 |
| `roll_mean_168h`, `roll_std_168h` | mean/std over the 168 hours ending at t-24 | t-24 |
| `hour`, `dayofweek`, `month`, `is_weekend` | from the timestamp | none (known in advance) |

The rule is that no feature uses anything newer than 24 hours before its own
row. For the last hour of a forecast window (T+23), the newest observed
value is T-1, exactly 24 hours earlier, so this is the loosest rule that
stays correct for every hour in the window.

`build_features` also refuses to run if the hourly series has a missing or
duplicated hour. Lags and rolling windows use positional `shift()`, which
would silently misalign across a gap instead of failing.

## A leak I shipped, and how I caught it

The first version of this module shifted the series by **1 hour** before
computing rolling windows. It came with a leakage test, and the test passed.

The problem showed up when I lined the features up against the actual
task. A 1-hour shift is correct for a 1-hour-ahead forecast. For a
24-hour-ahead forecast it isn't: the feature row for T+5 would average in
the value at T+4, which hasn't been observed when the forecast is made at
T. A model trained that way would look better in backtesting than it could
ever perform in real use.

The original test didn't catch it because it checked the wrong property.
It asked "does the feature row for hour t use any data after t?", and with
a 1-hour shift the answer really is no. The property that matters is "does
any feature row inside the forecast window [T, T+23] use data from T
onward?"

The rewritten test (`tests/test_features_leakage.py`) checks that directly.
It builds features twice, once on a normal series and once with every
value from T onward replaced by an extreme outlier, and fails if any
feature inside the forecast window changes. Run against the original code,
it failed on exactly the four rolling features:

```
FAILED test_no_feature_in_the_forecast_window_uses_data_from_the_origin_onward
AssertionError: assert ['roll_mean_2...oll_std_168h'] == []
  Left contains 4 more items, first extra item: 'roll_mean_24h'
```

The lags (all >= 24h) and calendar features passed. The fix was shifting by
the horizon (24h) instead of 1h before rolling. The original buggy builder
stays in the test file, with a test asserting the detector still flags it,
so the leakage check can't quietly stop working.

**The cost of the fix:** every prediction now uses inputs that are at least
24 hours old, including the forecast for T+0, which could legitimately
have used data from T-1. Per-horizon models or recursive forecasting would
recover some of that, but both add real complexity. A single horizon-safe
model is the simplest correct option, and the baselines show what it costs.

**Consequence for evaluation:** the design doc planned to report error
separately for 1-hour-ahead vs 24-hour-ahead predictions. With this model,
every prediction uses inputs of the same age, so that breakdown would show
nothing. Error by hour of day and by season are reported instead.

## Backtesting (`src/elec_forecast/splitting.py`, `configs/backtest.yaml`)

No random splits anywhere. Every boundary is a timestamp.

```
|--------------------- dev ---------------------|--- holdout ---|
| training (expanding) ...      | fold k: 30 days |   90 days   |
```

- **Holdout:** the last 90 days (2,160 hours), scored exactly once at the
  end, after every model and feature decision has been made on the folds.
- **Walk-forward folds:** 12 back-to-back 30-day test windows immediately
  before the holdout. That covers a full year, so every season gets scored.
  Fold k trains on every dev hour before its origin (an expanding window)
  and is scored on the 30 days after it.
- **Minimum history:** every fold must train on at least one full year
  (8,760 hours), so the model has seen each season before being scored on
  it. The split raises an error rather than silently training on less.

Inside a fold's 30-day window the model is frozen, but its inputs keep
arriving, because each hour's features use observed data from 24 hours
earlier. That's the same as retraining monthly and issuing a fresh
next-day forecast every day.

Features are built once on the full series, holdout included. That's safe
only because every feature is causal, which the leakage tests prove, so
holdout values can't reach a dev row's features.

`scripts/describe_folds.py` prints the real fold boundaries and training
sizes for the downloaded data.
