# Backtest Results — Phases 7-8

Measured by `scripts/run_backtest.py` on the real downloaded data: 12
walk-forward folds, 30 days each, spanning September 2009 to August 2010.
Every fold trains on all dev hours before its origin (23,596 hours in the
first fold, growing to 31,516 in the last) and is scored on the 720 hours
that follow. The holdout (the final 90 days) is not used here.

## Overall (all 12 folds pooled)

| model | RMSE (kW) | MAE (kW) | MAPE (%) | bias (kW) |
|---|---|---|---|---|
| seasonal_naive | 0.8194 | 0.5619 | 69.25 | +0.0008 |
| linear | 0.6384 | 0.4742 | 67.10 | +0.0019 |
| **lightgbm** | **0.6156** | **0.4455** | **59.93** | +0.0195 |

LightGBM has the lowest error on every pooled metric. Relative to
seasonal-naive, RMSE drops 24.9% and MAPE drops 9.3 percentage points.
Relative to linear, the gain is much smaller: RMSE drops 3.6%, MAE drops
6.1%. The features are doing most of the work here (lags and rolling
stats over calendar structure); LightGBM's advantage over a linear model
on the same features is real but modest.

**MAPE is high (60-69%) in absolute terms.** This is a property of the
target, not a broken metric: household load has an average around 1 kW
but drops toward near-zero standby draw overnight, and MAPE's denominator
is the actual value, so small-value hours produce large percentage errors
even from small absolute errors. RMSE and MAE, in kW, are the more
meaningful numbers for this series; MAPE is reported because the design
doc specified it, with this caveat attached rather than silently
included.

**Bias:** all three models are close to unbiased overall, but LightGBM's
bias (+0.0195 kW) is an order of magnitude larger than the other two
(+0.0008, +0.0019), meaning it has a small, real tendency to over-forecast
on average across this backtest. It doesn't change the ranking (LightGBM
still has the lowest RMSE and MAE), but it's worth watching if the model
is ever calibrated against real usage rather than backtested.

## Per fold (RMSE, kW)

| fold | start | seasonal_naive | linear | lightgbm |
|---|---|---|---|---|
| 0 | 2009-09-02 | 0.6889 | 0.5536 | 0.5215 |
| 1 | 2009-10-02 | 0.7965 | 0.6799 | 0.6435 |
| 2 | 2009-11-01 | 0.8841 | 0.7014 | 0.6778 |
| 3 | 2009-12-01 | 0.9594 | 0.7520 | 0.7382 |
| 4 | 2009-12-31 | 0.9651 | 0.7607 | 0.7508 |
| 5 | 2010-01-30 | 0.9600 | 0.7207 | 0.7004 |
| 6 | 2010-03-01 | 0.9158 | 0.6566 | 0.6328 |
| 7 | 2010-03-31 | 0.7643 | 0.5956 | 0.5959 |
| 8 | 2010-04-30 | 0.8400 | 0.5959 | 0.6035 |
| 9 | 2010-05-30 | 0.6780 | 0.5410 | 0.5067 |
| 10 | 2010-06-29 | 0.7019 | 0.5530 | 0.5241 |
| 11 | 2010-07-29 | 0.5591 | 0.4764 | 0.3888 |

LightGBM beats seasonal-naive in all 12 folds. It beats linear in 10 of
12; in fold 7 the two are essentially tied (0.5959 vs 0.5956, LightGBM
0.0003 worse) and in fold 8 linear is clearly better (0.5959 vs 0.6035).
Both are spring folds. This is reported rather than smoothed over: the
pooled numbers say LightGBM wins, but "wins in every fold" would be a
stronger and false claim.

## By season (RMSE, kW)

| season | seasonal_naive | linear | lightgbm |
|---|---|---|---|
| winter | 0.9575 | 0.7429 | 0.7285 |
| spring | 0.8427 | 0.6159 | 0.6103 |
| summer | 0.6496 | 0.5245 | 0.4765 |
| autumn | 0.7953 | 0.6504 | 0.6195 |

Every model is worst in winter and best in summer, for all three models.
Winter household load likely has more day-to-day variance (heating
behavior responding to weather this dataset doesn't include — see the
"no weather covariate" limitation in the design doc), which is a plausible
explanation but not something this backtest measures directly.

## What this means for the next phase

LightGBM is the model that ships (as planned). The margin over the linear
baseline is real but small enough that it's worth being honest about in
the career write-up: the bulk of the accuracy gain here comes from
feature engineering (lags, rolling stats, calendar structure), not from
the choice of a more complex model over a linear one.

The holdout (final 90 days) has not been touched and is evaluated once,
after this comparison, with the model and any remaining decisions fixed.
