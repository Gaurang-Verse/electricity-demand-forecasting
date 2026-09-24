# Final Holdout Evaluation — Phase 9

Measured once by `scripts/evaluate_holdout.py`, on real data, after every
feature and model decision was already fixed by the backtest
(docs/backtest_results.md). This number was not tuned against, and the
script is not re-run after seeing it.

## Setup

LightGBM (the backtest winner, `configs/models.yaml` settings) trained on
**all 32,236 dev hours** (2006-12-24 to 2010-08-28 20:00) — not one fold's
training window, the full dev history — then scored once on the 2,160-hour
holdout (2010-08-28 21:00 to 2010-11-26 20:00), which spans the last 90
days of the dataset and was never used in any backtest fold, feature
decision, or model comparison.

## Result

| | RMSE (kW) | MAE (kW) | MAPE (%) | bias (kW) | n |
|---|---|---|---|---|---|
| **Holdout (final)** | **0.6549** | **0.4764** | **76.82** | +0.0429 | 2,160 |
| Backtest, pooled (12 folds) | 0.6156 | 0.4455 | 59.93 | +0.0195 | 8,640 |

| season | RMSE (kW) | MAPE (%) | n |
|---|---|---|---|
| summer | 0.6257 | 48.26 | 75 |
| autumn | 0.6559 | 77.85 | 2,085 |

(The holdout only touches 3 days of meteorological summer, so that row is
noisy — treat autumn, with 2,085 of the 2,160 holdout hours, as the
representative number.)

## This is worse than the backtest average, and that's reported directly

RMSE is 6.4% higher on holdout than the backtest's pooled average, MAPE is
17 points higher, and bias (the tendency to over-forecast) more than
doubled, from +0.0195 kW to +0.0429 kW. This is not a case where the
holdout confirms the backtest and there's nothing more to say.

**The most directly comparable backtest fold is fold 11**
(docs/backtest_results.md), which trained on data up to 2010-07-29 and
tested on the 30 days immediately before the holdout starts. Its RMSE was
0.3888 kW — barely half the holdout's 0.6549 kW, on a nearly adjacent
window. Autumn's backtest RMSE (pooled across the autumn-labeled fold
windows) was 0.6195, close to the holdout's 0.6559, so the degradation is
smaller when comparing like season to like season, and most of the gap to
fold 11 specifically looks like a seasonal effect (fold 11 is
July-August, genuinely one of the easiest windows in the whole backtest;
see docs/backtest_results.md's by-season table) rather than a holdout-vs-
backtest generalization gap.

**What this does and doesn't establish, stated plainly:** the holdout
result is close to the backtest's autumn-season numbers, which is
reassuring — it isn't wildly out of the distribution the backtest already
showed. But it is measurably worse than the *overall* backtest average,
and this project doesn't have a second, independent holdout to check
whether that's this particular 90-day window being unusually hard or a
genuine sign that backtest performance overstates what a truly new time
window will score. A longer holdout, or holdouts from multiple points in
the series, would be needed to tell those apart, and that's out of scope
here — noted as a limitation, not resolved by asserting an explanation
this single number can't support.

## What ships

`configs/models.yaml`'s LightGBM settings, trained on all of dev, are the
model saved to `data/processed/model/` and used by the API. The headline
number in the README and the career write-up is the **holdout** result
(RMSE 0.65 kW, MAPE 77%), not the more favorable backtest average — the
holdout is the honest answer to "how will this actually perform," and
quoting the backtest number instead would be cherry-picking the better of
two available numbers.
