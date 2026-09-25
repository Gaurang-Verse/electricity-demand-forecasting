# Household Electricity Demand Forecasting

Forecasts the next 24 hours of household electricity consumption from
historical smart-meter readings, built leakage-safe: chronological splits
only, and walk-forward backtesting across multiple time windows instead of
a single train/test split.

**Final result, evaluated once on a 90-day holdout no model decision ever
touched: RMSE 0.655 kW, MAPE 76.8%.** LightGBM beat a seasonal-naive
baseline (0.819 kW) and a linear-regression baseline (0.638 kW) in
backtesting, though only modestly over the linear model. The holdout
result is measurably worse than the backtest average (0.616 kW) — see
`docs/holdout_evaluation.md` for what that gap does and doesn't mean; it
isn't glossed over here.

This README is being built up phase by phase, alongside the project. See
`docs/` for the full design rationale and measured results.

## Status

✅ **Problem definition, dataset selection, design** — see project docs

✅ **Data acquisition** — real dataset downloaded and verified (2,075,259
rows, Dec 2006-Nov 2010, see docs/data_validation.md)

✅ **Data validation** — measured missing-value rate 1.25%, zero missing
timestamps, zero duplicates

✅ **Preprocessing** — resamples to hourly (34,589 hourly rows measured),
forward-fill only (no leakage); confirmed missingness co-occurs across all
columns as predicted

✅ **Feature engineering** — horizon-safe lag/rolling/calendar features.
The first version leaked into the 24h forecast window; caught and fixed,
see docs/features_and_backtesting.md

✅ **Walk-forward split framework** — 12 monthly folds + 90-day holdout

✅ **Models** — seasonal-naive, linear, LightGBM; backtested on all 12
folds. LightGBM wins overall (RMSE 0.616 kW vs 0.638 linear vs 0.819
seasonal-naive); see docs/backtest_results.md

✅ **Experiment tracking** — one MLflow run per model, per-fold metrics

✅ **Holdout evaluation + model versioning** — final result: RMSE 0.655 kW,
MAPE 76.8% on the untouched 90-day holdout (worse than the 0.616 kW
backtest average; see docs/holdout_evaluation.md for why)

✅ **Inference** — `Predictor` class turns a window of recent history into
a 24h forecast from the saved model artifact; see "Inference" below

✅ **REST API** — FastAPI `POST /forecast`, `GET /health`, `GET /metrics`;
see "REST API" below

✅ **Testing** — unit, leakage, API, integration, and monitoring tests, 74
total; see "Tests and lint" below

✅ **Docker** — image runs the API only (not training); verified locally:
build succeeds, `/health` degrades to "unhealthy" with no model mounted
and reports "ok" once one is, `/forecast` responds over real HTTP. See
"Docker" below.

✅ **CI/CD** — GitHub Actions workflow (lint + full test suite + a gated
Docker build) added; see "CI/CD" below. The same steps have been verified
locally (`ruff check` clean, 74/74 tests passing).

✅ **Monitoring** — feature drift detection and rolling forecast-error
tracking, validated with a null test (stays quiet on a fresh sample from
the same distribution) and a positive control (fires on a genuinely
shifted batch); demonstrated against real backtest/holdout data, since
there's no live production traffic yet. See "Monitoring" below.

⬜ **Full documentation** — `docs/` covers the data and modeling phases in
depth; inference/API/Docker/CI/monitoring design decisions are documented
in the README only so far.

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

**macOS only:** LightGBM needs the OpenMP runtime library, which isn't
bundled by pip on Mac. If `import lightgbm` fails with a `libomp.dylib`
error, install it once with Homebrew:

```
brew install libomp
```

## Getting the data

```
python scripts/download_data.py
```

See `docs/dataset.md` for licensing and a manual-download fallback if the
script can't reach the network from your machine.

## Validating the data

```
python scripts/validate_data.py
```

Prints row count, date range, missing-value rate, timestamp gaps, and
duplicate rows measured directly from the raw file, then saves a parquet
copy to `data/processed/`.

## Preprocessing

```
python scripts/preprocess.py
```

Forward-fills missing per-minute values (never backward-fill or
interpolation, to avoid leaking future values into a "past" row), then
resamples to hourly. Saves `data/processed/hourly.parquet`. See
`docs/data_validation.md` for the reasoning behind the fill strategy.

## Feature engineering and backtesting

`src/elec_forecast/features.py` builds lags (24h, 48h, 1 week), rolling
mean/std (24h and 168h windows) and calendar features. No feature uses
data newer than 24 hours before its own row, because that's the newest
data available for every hour of a 24-hour-ahead forecast.

The first version of the rolling features broke that rule, and its
leakage test passed anyway because it checked the wrong property.
`docs/features_and_backtesting.md` covers the bug, how it was caught, and
the rewritten test that now proves the fix.

`src/elec_forecast/splitting.py` splits the data chronologically: a
90-day holdout scored once at the end, plus 12 back-to-back 30-day
walk-forward folds with expanding training windows (settings in
`configs/backtest.yaml`). Print the real fold boundaries with:

```
python scripts/describe_folds.py
```

## Backtesting the models

```
python scripts/run_backtest.py
```

Fits seasonal-naive, linear regression and LightGBM (settings in
`configs/models.yaml`) on each of the 12 folds, prints RMSE / MAE / MAPE /
bias overall, per fold and per season, and saves the numbers to
`data/processed/backtest_results.json`. Only dev data is used; the holdout
isn't touched. Each model is logged as an MLflow run.

**Measured result:** LightGBM has the lowest error on every pooled metric
(RMSE 0.616 kW, vs. 0.638 for linear regression and 0.819 for the
seasonal-naive baseline). The margin over the linear model is real but
modest — most of the accuracy gain comes from the lag/rolling/calendar
features, not from model complexity. Full breakdown, including where
LightGBM does *not* beat the linear model (2 of 12 folds), in
`docs/backtest_results.md`.

To browse the MLflow runs:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Final holdout evaluation and model versioning

```
python scripts/evaluate_holdout.py
```

Trains the shipped model (LightGBM, on ALL dev hours, not just one fold)
and scores it once on the 90-day holdout no backtest fold has touched.
Saves the trained model to `data/processed/model/` (gitignored, same
reasoning as the arXiv project's model checkpoint: it's a regenerable
binary) and the measured metrics to
`data/processed/holdout_results.json` (committed).

**Run this once.** Re-running it after seeing the result and changing
something in response turns the holdout into a second validation set --
the script prints a warning if the results file already exists, but
doesn't stop you, because that decision belongs to you, not the script.

**Measured result: RMSE 0.6549 kW, MAPE 76.82%** on the 90-day holdout --
worse than the 0.6156 kW / 59.93% backtest average. This is reported as
the headline number for this project, not the more favorable backtest
figure. `docs/holdout_evaluation.md` covers what the gap does and doesn't
tell us, and why the most comparable backtest fold (nearly adjacent in
time) actually did much better, most likely a seasonal effect rather than
a sign the backtest was optimistic across the board.

## Inference

`src/elec_forecast/inference.py` provides `Predictor`, the one thing the
REST API depends on:

```python
from elec_forecast.inference import Predictor

predictor = Predictor("data/processed/model")  # loads the saved artifact once
forecast = predictor.predict(history)          # history: DataFrame with
                                                # datetime + target_mean_kw,
                                                # ending at origin - 1h
```

It calls `build_forecast_features`, a second entry point into
`features.py` alongside the training-time `build_features`. Both share one
private feature-computation core so the two can't silently drift apart;
`tests/test_forecast_features.py` proves it directly, by checking that
features computed from history alone (before the fact) exactly match
features computed from the full series with hindsight (after the fact).

At construction, `Predictor` checks the loaded model's metadata (feature
list, horizon, target column) against what the installed code currently
expects, and fails loudly if they've drifted apart, rather than silently
scoring a model against the wrong features.

**Bug found and fixed while building this:** the rolling-window features
computed correctly for training (`build_features`, where the series
already spans the dates being computed) but returned all-NaN for inference
(`build_forecast_features`, where the series is history-only and the
dates being forecast aren't in it at all). The original implementation
shifted the series by the horizon and then looked up the forecast dates
directly in that shifted series — which only has values where the
original series did, i.e. never at the forecast dates themselves. The fix
computes the rolling stats on the unshifted series and looks them up at
`(forecast date - horizon)` instead, which lands inside the history range
for every valid forecast window. Arithmetically it's the same rolling
window either way; only the lookup direction changed. Caught by the Phase
11 dry run (`tests/test_inference.py`, `tests/test_forecast_features.py`)
before any of this shipped — not reported by a user, not found in
production, because there isn't one yet.

## REST API

`src/elec_forecast/api.py` is a FastAPI service wrapping `Predictor`. Run
it (from the repo root, so the default model path resolves):

```
uvicorn elec_forecast.api:app --reload --port 8000
```

By default it loads the model from `data/processed/model` (the artifact
`scripts/evaluate_holdout.py` saves). Point it elsewhere with the
`MODEL_DIR` environment variable — useful for serving a different trained
model without a rebuild.

**`POST /forecast`** — body: `{"history": [{"datetime": ..., "target_mean_kw": ...}, ...], "origin": null}`.
`history` must be a contiguous run of hourly readings ending at
`origin - 1h`; omit `origin` and it defaults to one hour after the last
history point. Returns 24 `{"datetime": ..., "predicted_kw": ...}` rows.
Malformed input (a gap, history that doesn't reach the origin, not enough
lookback) returns 400 with the same message `build_forecast_features`
raises — no separate error-message layer to keep in sync.

**`GET /health`** — `{"status": "ok", "model": "lightgbm"}` once the model
loads at startup, or `{"status": "unhealthy", "detail": ...}` if it
didn't. The app still starts even when the model fails to load (so an
orchestrator can see `/health` report the problem), and `/forecast`
returns 503 in that state rather than crashing.

**`GET /metrics`** — Prometheus text format: request count by outcome
(`success/rejected`) and a request-latency histogram. This tracks that the
API is being called and how it's responding, not forecast accuracy --
that's Phase 16.

Tested with `tests/test_api.py` (FastAPI `TestClient`, a real fitted
LightGBM model built in a temp directory per test, no mocking) and
manually smoke-tested against a real running `uvicorn` process hitting
`/health`, `/forecast`, `/metrics`, and the gap-in-history error path over
real HTTP before this shipped.

## Tests and lint

```
ruff check src/ tests/ scripts/
python -m pytest tests/ -v
```

Use `python -m pytest`, not bare `pytest`, so the tests always run under
the venv's Python.

**Test layers:**

- **Unit** (`test_features.py`, `test_splitting.py`, `test_pipeline.py`,
  `test_models.py`, `test_metrics.py`, `test_backtest.py`,
  `test_artifact.py`, `test_forecast_features.py`, `test_inference.py`) --
  each module in isolation, with hand-built inputs.
- **Leakage** (`test_features_leakage.py`) — the property that makes the
  whole project defensible: no feature in the forecast window uses data
  from the forecast origin onward. Keeps the original 1-hour-shift bug's
  builder around as a fixture, with a dedicated test proving the detector
  still catches it, so the check can't quietly regress into one that
  always passes.
- **API** (`test_api.py`) — the FastAPI service via `TestClient`, real
  fitted models, no mocking.
- **Integration** (`test_integration.py`) — the seam the other layers
  can't see: data through `prepare_backtest`, a full backtest for every
  registered model, and a complete train -> save -> load -> `Predictor` ->
  live API round trip, all through the real functions each script calls,
  against a small synthetic dataset built for speed rather than the real
  one. This is a wiring check, not a source of measured results — those
  come only from the real scripts against the real data, in `docs/`.

74 tests total (measured with `python -m pytest tests/ -v`; ruff clean).

## Docker

The image runs the FastAPI service only — not the training or backtest
scripts. Training happens once, locally, against the real dataset; the
image serves whatever model you point it at. See `Dockerfile` for the
full reasoning (why `libgomp1` is installed explicitly, why the model
directory is never baked in).

Build:

```
docker build -t elec-forecast-api .
```

Run with your locally trained model mounted read-only:

```
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data/processed/model:/app/data/processed/model:ro" \
  elec-forecast-api
```

or with `docker compose up --build`, which does the same thing from
`docker-compose.yml`.

Without a mounted model the container still starts — `/health` reports
`{"status": "unhealthy", "detail": "..."}` instead of the process
crashing, and `/forecast` returns 503. This was verified directly, not
assumed: the container was run both with and without the mount, against
a real model trained by `scripts/evaluate_holdout.py`, and `/health`
reported the correct status each way.

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

- **`lint-and-test`** — installs the package (`pip install -e ".[dev]"`),
  runs `ruff check`, then the full test suite.
- **`docker-build`** — gated on the first job passing; confirms the
  Dockerfile still builds. This is the one check that couldn't be
  dry-run during development (the development sandbox's network policy
  blocks every container registry), so it runs for the first time for
  real on GitHub's own runners.

## Monitoring

`src/elec_forecast/monitoring.py` provides two lightweight checks: feature
drift detection and rolling forecast-error tracking. There's no live
production traffic for this project (see below), so both are demonstrated
against real backtest/holdout data rather than a live stream — but the
checks themselves are exactly what a real deployment would run.

**Drift detection** (`compute_reference_stats` / `drift_report`) compares a
batch of feature values against a trusted reference (per-feature mean,
expressed as a z-score using the reference std) and flags any feature whose
average has moved more than `z_threshold` standard deviations. It's
validated two ways, deliberately: a null test proving it stays quiet on a
fresh sample drawn from the same distribution as the reference, and a
positive control proving it actually fires on a batch shifted 4 reference
std devs away. Either test alone would pass trivially — a detector that
never fires passes the null test, one that always fires passes the positive
control — so both are required, the same discipline the leakage test
(`tests/test_features_leakage.py`) already applies. Run it against real
project data with:

```
python scripts/check_drift.py
```

This uses the dev feature set as the reference and the 90-day holdout as
the "current" batch. The holdout isn't drawn from a different population
than dev — it's just the same household's last 90 days — so no drift is
expected here; the script exists to run the check end-to-end against real
data, not to demonstrate genuine drift against live traffic that doesn't
exist yet.

**Rolling forecast-error tracking** (`rolling_forecast_error`) takes a
time-ordered log of `(datetime, actual, predicted)` rows — the shape a
real deployment would log per served forecast once the true value is
observed — and returns a rolling RMSE/MAE over a configurable window.
Demonstrated against real backtest predictions with:

```
python scripts/monitor_forecast_errors.py
```

This re-runs the backtest (retrains all 12 folds) to get row-level
predictions, since `scripts/run_backtest.py` only saves the summarized
metrics, then prints the rolling 7-day forecast error.

Both checks are unit-tested in `tests/test_monitoring.py`, including the
null-test/positive-control pair for drift and a hand-computed check of the
rolling-error arithmetic.

**What this is and isn't:** there's no live production endpoint serving
real users, so "monitoring" here means the mechanism is built, tested, and
demonstrated end-to-end against real historical data — not that it's
watching genuine production traffic. If this were deployed, `/forecast`
requests and their eventual actuals would feed these same functions
directly.

## Data and credits

Household power consumption data is from the UCI Machine Learning
Repository, released under CC BY 4.0 — see `docs/dataset.md` for the full
citation.

Built by Gaurang Kumbhar.