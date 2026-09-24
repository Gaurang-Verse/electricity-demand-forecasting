# Household Electricity Demand Forecasting

Forecasts the next 24 hours of household electricity consumption from
historical smart-meter readings, built leakage-safe: chronological splits
only, and walk-forward backtesting across multiple time windows instead of
a single train/test split.

This README is being built up phase by phase, alongside the project. Right
now it only covers what's actually implemented so far (data acquisition
through the model backtest). See `docs/` for the full design rationale and
measured results.

## Status

- [x] Problem definition, dataset selection, design (see project docs)
- [x] Data acquisition — real dataset downloaded and verified (2,075,259
      rows, Dec 2006-Nov 2010, see docs/data_validation.md)
- [x] Data validation — measured missing-value rate 1.25%, zero missing
      timestamps, zero duplicates
- [x] Preprocessing — resamples to hourly (34,589 hourly rows measured),
      forward-fill only (no leakage); confirmed missingness co-occurs
      across all columns as predicted
- [x] Feature engineering — horizon-safe lag/rolling/calendar features.
      The first version leaked into the 24h forecast window; caught and
      fixed, see docs/features_and_backtesting.md
- [x] Walk-forward split framework — 12 monthly folds + 90-day holdout
- [x] Models — seasonal-naive, linear, LightGBM; backtested on all 12
      folds. LightGBM wins overall (RMSE 0.616 kW vs 0.638 linear vs
      0.819 seasonal-naive); see docs/backtest_results.md
- [x] Experiment tracking — one MLflow run per model, per-fold metrics
- [ ] Holdout evaluation (once, at the end)
- [ ] Model versioning
- [ ] Inference / REST API
- [ ] Testing
- [ ] Docker
- [ ] CI/CD
- [ ] Monitoring
- [ ] Full documentation

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

## Tests and lint

```
ruff check src/ tests/ scripts/
python -m pytest tests/ -v
```

Use `python -m pytest`, not bare `pytest`, so the tests always run under
the venv's Python.

## Data and credits

Household power consumption data is from the UCI Machine Learning
Repository, released under CC BY 4.0 — see `docs/dataset.md` for the full
citation.

Built by Gaurang Kumbhar.
