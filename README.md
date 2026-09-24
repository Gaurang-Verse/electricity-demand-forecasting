# Household Electricity Demand Forecasting

Forecasts the next 24 hours of household electricity consumption from
historical smart-meter readings, built leakage-safe: chronological splits
only, and walk-forward backtesting across multiple time windows instead of
a single train/test split.

This README is being built up phase by phase, alongside the project. Right
now it only covers what's actually implemented so far (data acquisition
through preprocessing). See `docs/` for the full design rationale and
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
- [x] Feature engineering — leakage-safe lag/rolling/calendar features,
      verified by a dedicated test that corrupts future values and checks
      earlier features don't change (tests/test_features_leakage.py)
- [ ] Baseline + walk-forward split framework
- [ ] Model training
- [ ] Experiment tracking
- [ ] Model evaluation
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

## Feature engineering

`src/elec_forecast/features.py` builds lag features (same hour 1, 2, and 7
days prior), rolling mean/std (trailing 24h and 168h windows, computed on
data shifted by 1 hour so the window never includes the current hour), and
calendar features (hour, day of week, month, weekend flag) — all
leakage-safe by construction. `tests/test_features_leakage.py` proves this
rather than asserting it: it corrupts every value after a chosen cutoff
hour with an extreme outlier and checks that features at or before the
cutoff are byte-for-byte unchanged, plus a sanity check that corrupting the
*past* does change later features (so the test isn't trivially passing).

```
pytest tests/ -v
```

## Data and credits

Household power consumption data is from the UCI Machine Learning
Repository, released under CC BY 4.0 — see `docs/dataset.md` for the full
citation.

Built by Gaurang Kumbhar.
