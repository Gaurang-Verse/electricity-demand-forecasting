# Household Electricity Demand Forecasting

Forecasts the next 24 hours of household electricity consumption from
historical smart-meter readings, built leakage-safe: chronological splits
only, and walk-forward backtesting across multiple time windows instead of
a single train/test split.

This README is being built up phase by phase, alongside the project. Right
now it only covers what's actually implemented so far (data acquisition and
validation). See `docs/` for the full design rationale.

## Status

- [x] Problem definition, dataset selection, design (see project docs)
- [x] Data acquisition
- [ ] Data validation — script written, not yet run against real data
- [ ] Preprocessing
- [ ] Feature engineering
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

## Data and credits

Household power consumption data is from the UCI Machine Learning
Repository, released under CC BY 4.0 — see `docs/dataset.md` for the full
citation.

Built by Gaurang Kumbhar.
