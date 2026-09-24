"""Backtest every model on the walk-forward folds and log the runs to MLflow.

Only dev data is used. The 90-day holdout isn't touched here; it gets
scored once, by a separate script, after every modelling decision is made.

Writes:
  data/processed/backtest_results.json          (committed: the measured numbers)
  data/processed/backtest_predictions_<model>.parquet  (not committed)

Run: python scripts/run_backtest.py [--no-mlflow]
"""

import argparse
import json
from functools import partial
from pathlib import Path

import yaml

from elec_forecast.backtest import run_backtest, summarize
from elec_forecast.models import make_model
from elec_forecast.pipeline import prepare_backtest
from elec_forecast.tracking import log_backtest

OUT_DIR = Path("data/processed")

parser = argparse.ArgumentParser()
parser.add_argument("--backtest-config", default="configs/backtest.yaml")
parser.add_argument("--models-config", default="configs/models.yaml")
parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
args = parser.parse_args()

with open(args.backtest_config) as f:
    backtest_config = yaml.safe_load(f)
with open(args.models_config) as f:
    models_config = yaml.safe_load(f)["models"]

data = prepare_backtest(backtest_config)
print(f"{len(data.folds)} folds, {len(data.dev):,} dev hours "
      f"(holdout of {len(data.holdout):,} hours not used)\n")

results = {}
for name, params in models_config.items():
    params = params or {}
    predictions = run_backtest(data, partial(make_model, name, params))
    summary = summarize(predictions)
    results[name] = {"params": params, **summary}

    predictions.to_parquet(OUT_DIR / f"backtest_predictions_{name}.parquet", index=False)
    if not args.no_mlflow:
        log_backtest(name, params, summary, backtest_config)

with open(OUT_DIR / "backtest_results.json", "w") as f:
    json.dump(results, f, indent=2)

names = list(results)

print("Overall (all 12 folds pooled)")
print(f"{'model':<16}{'RMSE kW':>10}{'MAE kW':>10}{'MAPE %':>10}{'bias kW':>10}")
for name in names:
    o = results[name]["overall"]
    print(f"{name:<16}{o['rmse']:>10.4f}{o['mae']:>10.4f}{o['mape']:>10.2f}{o['bias']:>+10.4f}")

print("\nRMSE (kW) per fold")
print(f"{'fold':>4}  {'start':<11}" + "".join(f"{n:>16}" for n in names))
for i, fold in enumerate(results[names[0]]["per_fold"]):
    row = "".join(f"{results[n]['per_fold'][i]['rmse']:>16.4f}" for n in names)
    print(f"{fold['fold']:>4}  {fold['start'][:10]:<11}{row}")

print("\nRMSE (kW) by season")
for season in ("winter", "spring", "summer", "autumn"):
    row = "".join(f"{results[n]['by_season'][season]['rmse']:>16.4f}" for n in names)
    print(f"{season:<17}{row}")

print(f"\nSaved {OUT_DIR / 'backtest_results.json'}")
