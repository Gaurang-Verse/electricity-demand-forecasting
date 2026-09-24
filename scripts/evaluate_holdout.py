"""Phase 9-10 -- Final holdout evaluation and model versioning.

This trains the model that actually ships -- LightGBM, fit on every hour of
dev (not just one fold) -- and scores it on the 90-day holdout block that no
backtest fold, no feature decision, and no model comparison has touched.

Run this ONCE. Its whole point is to be the number nothing was tuned
against. If you change a feature, a hyperparameter, or the model choice
after seeing this output and rerun it "to check", the holdout has become a
second validation set and this number stops meaning what it claims to
mean. If you need to iterate, go back to scripts/run_backtest.py and the
12 walk-forward folds -- that's what they're for.

Writes:
  data/processed/model/model.joblib, metadata.json   (the shipped model;
    gitignored -- see docs/dataset.md-style reasoning in .gitignore)
  data/processed/holdout_results.json                (committed: the
    measured numbers this whole exercise exists to produce)

Run: python scripts/evaluate_holdout.py
"""

import json
import sys
from pathlib import Path

import yaml

from elec_forecast.artifact import ModelMetadata, save_model
from elec_forecast.features import FEATURE_COLUMNS, HORIZON_HOURS, TARGET_COL
from elec_forecast.metrics import regression_metrics
from elec_forecast.models import make_model
from elec_forecast.pipeline import prepare_backtest

RESULTS_PATH = Path("data/processed/holdout_results.json")
MODEL_DIR = Path("data/processed/model")
FINAL_MODEL = "lightgbm"

if RESULTS_PATH.exists():
    print(
        f"WARNING: {RESULTS_PATH} already exists. This script is meant to run once. "
        "Re-running after looking at a previous result and changing something in "
        "response defeats the purpose of a holdout. Proceeding anyway -- make sure "
        "that's actually what you intend.",
        file=sys.stderr,
    )

with open("configs/backtest.yaml") as f:
    backtest_config = yaml.safe_load(f)
with open("configs/models.yaml") as f:
    model_params = yaml.safe_load(f)["models"].get(FINAL_MODEL, {})

data = prepare_backtest(backtest_config)
print(f"Training {FINAL_MODEL} on all {len(data.dev):,} dev hours "
      f"({data.dev['datetime'].min()} to {data.dev['datetime'].max()})")
print(f"Evaluating once on {len(data.holdout):,} holdout hours "
      f"({data.holdout['datetime'].min()} to {data.holdout['datetime'].max()})\n")

model = make_model(FINAL_MODEL, model_params)
model.fit(data.dev[FEATURE_COLUMNS], data.dev[TARGET_COL])

predicted = model.predict(data.holdout[FEATURE_COLUMNS])
actual = data.holdout[TARGET_COL]
overall = regression_metrics(actual, predicted)

season_map = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
              6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
season = data.holdout["datetime"].dt.month.map(season_map)
by_season = {
    s: regression_metrics(actual[season == s], predicted[(season == s).to_numpy()])
    for s in season.unique()
}

results = {
    "model": FINAL_MODEL,
    "params": model_params,
    "trained_on_hours": len(data.dev),
    "trained_on_range": [str(data.dev["datetime"].min()), str(data.dev["datetime"].max())],
    "holdout_hours": len(data.holdout),
    "holdout_range": [str(data.holdout["datetime"].min()), str(data.holdout["datetime"].max())],
    "overall": overall,
    "by_season": by_season,
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

save_model(
    model,
    ModelMetadata(
        model_name=FINAL_MODEL,
        feature_columns=list(FEATURE_COLUMNS),
        target_col=TARGET_COL,
        horizon_hours=HORIZON_HOURS,
        trained_on=f"{len(data.dev):,} dev hours, "
                    f"{data.dev['datetime'].min()} to {data.dev['datetime'].max()}",
    ),
    MODEL_DIR,
)

print(f"HOLDOUT  rmse={overall['rmse']:.4f} kW  mae={overall['mae']:.4f} kW  "
      f"mape={overall['mape']:.2f}%  bias={overall['bias']:+.4f} kW  n={overall['n']:,}")
print()
for s, m in by_season.items():
    print(f"  {s:<8} rmse={m['rmse']:.4f} kW  mape={m['mape']:.2f}%  n={m['n']:,}")
print(f"\nSaved {RESULTS_PATH}")
print(f"Saved model artifact to {MODEL_DIR}/ (model.joblib, metadata.json)")
