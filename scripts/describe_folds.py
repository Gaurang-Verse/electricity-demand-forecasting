"""Print the real backtest layout: dev, holdout and every fold's boundaries.

Run this before training anything, to check the folds cover what the
config says they cover.

Run: python scripts/describe_folds.py [configs/backtest.yaml]
"""

import sys

import yaml

from elec_forecast.pipeline import prepare_backtest

config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/backtest.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

data = prepare_backtest(config)
dev_ts = data.dev["datetime"]

print(f"Feature rows: {len(data.features):,}  "
      f"({data.features['datetime'].min()} to {data.features['datetime'].max()})")
print(f"Dev:          {len(data.dev):,} hours  "
      f"({dev_ts.min()} to {dev_ts.max()})")
print(f"Holdout:      {len(data.holdout):,} hours  "
      f"({data.holdout['datetime'].min()} to {data.holdout['datetime'].max()})")
print()
header = ("fold", "origin", "test_end (excl.)", "train_hours", "test_hours")
print(f"{header[0]:>4}  {header[1]:<19}  {header[2]:<19}  {header[3]:>11}  {header[4]:>10}")
for fold in data.folds:
    n_train = int(fold.train_mask(dev_ts).sum())
    n_test = int(fold.test_mask(dev_ts).sum())
    print(f"{fold.index:>4}  {fold.origin!s:<19}  {fold.test_end!s:<19}  "
          f"{n_train:>11,}  {n_test:>10,}")
