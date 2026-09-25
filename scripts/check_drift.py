"""Phase 16 -- drift check demonstration.

Runs the drift detector on real project data: the dev feature set
(everything the model was trained/backtested on) as the reference, the
90-day holdout block as the "current" batch.

The holdout isn't drawn from a different population than dev -- it's just
the last 90 days of the same household -- so this isn't expected to fire.
It's here to run the check end-to-end against real data the way it would
run in a real serving loop, not to claim genuine drift monitoring against
live traffic that doesn't exist yet (see README "Monitoring").

Run: python scripts/check_drift.py
"""

import yaml

from elec_forecast.features import FEATURE_COLUMNS
from elec_forecast.monitoring import compute_reference_stats, drift_report
from elec_forecast.pipeline import prepare_backtest

with open("configs/backtest.yaml") as f:
    backtest_config = yaml.safe_load(f)

data = prepare_backtest(backtest_config)
reference = compute_reference_stats(data.dev, list(FEATURE_COLUMNS))
report = drift_report(data.holdout, reference)

print(f"Reference: {len(data.dev):,} dev hours")
print(f"Current:   {len(data.holdout):,} holdout hours\n")
for feature, stats in report["per_feature"].items():
    flag = "DRIFTED" if stats["drifted"] else "ok"
    print(f"  {feature:<16} z={stats['z_score']:+.2f}  {flag}")
print(f"\nany_drifted: {report['any_drifted']}")