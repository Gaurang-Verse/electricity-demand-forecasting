"""Phase 16 -- rolling forecast-error tracking demonstration.

Applies rolling_forecast_error() to real backtest predictions -- the
shape a real deployment would log continuously as true values arrive
(datetime, actual, predicted), applied here once, after the fact, to real
project data as a demonstration of the mechanism, not live traffic (see
README "Monitoring").

This re-runs the backtest (retrains all 12 folds) to get per-hour
predictions, since scripts/run_backtest.py only saves the summarized
metrics, not the row-level predictions this needs.

Run: python scripts/monitor_forecast_errors.py
"""

import yaml

from elec_forecast.backtest import run_backtest
from elec_forecast.models import make_model
from elec_forecast.monitoring import rolling_forecast_error
from elec_forecast.pipeline import prepare_backtest

with open("configs/backtest.yaml") as f:
    backtest_config = yaml.safe_load(f)
with open("configs/models.yaml") as f:
    model_params = yaml.safe_load(f)["models"].get("lightgbm", {})

data = prepare_backtest(backtest_config)
predictions = run_backtest(data, lambda: make_model("lightgbm", model_params))

window_hours = 24 * 7  # 1 week
rolling = rolling_forecast_error(predictions, window=window_hours)

print(f"Rolling {window_hours}h (7-day) forecast error over the backtest window:\n")
print(rolling.dropna().tail(10).to_string(index=False))