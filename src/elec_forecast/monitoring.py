"""Lightweight monitoring: feature drift detection and rolling
forecast-error tracking.

This project has no live production traffic (see README), so these are
demonstrated against real backtest/holdout data and validated against
synthetic data -- but the checks themselves are the ones a real
deployment would run, and they're validated the way any statistical
check should be: a null test proving the drift detector stays quiet on
data from the same distribution as the reference, and a positive control
proving it actually fires when the distribution has genuinely shifted.
Either alone is not enough -- a detector that never fires passes a null
test trivially, and one that always fires passes a positive control
trivially. Same discipline tests/test_features_leakage.py already
applies to the leakage check.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ReferenceStats:
    """Per-column mean/std computed once from a trusted reference dataset
    (e.g. the dev/training features), to compare later batches against."""

    means: dict[str, float]
    stds: dict[str, float]
    columns: list[str]


def compute_reference_stats(reference: pd.DataFrame, columns: list[str]) -> ReferenceStats:
    means = {c: float(reference[c].mean()) for c in columns}
    stds = {c: float(reference[c].std()) for c in columns}
    zero_std = [c for c, s in stds.items() if s == 0]
    if zero_std:
        raise ValueError(
            f"columns with zero variance in the reference set can't be z-scored: {zero_std}"
        )
    return ReferenceStats(means=means, stds=stds, columns=columns)


def drift_report(
    current: pd.DataFrame, reference: ReferenceStats, z_threshold: float = 3.0
) -> dict:
    """For each reference column, compare `current`'s mean against the
    reference mean, expressed as a z-score using the reference std. A
    column is flagged drifted if |z| exceeds `z_threshold`.

    This catches a shift in the *average* value of a feature between two
    batches -- coarser than a full distributional test (e.g. PSI or a
    KS test), but simple, fast, and enough to catch the kind of drift
    that matters most for a tabular forecasting model: the input
    population has genuinely shifted, not just gotten noisier.
    """
    if len(current) == 0:
        raise ValueError("current batch is empty, nothing to compare")

    per_feature = {}
    for c in reference.columns:
        current_mean = float(current[c].mean())
        z = (current_mean - reference.means[c]) / reference.stds[c]
        per_feature[c] = {
            "reference_mean": reference.means[c],
            "current_mean": current_mean,
            "z_score": z,
            "drifted": abs(z) > z_threshold,
        }
    return {
        "z_threshold": z_threshold,
        "n_current": len(current),
        "per_feature": per_feature,
        "any_drifted": any(v["drifted"] for v in per_feature.values()),
    }


def rolling_forecast_error(predictions: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """Given a time-ordered log of (datetime, actual, predicted) rows --
    the shape scripts/run_backtest.py already produces, and what a real
    deployment would log per served forecast once the true value is
    observed -- return a rolling RMSE/MAE over the last `window` rows,
    one value per row (NaN until `window` rows have accumulated).

    This is the tracking primitive: plug in real logged (actual,
    predicted) pairs as they arrive and this reports whether recent
    error is drifting away from the backtested baseline, without
    waiting for a full new evaluation cycle.
    """
    required = {"datetime", "actual", "predicted"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing required columns: {missing}")

    ordered = predictions.sort_values("datetime").reset_index(drop=True)
    error = ordered["actual"] - ordered["predicted"]
    rolling_rmse = np.sqrt((error**2).rolling(window=window).mean())
    rolling_mae = error.abs().rolling(window=window).mean()
    return pd.DataFrame(
        {
            "datetime": ordered["datetime"],
            "rolling_rmse": rolling_rmse,
            "rolling_mae": rolling_mae,
        }
    )