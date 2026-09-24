"""Forecast error metrics.

All errors are predicted minus actual, so a positive bias means the model
over-forecasts on average.

MAPE divides by the actual value, so it's only defined when every actual
is positive. This data never hits zero (the lowest hourly mean is a
household's standby draw), but the function checks rather than assumes,
because one near-zero hour would otherwise dominate the average silently.
"""

import numpy as np


def regression_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")
    if (actual <= 0).any():
        raise ValueError("MAPE is undefined when an actual value is zero or negative")

    err = predicted - actual
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / actual) * 100),
        "bias": float(np.mean(err)),
        "n": len(actual),
    }
