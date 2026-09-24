"""Chronological splitting: one final holdout block plus walk-forward folds.

Nothing here is random. Every boundary is a timestamp, and every training
set ends strictly before the data it's scored on. Randomly splitting hours
from a time series would put tomorrow in the training set and today in the
test set, which scores the model on a problem it will never face.

Layout, oldest to newest:

    |------------------ dev ------------------|---- holdout ----|
    | train (expanding) ...  | fold k test    |                 |

- Holdout: the last `holdout_hours` hours. Scored exactly once, at the end.
- Folds: `n_folds` back-to-back test windows at the end of dev. Fold k
  trains on every dev hour before its origin (an expanding window) and is
  scored on the `fold_test_hours` hours that follow.

Within a fold's test window, the model is frozen but its inputs keep
arriving: the features for each hour use observed data up to 24h earlier
(see features.py). That matches running a monthly-retrained model that
issues a fresh 24h forecast every day.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    index: int
    origin: pd.Timestamp  # first scored hour; training uses only hours before this
    test_end: pd.Timestamp  # exclusive

    def train_mask(self, timestamps: pd.Series) -> pd.Series:
        return timestamps < self.origin

    def test_mask(self, timestamps: pd.Series) -> pd.Series:
        return (timestamps >= self.origin) & (timestamps < self.test_end)


def holdout_start(timestamps: pd.Series, holdout_hours: int) -> pd.Timestamp:
    """Return the first timestamp of the holdout block (the last `holdout_hours` hours)."""
    ts = pd.DatetimeIndex(timestamps).sort_values()
    if len(ts) <= holdout_hours:
        raise ValueError(f"only {len(ts):,} hours of data, can't hold out {holdout_hours:,}")
    return ts[-holdout_hours]


def walk_forward_folds(
    dev_timestamps: pd.Series,
    n_folds: int,
    fold_test_hours: int,
    min_train_hours: int,
) -> list[Fold]:
    """Build `n_folds` consecutive test windows ending at the end of dev."""
    ts = pd.DatetimeIndex(dev_timestamps).sort_values()
    dev_end = ts[-1] + pd.Timedelta(hours=1)  # exclusive
    step = pd.Timedelta(hours=fold_test_hours)

    folds = []
    for k in range(n_folds):
        origin = dev_end - (n_folds - k) * step
        n_train = int((ts < origin).sum())
        if n_train < min_train_hours:
            raise ValueError(
                f"fold {k} would train on {n_train:,} hours, fewer than the required "
                f"{min_train_hours:,}; use fewer or shorter folds"
            )
        folds.append(Fold(index=k, origin=origin, test_end=origin + step))
    return folds
