"""Tests for chronological holdout + walk-forward splitting."""

from itertools import pairwise

import pandas as pd
import pytest

from elec_forecast.splitting import holdout_start, walk_forward_folds


def _hours(n: int) -> pd.Series:
    return pd.Series(pd.date_range("2020-01-01", periods=n, freq="h"))


def test_holdout_is_exactly_the_last_n_hours():
    ts = _hours(1000)
    start = holdout_start(ts, holdout_hours=100)
    assert (ts >= start).sum() == 100
    assert start == ts.iloc[900]


def test_holdout_rejects_holding_out_everything():
    with pytest.raises(ValueError):
        holdout_start(_hours(100), holdout_hours=100)


def test_folds_are_back_to_back_and_end_at_the_end_of_dev():
    ts = _hours(1000)
    folds = walk_forward_folds(ts, n_folds=4, fold_test_hours=50, min_train_hours=500)

    assert len(folds) == 4
    for prev, nxt in pairwise(folds):
        assert nxt.origin == prev.test_end
    assert folds[-1].test_end == ts.iloc[-1] + pd.Timedelta(hours=1)


def test_every_fold_trains_only_on_hours_before_it_scores():
    ts = _hours(1000)
    for fold in walk_forward_folds(ts, n_folds=4, fold_test_hours=50, min_train_hours=500):
        train, test = ts[fold.train_mask(ts)], ts[fold.test_mask(ts)]
        assert len(test) == 50
        assert train.max() < test.min()
        assert not (fold.train_mask(ts) & fold.test_mask(ts)).any()


def test_training_window_expands_fold_by_fold():
    ts = _hours(1000)
    folds = walk_forward_folds(ts, n_folds=4, fold_test_hours=50, min_train_hours=500)
    sizes = [int(f.train_mask(ts).sum()) for f in folds]
    assert sizes == [800, 850, 900, 950]


def test_folds_refuse_to_train_on_too_little_history():
    with pytest.raises(ValueError, match="fewer than the required"):
        walk_forward_folds(_hours(1000), n_folds=4, fold_test_hours=50, min_train_hours=900)
