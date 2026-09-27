"""
cv_utils.py

Shared purged walk-forward cross-validation split generator, used by
both pipelines' 05a_tune_hyperparameters.py scripts.

Centralized here specifically because this exact class of bug -- panel
data (multiple tickers sharing a Date index) combined with groupby or
index-based reassembly -- has bitten this project before. See
scripts/volatility/03_label_data.py's docstring for the full story of
a real, confirmed bug where duplicate date labels across tickers
caused values to silently swap between tickers. Having ONE
well-tested implementation that both pipelines import, rather than two
independently written copies, means a single point to get right and a
single point to fix if something is ever found wrong.

WHAT THIS PROVIDES
----------------------
purged_walk_forward_splits() generates several (train_idx, val_idx)
pairs, each fold's training window EXPANDING to include everything
before that fold's validation window -- mirroring how the model would
actually be retrained in practice (on everything seen so far), not a
fixed-size rolling window.

Plain time-based CV (e.g. sklearn's TimeSeriesSplit without a gap)
prevents shuffling leakage but not PURGE-GAP leakage: a training row
near a fold boundary can still have a label that peeks into the
validation chunk (this project's labels look forward FORWARD_WINDOW
days). This function purges the last `purge_gap` rows of EACH
TICKER's training data -- by that ticker's own chronological order,
not global row count -- before each validation window, exactly
mirroring scripts/volatility/04_train_test_split.py's purge_boundary()
logic, just generalized to multiple folds.

SAFE INDEXING PATTERN -- READ BEFORE MODIFYING
----------------------------------------------------
This function returns INTEGER POSITIONS only (via np.where-style
boolean masks and groupby().tail(), which selects rows by position
within each group). It never does label-based indexing or assignment
on a frame with duplicate index labels -- the exact pattern that
caused the earlier bug. The caller is expected to pass `dates` and
`ticker` as plain pd.Series sharing the same row order as whatever
DataFrame the returned positions will be applied to (typically via
.iloc[] or array indexing, not .loc[]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def purged_walk_forward_splits(
    dates: pd.Series,
    n_splits: int,
    purge_gap: int,
    ticker: pd.Series,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Returns a list of (train_idx, val_idx) integer-position arrays,
    suitable for passing directly as the `cv` argument to sklearn's
    GridSearchCV / RandomizedSearchCV.
    """
    n = len(dates)
    if len(ticker) != n:
        raise ValueError("dates and ticker must be the same length.")

    positions = np.arange(n)
    dates_arr = pd.Series(dates).reset_index(drop=True)
    ticker_arr = pd.Series(ticker).reset_index(drop=True)

    unique_dates = np.array(sorted(dates_arr.unique()))
    n_dates = len(unique_dates)

    # +1 reserves the first chunk as the initial training window before
    # any validation fold begins.
    fold_size = n_dates // (n_splits + 1)
    if fold_size < 1:
        raise ValueError(
            f"Not enough unique dates ({n_dates}) for {n_splits} folds. "
            f"Reduce n_splits or provide more data."
        )

    splits = []
    for i in range(1, n_splits + 1):
        train_end_idx = fold_size * i
        val_end_idx = fold_size * (i + 1) if i < n_splits else n_dates

        train_end_date = unique_dates[train_end_idx - 1]
        val_start_date = unique_dates[train_end_idx]
        val_end_date = unique_dates[val_end_idx - 1]

        train_mask = (dates_arr.values <= train_end_date)
        val_mask = (dates_arr.values >= val_start_date) & (dates_arr.values <= val_end_date)

        if purge_gap > 0:
            train_positions = positions[train_mask]
            fold_df = pd.DataFrame({
                "pos": train_positions,
                "ticker": ticker_arr.values[train_mask],
                "date": dates_arr.values[train_mask],
            }).sort_values(["ticker", "date"])
            to_purge = fold_df.groupby("ticker", group_keys=False).tail(purge_gap)
            purge_positions = set(to_purge["pos"].values)
            train_idx = np.array(
                [p for p in train_positions if p not in purge_positions]
            )
        else:
            train_idx = positions[train_mask]

        val_idx = positions[val_mask]

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        splits.append((train_idx, val_idx))

    return splits


def rolling_walk_forward_splits(
    dates: pd.Series,
    window_years: float,
    step_months: float,
    purge_gap: int,
    ticker: pd.Series,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Like purged_walk_forward_splits above, but each fold's training
    window is a FIXED-SIZE TRAILING window (rolling), not expanding to
    include all history. Old data is discarded as folds advance --
    exactly the property needed to test or mitigate regime drift: a
    model trained on only the most recent `window_years` of history,
    not diluted by decades of older, possibly no-longer-relevant market
    conditions.

    Each fold: train on [test_date - window_years, test_date - purge_gap],
    test on [test_date, test_date + step_months). Folds step forward by
    step_months each time, walking through the full available history.

    Same safe integer-position-only design as purged_walk_forward_splits
    -- see that function's docstring for the full rationale.
    """
    n = len(dates)
    if len(ticker) != n:
        raise ValueError("dates and ticker must be the same length.")

    positions = np.arange(n)
    dates_arr = pd.Series(dates).reset_index(drop=True)
    ticker_arr = pd.Series(ticker).reset_index(drop=True)

    unique_dates = np.array(sorted(dates_arr.unique()))
    window = pd.Timedelta(days=int(window_years * 365.25))
    step = pd.Timedelta(days=int(step_months * 30.44))

    first_date = unique_dates[0]
    last_date = unique_dates[-1]

    splits = []
    test_start = pd.Timestamp(first_date) + window
    while test_start <= pd.Timestamp(last_date):
        test_end = test_start + step
        train_start = test_start - window
        train_end = test_start  # purging below trims the boundary

        train_mask = (dates_arr.values >= train_start.to_datetime64()) & \
                     (dates_arr.values < train_end.to_datetime64())
        test_mask = (dates_arr.values >= test_start.to_datetime64()) & \
                    (dates_arr.values < test_end.to_datetime64())

        if purge_gap > 0:
            train_positions = positions[train_mask]
            fold_df = pd.DataFrame({
                "pos": train_positions,
                "ticker": ticker_arr.values[train_mask],
                "date": dates_arr.values[train_mask],
            }).sort_values(["ticker", "date"])
            to_purge = fold_df.groupby("ticker", group_keys=False).tail(purge_gap)
            purge_positions = set(to_purge["pos"].values)
            train_idx = np.array(
                [p for p in train_positions if p not in purge_positions]
            )
        else:
            train_idx = positions[train_mask]

        test_idx = positions[test_mask]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx, test_start, test_end))

        test_start = test_start + step

    return splits
