"""
03_label_data.py

Builds the classification target: for each trading day t, was the
*forward-looking* 5-day realized volatility "high" (1) or "low" (0)
relative to a trailing baseline?

Two pieces, computed separately and deliberately kept apart:

1. forward_vol_5d(t): the actual future outcome we're trying to predict.
   Computed from log returns at t+1, t+2, t+3, t+4, t+5 -- i.e. it uses
   data that would NOT be available at prediction time t. This is
   correct and necessary for a *label* (we're teaching the model what
   actually happened next), but it must never leak into the *features*
   file. It doesn't: 02_engineer_features.py only ever looks backward.

2. The regime threshold: a trailing 60-day median of realized_vol_20d
   (the backward-looking 20-day realized vol column from the features
   file). This uses ONLY information available at or before time t, so
   comparing forward_vol_5d(t) against it is a leakage-safe way to
   define "high" vs. "low" relative to *recent* conditions, rather than
   against some fixed threshold that would drift out of relevance
   across a multi-year, multi-regime dataset.

Why 20-day realized vol (not the forward target's own history) as the
baseline: it's already available with zero look-ahead risk, and the
20-day window is smooth enough to represent "what's been normal lately"
without over-reacting to single-day noise the way a 5-day window would.

Label: 1 = "high vol regime" (forward_vol_5d > trailing median),
       0 = "low vol regime" (forward_vol_5d <= trailing median)

Usage:
    python scripts/03_label_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

FORWARD_WINDOW = 5          # trading days ahead to measure realized vol over
TRAILING_MEDIAN_WINDOW = 60  # trading days of history to define "normal"
TRADING_DAYS_PER_YEAR = 252


def load_features() -> pd.DataFrame:
    features_path = PROCESSED_DATA_DIR / "features.csv"
    if not features_path.exists():
        print(
            "ERROR: features.csv not found. Run 02_engineer_features.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return pd.read_csv(features_path, index_col="Date", parse_dates=True)


def compute_forward_realized_vol(log_return: pd.Series, window: int) -> pd.Series:
    """
    For each t, computes the annualized std of log returns at
    t+1, ..., t+window -- the actual future volatility outcome.

    Built explicitly via shift(-1)...shift(-window) rather than a
    reversed rolling window, so the direction is unambiguous and easy
    to verify by eye.
    """
    future_returns = pd.concat(
        [log_return.shift(-i) for i in range(1, window + 1)], axis=1
    )
    return future_returns.std(axis=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def main():
    df = load_features()

    df["forward_vol_5d"] = compute_forward_realized_vol(
        df["log_return"], FORWARD_WINDOW
    )

    # Trailing baseline: rolling median of the backward-looking 20-day
    # realized vol column already in features.csv. Uses only data at or
    # before t -- no leakage.
    df["trailing_vol_median_60d"] = (
        df["realized_vol_20d"].rolling(TRAILING_MEDIAN_WINDOW).median()
    )

    df["label"] = (df["forward_vol_5d"] > df["trailing_vol_median_60d"]).astype(int)

    n_before = len(df)
    # Drop rows where we can't compute the label:
    # - last FORWARD_WINDOW rows: no future returns to look ahead to
    # - first TRAILING_MEDIAN_WINDOW-ish rows: not enough history for the
    #   trailing median yet
    df = df.dropna(subset=["forward_vol_5d", "trailing_vol_median_60d"])
    n_after = len(df)
    print(
        f"Dropped {n_before - n_after} rows without a valid label "
        f"(no forward window and/or insufficient trailing history)."
    )

    out_path = PROCESSED_DATA_DIR / "labeled_data.csv"
    df.to_csv(out_path)

    print(f"Saved {len(df):,} labeled rows -> {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")

    class_counts = df["label"].value_counts().sort_index()
    class_pct = (class_counts / len(df) * 100).round(1)
    print("\nClass balance:")
    print(f"  0 (low vol regime):  {class_counts.get(0, 0):>5,} rows ({class_pct.get(0, 0)}%)")
    print(f"  1 (high vol regime): {class_counts.get(1, 0):>5,} rows ({class_pct.get(1, 0)}%)")

    if class_pct.min() < 30:
        print(
            "\nNote: classes are somewhat imbalanced. Worth using "
            "class_weight='balanced' (or similar) at training time, and "
            "leaning on precision/recall/ROC-AUC over raw accuracy when "
            "evaluating -- expected given how the label is defined "
            "(median-based thresholds should land close to 50/50, but "
            "regime persistence can skew this)."
        )

    print("\nDone. Next: run 04_train_test_split.py")


if __name__ == "__main__":
    main()
