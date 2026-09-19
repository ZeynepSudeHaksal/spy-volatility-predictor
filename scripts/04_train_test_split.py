"""
04_train_test_split.py

Splits labeled_data.csv into train/validation/test sets -- chronologically,
NOT randomly. This is the single most important methodological choice in
the whole project, so it's worth spelling out why.

THE PROBLEM WITH RANDOM K-FOLD ON FINANCIAL TIME SERIES
---------------------------------------------------------
A standard `train_test_split(shuffle=True)` or k-fold CV assumes rows are
independent. Ours are not, in two compounding ways:

1. The FEATURES are autocorrelated. realized_vol_20d on Tuesday and
   Wednesday share 19 of the same 20 days of returns. If a Tuesday row
   lands in training and the near-identical Wednesday row lands in
   test, the model isn't really being tested on unseen data -- it's
   nearly memorizing a neighbor.

2. The LABELS look into the future. forward_vol_5d(t) uses returns from
   t+1 to t+5. If row t is in the training set and row t+3 is in the
   test set, then t's label was partly computed from data that
   "belongs" to the test period -- the model would be trained on
   information it shouldn't have had access to at decision time. This
   is a subtler, easier-to-miss version of the same leakage problem.

A model evaluated under random k-fold on this kind of data will report
inflated performance that will NOT hold up on genuinely new, future
data -- which defeats the entire point of the project.

THE FIX: CHRONOLOGICAL SPLIT + PURGE GAP
------------------------------------------
- Split by DATE, not randomly: train = earliest ~70%, val = next ~15%,
  test = most recent ~15%. This mirrors how the model would actually be
  used (train on the past, predict the future).
- Additionally PURGE a gap of FORWARD_WINDOW trading days at each split
  boundary. Without this gap, the last few rows of train would have
  labels computed using return data that falls inside the val period --
  a boundary leak that's easy to miss but real. Purging removes any
  training row whose label window (t+1..t+5) could overlap the next
  split.

Usage:
    python scripts/04_train_test_split.py
    python scripts/04_train_test_split.py --train-frac 0.7 --val-frac 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

FORWARD_WINDOW = 5  # must match 03_label_data.py -- the purge gap size


def load_labeled_data() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "labeled_data.csv"
    if not path.exists():
        print(
            "ERROR: labeled_data.csv not found. Run 03_label_data.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def chronological_split_with_purge(
    df: pd.DataFrame, train_frac: float, val_frac: float, purge: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits df (already sorted by date) into train/val/test by position,
    then removes `purge` rows from the end of train and the end of val
    so no training/validation row's label window can overlap the next
    split's time period.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    # Purge: drop the last `purge` rows of train and val so their label
    # windows (t+1..t+FORWARD_WINDOW) can't reach into the next split.
    train = train.iloc[: len(train) - purge] if len(train) > purge else train.iloc[0:0]
    val = val.iloc[: len(val) - purge] if len(val) > purge else val.iloc[0:0]

    return train, val, test


def summarize_split(name: str, split_df: pd.DataFrame) -> None:
    if len(split_df) == 0:
        print(f"{name}: EMPTY (check your fractions / purge size)")
        return
    class_counts = split_df["label"].value_counts().sort_index()
    high_pct = (class_counts.get(1, 0) / len(split_df) * 100)
    print(
        f"{name}: {len(split_df):>4,} rows | "
        f"{split_df.index.min().date()} to {split_df.index.max().date()} | "
        f"high-vol regime: {high_pct:.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(description="Chronological walk-forward split.")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    if args.train_frac + args.val_frac >= 1.0:
        print("ERROR: train-frac + val-frac must be < 1.0 (remainder is test).",
              file=sys.stderr)
        sys.exit(1)

    df = load_labeled_data().sort_index()

    train, val, test = chronological_split_with_purge(
        df, args.train_frac, args.val_frac, purge=FORWARD_WINDOW
    )

    print("Split summary (chronological, purge gap of "
          f"{FORWARD_WINDOW} days at each boundary):\n")
    summarize_split("Train", train)
    summarize_split("Val  ", val)
    summarize_split("Test ", test)

    train.to_csv(PROCESSED_DATA_DIR / "train.csv")
    val.to_csv(PROCESSED_DATA_DIR / "val.csv")
    test.to_csv(PROCESSED_DATA_DIR / "test.csv")

    print(f"\nSaved train.csv, val.csv, test.csv -> "
          f"{PROCESSED_DATA_DIR.relative_to(PROJECT_ROOT)}")

    print(
        "\nNote: it's normal (and expected, not a bug) for the class balance "
        "and general vol level to differ noticeably between splits -- e.g. "
        "if your test period happens to land on a genuinely calmer or more "
        "turbulent stretch of market history than train. That's a real "
        "property of financial data, not something to 'fix' by shuffling; "
        "it's exactly the kind of regime shift the model needs to be "
        "evaluated against honestly."
    )

    print("\nDone. Next: run 05_train_model.py")


if __name__ == "__main__":
    main()
