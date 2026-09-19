"""
04_train_test_split.py

Splits labeled_data.csv into train/validation/test sets -- chronologically
by DATE CUTOFF, applied identically across every ticker in the basket,
NOT randomly and not by row-count position per ticker.

WHY DATE CUTOFFS RATHER THAN PER-TICKER ROW-COUNT SPLITS
--------------------------------------------------------------
labeled_data.csv is a long panel (one row per ticker per date). Splitting
by row *position* independently per ticker could let SPY's train set
extend into 2023 while QQQ's train set only reaches 2022, if their row
counts differ even slightly (e.g. a data gap). Using the same DATE
cutoffs for every ticker keeps train/val/test periods perfectly aligned
across the whole basket -- essential for a fair, honest evaluation, and
for the "regime shift" property (a test period landing on a calmer or
more turbulent stretch) to mean the same thing for every ticker.

WHY RANDOM K-FOLD IS STILL WRONG HERE (see 03_label_data.py's docstring
for the full explanation of return autocorrelation and label lookahead)
-- applies per ticker exactly as it did for the single-ticker version.

THE PURGE GAP, NOW PER TICKER
----------------------------------
Same leakage risk as before: a training row's label looks FORWARD_WINDOW
days into the future, so rows too close to a split boundary need to be
dropped. This must be done separately for each ticker (a ticker's last
FORWARD_WINDOW rows before the boundary, in ITS OWN date sequence -- not
the basket's row count), using groupby('ticker').tail(), which selects
actual rows by position within each group and is safe regardless of any
duplicate-date index elsewhere in the frame.

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
        print("ERROR: labeled_data.csv not found. Run 03_label_data.py first.",
              file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def compute_date_cutoffs(all_dates: pd.Series, train_frac: float, val_frac: float):
    unique_dates = sorted(all_dates.unique())
    n = len(unique_dates)
    train_end_date = unique_dates[int(n * train_frac) - 1]
    val_end_date = unique_dates[int(n * (train_frac + val_frac)) - 1]
    return train_end_date, val_end_date


def purge_boundary(df: pd.DataFrame, purge: int) -> pd.DataFrame:
    """Drops the chronologically-last `purge` rows of each ticker in df.
    Uses groupby().tail(), which selects actual rows by position within
    each group -- safe regardless of duplicate date labels elsewhere."""
    df_sorted = df.sort_values(["ticker", "Date"])
    to_drop = df_sorted.groupby("ticker", group_keys=False).tail(purge)
    return df_sorted.drop(to_drop.index)


def summarize_split(name: str, split_df: pd.DataFrame) -> None:
    if len(split_df) == 0:
        print(f"{name}: EMPTY (check your fractions / purge size)")
        return
    print(f"{name}: {len(split_df):>5,} rows total | "
          f"{split_df['Date'].min().date()} to {split_df['Date'].max().date()}")
    for ticker, group in split_df.groupby("ticker"):
        high_pct = (group["label"] == 1).mean() * 100
        print(f"    {ticker:>4}: {len(group):>4,} rows | high-vol: {high_pct:5.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Chronological walk-forward split.")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    if args.train_frac + args.val_frac >= 1.0:
        print("ERROR: train-frac + val-frac must be < 1.0 (remainder is test).",
              file=sys.stderr)
        sys.exit(1)

    df = load_labeled_data().reset_index()  # 'Date' becomes a plain column

    train_end_date, val_end_date = compute_date_cutoffs(
        df["Date"], args.train_frac, args.val_frac
    )

    train = df[df["Date"] <= train_end_date]
    val = df[(df["Date"] > train_end_date) & (df["Date"] <= val_end_date)]
    test = df[df["Date"] > val_end_date]

    train = purge_boundary(train, FORWARD_WINDOW)
    val = purge_boundary(val, FORWARD_WINDOW)

    print("Split summary (chronological by date cutoff, purge gap of "
          f"{FORWARD_WINDOW} days per ticker at each boundary):\n")
    summarize_split("Train", train)
    summarize_split("Val", val)
    summarize_split("Test", test)

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        out = split_df.sort_values(["Date", "ticker"]).set_index("Date")
        out.to_csv(PROCESSED_DATA_DIR / f"{name}.csv")

    print(f"\nSaved train.csv, val.csv, test.csv -> "
          f"{PROCESSED_DATA_DIR.relative_to(PROJECT_ROOT)}")

    print(
        "\nNote: it's normal (and expected, not a bug) for class balance and "
        "vol level to differ across tickers and splits -- that reflects real "
        "differences (IWM/small-cap runs structurally hotter than DIA/blue-chip) "
        "and real regime shifts over time, not something to 'fix' by shuffling."
    )

    print("\nDone. Next: run 05_train_model.py")


if __name__ == "__main__":
    main()
