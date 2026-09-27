"""
03_label_data.py

Builds the classification target: for each (ticker, date), was the
5-day forward move UP or DOWN -- but only for moves big enough, relative
to that ticker's own recent volatility, to be a real signal rather than
noise.

WHY VOLATILITY-ADJUSTED, NOT RAW SIGN
------------------------------------------
A 1% move means something very different on a calm ticker than on a
turbulent one. Comparing raw returns across tickers (or across calm vs.
turbulent periods for the SAME ticker) treats these as equivalent when
they're not. Instead:

    five_day_vol_estimate(t) = realized_vol_20d(t) * sqrt(5/252)

    -- realized_vol_20d is the same backward-looking, no-leakage column
    already in features.csv (built in 02_engineer_features.py from data
    up to t). Annualized vol is converted to an EXPECTED 5-day move size
    via sqrt(5/252), the standard square-root-of-time scaling.

    z(t) = forward_return_5d(t) / five_day_vol_estimate(t)

    -- how large the ACTUAL forward move was, in units of "how big a
    move normally looks like for this ticker right now."

WHY DROP THE AMBIGUOUS MIDDLE
----------------------------------
Forcing every single day into up/down -- including days where the
5-day move was essentially noise around zero -- teaches the model to
memorize noise on exactly the days that were never callable in the
first place. Rows where |z| < THRESHOLD_Z are dropped from BOTH
training and evaluation: label = 1 if z > threshold, 0 if z <
-threshold, and the row simply doesn't exist otherwise. This is a
deliberate choice to make the problem the model needs to solve more
learnable, not a way to inflate accuracy -- evaluation still only ever
happens on genuinely-labeled rows, same as before.

SAME SAFE PATTERN AS BEFORE (unique row id before groupby+apply
realignment) -- see the volatility pipeline's 03_label_data.py for the
full history of why this matters.

Usage:
    python scripts/direction/03_label_data.py
    python scripts/direction/03_label_data.py --threshold-z 0.35
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "direction"

FORWARD_WINDOW = 5
TRADING_DAYS_PER_YEAR = 252
DEFAULT_THRESHOLD_Z = 0.25  # drop moves smaller than a quarter of a "typical" move


def load_features() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "features.csv"
    if not path.exists():
        print("ERROR: features.csv not found. Run 02_engineer_features.py "
              "(and 02b_add_volatility_regime.py, if using it) first.",
              file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    if "realized_vol_20d" not in df.columns:
        print("ERROR: features.csv is missing realized_vol_20d, needed for "
              "volatility-adjusted labeling.", file=sys.stderr)
        sys.exit(1)
    return df


def main():
    parser = argparse.ArgumentParser(description="Volatility-adjusted direction labeling.")
    parser.add_argument("--threshold-z", type=float, default=DEFAULT_THRESHOLD_Z,
                         help="Drop rows where |forward move / typical move| is below this.")
    args = parser.parse_args()

    df = load_features()

    # --- SAFE PATTERN: unique row id before any groupby+apply realignment ---
    df = df.reset_index()
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    grouped = df.groupby("ticker", group_keys=False)
    df["forward_return_5d"] = grouped["Close"].apply(
        lambda s: s.shift(-FORWARD_WINDOW) / s - 1
    )
    # --- end safe pattern ---

    df["five_day_vol_estimate"] = df["realized_vol_20d"] * np.sqrt(FORWARD_WINDOW / TRADING_DAYS_PER_YEAR)
    df["forward_move_z"] = df["forward_return_5d"] / df["five_day_vol_estimate"]

    n_before_dropna = len(df)
    df = df.dropna(subset=["forward_return_5d", "five_day_vol_estimate"])
    n_after_forward_window = len(df)
    print(f"Dropped {n_before_dropna - n_after_forward_window} rows without a valid "
          f"forward window or vol estimate (per ticker).")

    is_up = df["forward_move_z"] > args.threshold_z
    is_down = df["forward_move_z"] < -args.threshold_z
    ambiguous = ~(is_up | is_down)

    n_ambiguous = ambiguous.sum()
    print(f"Dropped {n_ambiguous:,} rows as ambiguous (|z| <= {args.threshold_z}) "
          f"out of {n_after_forward_window:,} -- these moves weren't distinguishable "
          f"from normal noise for that ticker at that time.")

    df = df[~ambiguous].copy()
    df["label"] = is_up[~ambiguous].astype(int)

    df = df.sort_values(["Date", "ticker"]).set_index("Date")

    out_path = PROCESSED_DATA_DIR / "labeled_data.csv"
    df.to_csv(out_path)
    print(f"\nSaved {len(df):,} labeled rows -> {out_path.relative_to(PROJECT_ROOT)} "
          f"(threshold_z={args.threshold_z})")

    print("\nUp-rate by ticker, among rows that cleared the threshold:")
    for ticker, group in df.groupby("ticker"):
        up_pct = (group["label"] == 1).mean() * 100
        print(f"  {ticker:>4}: {len(group):>5,} rows | up {up_pct:5.1f}% of the time")

    overall_up_pct = (df["label"] == 1).mean() * 100
    print(f"\nPooled up-rate: {overall_up_pct:.1f}% -- any model needs to clear THIS "
          f"bar (not 50%) on this filtered set to demonstrate real skill. Note this "
          f"baseline is computed fresh on the FILTERED data downstream (04/05/06), "
          f"not carried over from an unfiltered run.")

    print("\nDone. Next: run 04_train_test_split.py")


if __name__ == "__main__":
    main()
