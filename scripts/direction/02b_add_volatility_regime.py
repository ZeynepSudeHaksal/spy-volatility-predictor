"""
02b_add_volatility_regime.py

Adds the TRAINED volatility model's predicted regime probability as a
feature for the direction model -- testing whether knowing "is this a
high-vol regime right now" carries information beyond direction's own
raw technical features (several of which, like realized_vol_20d and
vix_spy_spread, are already shared inputs -- the interesting question
is whether the volatility MODEL's refined, nonlinear combination of
those adds anything beyond the raw numbers direction already has).

REQUIRES scripts/volatility/05_train_model.py TO HAVE BEEN RUN FIRST
--------------------------------------------------------------------------
This loads models/volatility/xgboost.joblib. If it doesn't exist, this
script can't run -- see the error message below.

AN HONEST CAVEAT ABOUT THIS FEATURE, READ BEFORE INTERPRETING RESULTS
------------------------------------------------------------------------
The volatility model's predictions are computed here for EVERY date in
its own features.csv, including dates that were in ITS OWN training
set. For those dates, the regime probability is somewhat optimistic --
the model has already "seen" that exact data during its own fitting,
so its prediction there isn't genuinely out-of-sample the way a live
deployment would be. This is a real methodological wrinkle, not
something to ignore.

It matters less than it might sound, for one specific reason: both
pipelines split train/val/test using very similar date cutoffs (same
basket, same ~70/15/15 fractions), so direction's TEST period -- the
one number that actually matters for judging whether this feature
helped -- lines up closely with volatility's OWN val/test period. On
direction's test rows, the volatility model's predictions are mostly
genuinely out-of-sample for that model too. Interpret results on
direction's train period with more skepticism than results on its
test period, for exactly this reason.

Usage:
    python scripts/direction/02b_add_volatility_regime.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VOL_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "volatility"
VOL_MODELS_DIR = PROJECT_ROOT / "models" / "volatility"
DIR_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "direction"

# Reuse volatility's own feature-building logic (and its exact
# NUMERIC_FEATURE_COLS / ticker dummy columns) rather than duplicating
# it here -- guarantees the feature matrix fed to the volatility model
# is built exactly the way it was trained.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "volatility"))
import common as vol_common  # noqa: E402


def main():
    xgb_path = VOL_MODELS_DIR / "xgboost.joblib"
    if not xgb_path.exists():
        print(f"ERROR: {xgb_path} not found. Run scripts/volatility/05_train_model.py "
              f"first -- this script needs a trained volatility model.", file=sys.stderr)
        sys.exit(1)

    vol_features_path = VOL_PROCESSED_DIR / "features.csv"
    if not vol_features_path.exists():
        print(f"ERROR: {vol_features_path} not found. Run "
              f"scripts/volatility/02_engineer_features.py first.", file=sys.stderr)
        sys.exit(1)

    dir_features_path = DIR_PROCESSED_DIR / "features.csv"
    if not dir_features_path.exists():
        print(f"ERROR: {dir_features_path} not found. Run "
              f"scripts/direction/02_engineer_features.py first.", file=sys.stderr)
        sys.exit(1)

    xgb = joblib.load(xgb_path)
    with open(VOL_MODELS_DIR / "metadata.json") as f:
        import json
        vol_metadata = json.load(f)
    ticker_dummy_cols = vol_metadata["ticker_dummy_cols"]

    vol_features = pd.read_csv(vol_features_path, index_col="Date", parse_dates=True)
    X_vol, _ = vol_common.build_feature_matrix(vol_features, ticker_dummy_cols=ticker_dummy_cols)

    print(f"Scoring {len(X_vol):,} rows across {vol_features['ticker'].nunique()} "
          f"tickers with the trained volatility model...")
    vol_regime_proba = xgb.predict_proba(X_vol)[:, 1]

    regime_df = pd.DataFrame({
        "ticker": vol_features["ticker"].values,
        "Date": vol_features.index,
        "vol_regime_probability": vol_regime_proba,
    })

    dir_features = pd.read_csv(dir_features_path, index_col="Date", parse_dates=True)
    dir_features = dir_features.reset_index()

    merged = dir_features.merge(regime_df, on=["ticker", "Date"], how="left")

    n_missing = merged["vol_regime_probability"].isna().sum()
    if n_missing > 0:
        print(f"WARNING: {n_missing} rows in direction's features.csv had no matching "
              f"volatility-model date/ticker (dropping them).")
        merged = merged.dropna(subset=["vol_regime_probability"])

    merged = merged.sort_values(["Date", "ticker"]).set_index("Date")
    merged.to_csv(dir_features_path)

    print(f"\nAdded 'vol_regime_probability' to {len(merged):,} rows -> "
          f"{dir_features_path.relative_to(PROJECT_ROOT)}")
    print("\nDone. IMPORTANT: this changes the feature set -- re-run "
          "03_label_data.py through 06_evaluate.py so the new column "
          "is picked up. Also update scripts/direction/common.py's "
          "NUMERIC_FEATURE_COLS to include 'vol_regime_probability' "
          "if you haven't already.")


if __name__ == "__main__":
    main()
