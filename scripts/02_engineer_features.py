"""
02_engineer_features.py

Builds the feature set for the volatility regime classifier from the raw
SPY and VIX CSVs produced by 01_fetch_data.py.

Feature groups (and why each one is here):

1. Realized volatility (multiple lookback windows)
   The single most important input. Volatility clusters over time, so
   *recent* realized vol is the strongest predictor of *near-future*
   realized vol. Using three windows (5/10/20-day) instead of one lets
   the model see both fast-moving and slower-moving vol trends -- a
   spike in 5-day vol that hasn't shown up in 20-day vol yet is itself
   informative (it suggests a regime change may be starting).

2. ATR (Average True Range)
   A classic technical-analysis volatility measure, but it captures
   something realized volatility (built from close-to-close returns)
   misses: intraday range. A day that opens, spikes, and closes flat
   looks calm in close-to-close terms but was not calm to trade through.

3. Volume relative to its own moving average
   Volume spikes often precede or accompany volatility spikes (this is
   a well-known market microstructure pattern). We use a *relative*
   measure (today's volume / 20-day average volume) rather than raw
   volume, because raw share volume trends upward over decades as
   markets grow -- a relative measure stays meaningful across the
   full multi-year dataset.

4. VIX level and VIX-SPY spread
   VIX is the market's own forward-looking volatility estimate (it's
   literally derived from S&P 500 option prices). Including it isn't
   circular reasoning -- we're not predicting VIX itself, we're using
   it as one input among several to predict SPY's own forward realized
   volatility. The "spread" (VIX minus SPY's trailing realized vol)
   captures whether the market is currently pricing in MORE fear than
   recent price action would justify, which is itself a regime signal.

5. Return magnitude (not direction)
   |daily return| is a simple, direct volatility proxy at the single-day
   level, complementing the rolling-window measures above.

Usage:
    python scripts/02_engineer_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

REALIZED_VOL_WINDOWS = [5, 10, 20]   # trading days
ATR_WINDOW = 14                       # standard ATR lookback
VOLUME_MA_WINDOW = 20
TRADING_DAYS_PER_YEAR = 252


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw SPY and VIX CSVs produced by 01_fetch_data.py."""
    spy_path = RAW_DATA_DIR / "spy_raw.csv"
    vix_path = RAW_DATA_DIR / "vix_raw.csv"

    if not spy_path.exists() or not vix_path.exists():
        print(
            "ERROR: raw data not found. Run 01_fetch_data.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    spy = pd.read_csv(spy_path, index_col="Date", parse_dates=True)
    vix = pd.read_csv(vix_path, index_col="Date", parse_dates=True)
    return spy, vix


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log returns and absolute return magnitude."""
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["abs_return"] = df["log_return"].abs()
    return df


def add_realized_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling realized volatility, annualized, for each window in
    REALIZED_VOL_WINDOWS. Annualizing (multiplying by sqrt(252)) puts
    the numbers on a familiar "annualized vol %" scale (e.g. 0.15 =
    15% annualized vol), which is standard in finance and makes the
    features directly interpretable later during SHAP analysis.
    """
    for window in REALIZED_VOL_WINDOWS:
        col = f"realized_vol_{window}d"
        df[col] = (
            df["log_return"].rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    return df


def add_atr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Average True Range over ATR_WINDOW days, plus ATR as a % of price
    so it's comparable across different price levels over the years."""
    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.rolling(ATR_WINDOW).mean()
    df["atr_pct"] = df["atr"] / df["Close"]
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume relative to its own trailing moving average."""
    volume_ma = df["Volume"].rolling(VOLUME_MA_WINDOW).mean()
    df["volume_ratio"] = df["Volume"] / volume_ma
    return df


def add_vix_features(spy: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """
    Merge in VIX level and compute the VIX-SPY spread: VIX (as a
    fraction, e.g. 0.20 for "20") minus SPY's own trailing 20-day
    realized vol. A large positive spread means the market is pricing
    in more forward fear than recent realized moves would suggest.
    """
    vix_close = vix[["Close"]].rename(columns={"Close": "vix_level"})
    merged = spy.join(vix_close, how="inner")

    # VIX is quoted as a percentage (e.g. 18.5 = 18.5% annualized vol);
    # convert to the same fractional scale as realized_vol_* for a fair
    # comparison.
    merged["vix_level"] = merged["vix_level"] / 100
    merged["vix_spy_spread"] = merged["vix_level"] - merged["realized_vol_20d"]
    return merged


def main():
    spy, vix = load_raw_data()

    spy = add_return_features(spy)
    spy = add_realized_vol_features(spy)
    spy = add_atr_features(spy)
    spy = add_volume_features(spy)
    features = add_vix_features(spy, vix)

    feature_cols = [
        "log_return", "abs_return",
        "realized_vol_5d", "realized_vol_10d", "realized_vol_20d",
        "atr", "atr_pct",
        "volume_ratio",
        "vix_level", "vix_spy_spread",
    ]
    # Keep Close too -- 03_label_data.py needs it to compute forward
    # realized vol for the label.
    output = features[["Close"] + feature_cols].copy()

    n_before = len(output)
    output = output.dropna()
    n_after = len(output)
    print(
        f"Dropped {n_before - n_after} rows with NaN "
        f"(expected -- rolling windows need warmup history)."
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "features.csv"
    output.to_csv(out_path)

    print(f"Saved {len(output):,} rows x {len(feature_cols)} features -> "
          f"{out_path.relative_to(PROJECT_ROOT)}")
    print(f"Date range: {output.index.min().date()} to {output.index.max().date()}")
    print("\nFeature summary:")
    print(output[feature_cols].describe().round(4))
    print("\nDone. Next: run 03_label_data.py")


if __name__ == "__main__":
    main()
