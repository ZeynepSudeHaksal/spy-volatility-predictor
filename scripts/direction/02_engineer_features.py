"""
02_engineer_features.py

Builds the feature set for the direction classifier from raw per-ticker
CSVs and the shared VIX CSV. Long-format panel (one row per ticker per
date), exactly as in the volatility pipeline (scripts/volatility/) -- and for the
same reason, every feature here is computed WITHIN each ticker's own
series before stacking, never blended across tickers.

FEATURE GROUPS -- DIFFERENT FROM THE VOLATILITY PROJECT ON PURPOSE
------------------------------------------------------------------------
Volatility clusters (recent vol predicts near-future vol quite directly).
Direction doesn't work that way -- there's no single dominant signal, so
this project leans on the standard technical-analysis toolkit instead,
each capturing a different hypothesis about what might carry predictive
signal:

1. Momentum (past N-day cumulative return): tests whether recent moves
   continue (momentum) or reverse (mean reversion) -- both are
   documented, regime-dependent effects in the literature.
2. RSI (14-day): classic overbought/oversold oscillator.
3. MACD (12/26/9): trend-following indicator; the histogram (MACD minus
   its signal line) captures trend ACCELERATION, not just direction.
4. Price relative to its own moving averages (20d, 50d): a simple,
   robust trend-position signal.
5. Volume ratio: volume spikes often precede or confirm price moves.
6. Realized volatility + VIX level/spread: context features -- direction
   may be more or less predictable depending on the vol regime (e.g.
   trending markets vs. choppy/mean-reverting ones).

Usage:
    python scripts/02_engineer_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "direction"

BASKET = ["SPY", "QQQ", "IWM", "DIA", "XLF"]
MOMENTUM_WINDOWS = [5, 10, 20]
RSI_WINDOW = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MA_WINDOWS = [20, 50]
VOLUME_MA_WINDOW = 20
REALIZED_VOL_WINDOW = 20
TRADING_DAYS_PER_YEAR = 252


def load_ticker_raw(ticker: str) -> pd.DataFrame:
    path = RAW_DATA_DIR / f"{ticker.lower()}_raw.csv"
    if not path.exists():
        print(f"ERROR: {path.name} not found. Run 01_fetch_data.py first.", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def load_vix() -> pd.DataFrame:
    path = RAW_DATA_DIR / "vix_raw.csv"
    if not path.exists():
        print("ERROR: vix_raw.csv not found. Run 01_fetch_data.py first.", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def compute_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi / 100  # scale to 0-1 for consistency with other fractional features


def compute_macd(close: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    # normalize by price so it's comparable across tickers/price levels
    return (macd_line / close), (histogram / close)


def engineer_single_ticker(df: pd.DataFrame, vix: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    for window in MOMENTUM_WINDOWS:
        df[f"momentum_{window}d"] = df["Close"].pct_change(window)

    df["rsi_14"] = compute_rsi(df["Close"], RSI_WINDOW)
    df["macd"], df["macd_hist"] = compute_macd(
        df["Close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL
    )

    for window in MA_WINDOWS:
        ma = df["Close"].rolling(window).mean()
        df[f"price_vs_ma{window}"] = (df["Close"] / ma) - 1

    volume_ma = df["Volume"].rolling(VOLUME_MA_WINDOW).mean()
    df["volume_ratio"] = df["Volume"] / volume_ma

    df["realized_vol_20d"] = (
        df["log_return"].rolling(REALIZED_VOL_WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    merged = df.join(vix[["Close"]].rename(columns={"Close": "vix_level"}), how="inner")
    merged["vix_level"] = merged["vix_level"] / 100
    merged["vix_spy_spread"] = merged["vix_level"] - merged["realized_vol_20d"]

    merged["ticker"] = ticker
    return merged


def main():
    vix = load_vix()

    feature_cols = (
        [f"momentum_{w}d" for w in MOMENTUM_WINDOWS]
        + ["rsi_14", "macd", "macd_hist"]
        + [f"price_vs_ma{w}" for w in MA_WINDOWS]
        + ["volume_ratio", "realized_vol_20d", "vix_level", "vix_spy_spread", "log_return"]
    )

    all_rows = []
    for ticker in BASKET:
        raw = load_ticker_raw(ticker)
        engineered = engineer_single_ticker(raw, vix, ticker)
        subset = engineered[["ticker", "Close"] + feature_cols].copy()

        n_before = len(subset)
        subset = subset.dropna()
        n_after = len(subset)
        print(f"{ticker}: dropped {n_before - n_after} warmup rows, kept {n_after:,} rows "
              f"({subset.index.min().date()} to {subset.index.max().date()})")
        all_rows.append(subset)

    panel = pd.concat(all_rows)
    panel = panel.reset_index().sort_values(["Date", "ticker"]).set_index("Date")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "features.csv"
    panel.to_csv(out_path)

    print(f"\nSaved {len(panel):,} total rows -> {out_path.relative_to(PROJECT_ROOT)}")
    print("\nDone. Next: run 03_label_data.py")


if __name__ == "__main__":
    main()
