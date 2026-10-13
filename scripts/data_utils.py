"""
data_utils.py

Shared raw-data-fetch logic, pulled out of 01_fetch_data.py so other
scripts (specifically scripts/volatility/refresh_predictions.py, the
daily automated job) can reuse the exact same fetch function instead
of re-implementing it -- a second, drifted copy of "how we pull a
ticker from Yahoo Finance" would be exactly the kind of silent
inconsistency this project has otherwise been careful to avoid (see
common.py's docstring on this same principle for the feature-building
logic).

Not numbered like the pipeline scripts (01_, 02_, ...) specifically so
it CAN be imported normally -- a filename starting with a digit isn't
a valid Python module name.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

BASKET = ["SPY", "QQQ", "IWM", "DIA", "XLF", "GLD", "TLT"]
VIX_SYMBOL = "^VIX"
DEFAULT_START = "2005-01-01"  # far enough back to span multiple vol regimes


def fetch_ticker(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    """Download daily OHLCV data for a single ticker via yfinance."""
    print(f"Fetching {symbol} from {start} to {end or 'today'}...")
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,   # adjusts OHLC for splits/dividends
        progress=False,
    )

    if df.empty:
        raise ValueError(
            f"No data returned for {symbol}. Check the ticker symbol, "
            f"date range, or your network connection."
        )

    # yfinance sometimes returns MultiIndex columns even for a single
    # ticker -- flatten them so downstream scripts get plain column names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    return df
