"""Download daily crypto prices and build simple features.

WHY a loader module? Notebooks, scripts and the dashboard all need the same data.
Writing it once avoids copy-paste bugs and shows recruiters you structure code properly.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_prices(ticker: str = "BTC-USD", years: int = 5, refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame indexed by date with one column: Close.

    - First tries the local CSV cache (fast and reproducible).
    - If there is no cache (or refresh=True) it downloads from Yahoo Finance and saves a CSV.
    - If the download fails but a cache exists, it falls back to the cache.
    """
    path = DATA_DIR / f"{ticker}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    try:
        import yfinance as yf  # imported here so the module loads even without internet
        df = yf.download(ticker, period=f"{years}y", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):      # newer yfinance returns 2-level columns
            df.columns = df.columns.get_level_values(0)
        df = df[["Close"]].dropna()
        if df.empty:
            raise ValueError("Empty download")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        try:
            DATA_DIR.mkdir(exist_ok=True)
            df.to_csv(path)
        except OSError:
            pass                                       # read-only filesystems are fine
        return df
    except Exception:
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True)
        raise


def add_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Add log returns, a rolling mean and annualised rolling volatility.

    Log return = ln(P_t / P_{t-1}). Returns are (roughly) stationary, prices are not,
    which is why almost all finance analysis works on returns.
    Volatility = std of returns; x sqrt(365) annualises it (crypto trades every day).
    """
    out = df.copy()
    out["Return"] = np.log(out["Close"]).diff()
    out["MA"] = out["Close"].rolling(window).mean()
    out["Volatility"] = out["Return"].rolling(window).std() * np.sqrt(365)
    return out
