from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(
    df: pd.DataFrame,
    ma_windows: tuple[int, int] = (20, 50),
    bb_window: int = 20,
    bb_std: float = 2.0,
) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]

    for w in ma_windows:
        out[f"MA_{w}"] = close.rolling(window=w).mean()

    mid = close.rolling(window=bb_window).mean()
    std = close.rolling(window=bb_window).std(ddof=1)
    out[f"BB_MID_{bb_window}"] = mid
    out[f"BB_UPPER_{bb_window}"] = mid + bb_std * std
    out[f"BB_LOWER_{bb_window}"] = mid - bb_std * std

    out["Return"] = close.pct_change()
    out["LogReturn"] = np.log(close / close.shift(1))
    out["RollingVol_20"] = out["LogReturn"].rolling(20).std(ddof=1) * np.sqrt(252)
    return out


def trend_label(df: pd.DataFrame, short_ma: int = 20, long_ma: int = 50) -> str:
    s_col = f"MA_{short_ma}"
    l_col = f"MA_{long_ma}"
    if s_col not in df.columns or l_col not in df.columns:
        return "unknown"

    latest = df[["Close", s_col, l_col]].dropna()
    if latest.empty:
        return "unknown"

    row = latest.iloc[-1]
    if row["Close"] > row[s_col] > row[l_col]:
        return "uptrend"
    if row["Close"] < row[s_col] < row[l_col]:
        return "downtrend"
    return "sideways"


def summarize_asset(df: pd.DataFrame, ticker: str, short_ma: int = 20, long_ma: int = 50) -> dict[str, float | str]:
    returns = df["LogReturn"].dropna()
    annual_vol = float(returns.std(ddof=1) * np.sqrt(252)) if not returns.empty else float("nan")
    total_return = float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0)

    return {
        "ticker": ticker,
        "trend": trend_label(df, short_ma=short_ma, long_ma=long_ma),
        "total_return": total_return,
        "annualized_volatility": annual_vol,
        "latest_close": float(df["Close"].iloc[-1]),
    }
