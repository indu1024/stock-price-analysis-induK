from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests


@dataclass
class PriceData:
    ticker: str
    data: pd.DataFrame


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    out = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    out = out[required].copy()
    out = out.dropna()
    out = out.sort_index()
    return out


def fetch_from_yfinance(ticker: str, start: str, end: str | None = None, interval: str = "1d") -> PriceData:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required. Install dependencies first.") from exc

    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned from yfinance for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    out = _standardize_ohlcv(df)
    return PriceData(ticker=ticker, data=out)


def fetch_from_alphavantage(
    ticker: str,
    api_key: str,
    outputsize: str = "full",
) -> PriceData:
    if outputsize not in {"compact", "full"}:
        raise ValueError("outputsize must be compact or full")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": outputsize,
        "apikey": api_key,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    ts_key = "Time Series (Daily)"
    if ts_key not in payload:
        raise ValueError(f"Alpha Vantage response missing '{ts_key}'. Response: {payload}")

    raw = pd.DataFrame(payload[ts_key]).T
    raw.index = pd.to_datetime(raw.index)

    out = pd.DataFrame(
        {
            "Open": raw["1. open"].astype(float),
            "High": raw["2. high"].astype(float),
            "Low": raw["3. low"].astype(float),
            "Close": raw["4. close"].astype(float),
            "Volume": raw["6. volume"].astype(float),
        }
    ).sort_index()

    return PriceData(ticker=ticker, data=out)


def fetch_prices(
    tickers: Iterable[str],
    source: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
    alphavantage_api_key: str | None = None,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for t in tickers:
        if source == "yfinance":
            result[t] = fetch_from_yfinance(ticker=t, start=start, end=end, interval=interval).data
        elif source == "alphavantage":
            if not alphavantage_api_key:
                raise ValueError("alphavantage_api_key is required when source=alphavantage")
            df = fetch_from_alphavantage(ticker=t, api_key=alphavantage_api_key).data
            if start:
                df = df.loc[df.index >= pd.Timestamp(start)]
            if end:
                df = df.loc[df.index <= pd.Timestamp(end)]
            result[t] = df
        else:
            raise ValueError("source must be yfinance or alphavantage")

        if result[t].empty:
            raise ValueError(f"No price rows after filtering for ticker {t}")
    return result
