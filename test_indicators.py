import numpy as np
import pandas as pd

from stock_price_analysis.indicators import add_indicators, summarize_asset, trend_label


def _sample_df() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=80)
    close = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    return pd.DataFrame(
        {
            "Open": close * 0.995,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )


def test_add_indicators_columns() -> None:
    df = add_indicators(_sample_df())
    assert {"MA_20", "MA_50", "BB_UPPER_20", "BB_LOWER_20", "RollingVol_20"}.issubset(df.columns)


def test_trend_label() -> None:
    df = add_indicators(_sample_df())
    assert trend_label(df) in {"uptrend", "downtrend", "sideways", "unknown"}


def test_summary() -> None:
    df = add_indicators(_sample_df())
    summary = summarize_asset(df, ticker="AAPL")
    assert summary["ticker"] == "AAPL"
    assert "annualized_volatility" in summary
