from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import seaborn as sns


def plot_candlestick_with_indicators(
    df: pd.DataFrame,
    ticker: str,
    output_dir: Path,
    bb_window: int = 20,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    apds = []
    for col in ["MA_20", "MA_50", f"BB_UPPER_{bb_window}", f"BB_LOWER_{bb_window}"]:
        if col in df.columns:
            apds.append(mpf.make_addplot(df[col]))

    path = output_dir / f"{ticker}_candlestick.png"
    mpf.plot(
        df,
        type="candle",
        style="yahoo",
        title=f"{ticker} Candlestick with MA/Bollinger Bands",
        volume=True,
        addplot=apds,
        savefig=str(path),
        tight_layout=True,
        figsize=(12, 8),
    )
    return path


def plot_rolling_volatility(df: pd.DataFrame, ticker: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{ticker}_rolling_volatility.png"

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.index, df["RollingVol_20"], color="tab:blue", label="20D Rolling Annualized Vol")
    ax.set_title(f"{ticker} Rolling Volatility")
    ax.set_ylabel("Volatility")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_correlation_heatmap(close_prices: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    corr = close_prices.pct_change().dropna().corr()
    path = output_dir / "asset_correlation_heatmap.png"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0.0, vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Asset Return Correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
