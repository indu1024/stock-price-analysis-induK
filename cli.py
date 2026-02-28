from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data_sources import fetch_prices
from .indicators import add_indicators, summarize_asset
from .report import write_markdown_report
from .visualization import (
    plot_candlestick_with_indicators,
    plot_correlation_heatmap,
    plot_rolling_volatility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze stocks with candlestick charts, MA/Bollinger indicators, volatility, and correlations"
    )
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g., AAPL,MSFT,TSLA")
    parser.add_argument("--start", required=True, help="Start date, e.g., 2022-01-01")
    parser.add_argument("--end", help="End date, e.g., 2025-12-31")
    parser.add_argument("--source", choices=["yfinance", "alphavantage"], default="yfinance")
    parser.add_argument("--alphavantage-api-key", help="API key for Alpha Vantage")
    parser.add_argument("--interval", default="1d", help="Data interval for yfinance, e.g., 1d,1wk")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if len(tickers) < 2:
        raise ValueError("Provide at least 2 tickers to analyze correlations")

    raw = fetch_prices(
        tickers=tickers,
        source=args.source,
        start=args.start,
        end=args.end,
        interval=args.interval,
        alphavantage_api_key=args.alphavantage_api_key,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | str]] = []
    chart_paths: list[Path] = []
    closes: dict[str, pd.Series] = {}

    for ticker, df in raw.items():
        enriched = add_indicators(df)
        closes[ticker] = enriched["Close"].rename(ticker)
        summary_rows.append(summarize_asset(enriched, ticker))

        enriched_csv_path = output_dir / f"{ticker}_enriched_prices.csv"
        enriched.to_csv(enriched_csv_path)

        chart_paths.append(plot_candlestick_with_indicators(enriched, ticker=ticker, output_dir=output_dir))
        chart_paths.append(plot_rolling_volatility(enriched, ticker=ticker, output_dir=output_dir))

    close_df = pd.concat(closes.values(), axis=1).dropna(how="any")
    corr = close_df.pct_change().dropna().corr()
    corr_path = output_dir / "asset_correlation.csv"
    corr.to_csv(corr_path)

    chart_paths.append(plot_correlation_heatmap(close_df, output_dir=output_dir))

    summary_df = pd.DataFrame(summary_rows).sort_values("ticker")
    summary_path = output_dir / "asset_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    report_path = write_markdown_report(
        summary_df=summary_df,
        correlation_df=corr,
        output_dir=output_dir,
        chart_paths=chart_paths,
    )

    print("Saved outputs:")
    print(f"- Summary CSV: {summary_path}")
    print(f"- Correlation CSV: {corr_path}")
    print(f"- Report: {report_path}")
    for p in chart_paths:
        print(f"- Chart: {p}")


if __name__ == "__main__":
    main()
