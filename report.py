from __future__ import annotations

from pathlib import Path

import pandas as pd


def _pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.2%}"


def write_markdown_report(
    summary_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    output_dir: Path,
    chart_paths: list[Path],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "analysis_report.md"

    lines = [
        "# Stock Price Analysis Report",
        "",
        "## Trend and Volatility Summary",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            f"- {row['ticker']}: trend={row['trend']}, total_return={_pct(row['total_return'])}, annualized_vol={_pct(row['annualized_volatility'])}, latest_close={row['latest_close']:.2f}"
        )

    lines.extend(
        [
            "",
            "## Correlation Matrix",
            correlation_df.to_string(),
            "",
            "## Generated Charts",
        ]
    )

    for p in chart_paths:
        lines.append(f"- `{p.name}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
