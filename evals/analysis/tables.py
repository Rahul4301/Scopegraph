"""Markdown and CSV-friendly result tables."""

from collections.abc import Mapping
from pathlib import Path


def markdown_table(summary: Mapping[str, Mapping[str, float]]) -> str:
    metrics = sorted({metric for values in summary.values() for metric in values})
    lines = ["| system | " + " | ".join(metrics) + " |",
             "|---|" + "---|" * len(metrics)]
    for system, values in summary.items():
        lines.append("| " + system + " | " + " | ".join(
            f"{values.get(metric, 0.0):.4f}" for metric in metrics
        ) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_table(summary: Mapping[str, Mapping[str, float]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown_table(summary))
