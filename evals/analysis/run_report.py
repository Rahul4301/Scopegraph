"""Create processed JSON, Markdown, and SVG reports from raw JSONL."""

import argparse
from pathlib import Path

from evals.analysis.aggregate import aggregate_files
from evals.analysis.plots import metric_bar_svg
from evals.analysis.tables import write_markdown_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--metric", default="recall_at_8")
    args = parser.parse_args()
    summary = aggregate_files(args.inputs, args.output_root / "processed" / "summary.json")
    write_markdown_table(summary, args.output_root / "tables" / "summary.md")
    metric_bar_svg(summary, args.metric, args.output_root / "figures" / f"{args.metric}.svg")
    print(args.output_root / "processed" / "summary.json")


if __name__ == "__main__":
    main()
