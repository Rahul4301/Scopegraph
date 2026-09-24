"""Create processed JSON, Markdown, and SVG reports from raw JSONL."""

import argparse
from pathlib import Path

from evals.analysis.aggregate import aggregate_files
from evals.analysis.plots import metric_bar_svg
from evals.analysis.scope_classification import write_scope_classification_report
from evals.analysis.tables import write_markdown_table
from evals.schemas import ScopeClassificationEvaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--metric", default="recall_at_8")
    args = parser.parse_args()
    summary = aggregate_files(args.inputs, args.output_root / "processed" / "summary.json")
    write_markdown_table(summary, args.output_root / "tables" / "summary.md")
    metric_bar_svg(summary, args.metric, args.output_root / "figures" / f"{args.metric}.svg")
    classification_candidates = []
    if len({path.parent for path in args.inputs}) == 1:
        classification_candidates.append(args.inputs[0].parent / "classification.json")
    if len(args.inputs) == 1:
        classification_candidates.append(args.inputs[0].with_suffix(".classification.json"))
    classification_path = next(
        (path for path in classification_candidates if path.exists()), None
    )
    if classification_path is not None:
        write_scope_classification_report(
            ScopeClassificationEvaluation.model_validate_json(classification_path.read_text()),
            args.output_root,
        )
    print(args.output_root / "processed" / "summary.json")


if __name__ == "__main__":
    main()
