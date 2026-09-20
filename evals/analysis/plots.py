"""Dependency-free SVG plots for evaluation summaries."""

from collections.abc import Mapping
from html import escape
from pathlib import Path


def metric_bar_svg(summary: Mapping[str, Mapping[str, float]], metric: str,
                   path: str | Path) -> None:
    width, height = 720, 420
    systems = list(summary)
    max_value = max((float(summary[system].get(metric, 0.0)) for system in systems), default=1.0)
    max_value = max(max_value, 1e-9)
    bars = []
    for index, system in enumerate(systems):
        value = float(summary[system].get(metric, 0.0))
        x = 40 + index * max(1, (width - 80) // max(1, len(systems)))
        bar_width = max(24, (width - 100) // max(1, len(systems)) - 14)
        bar_height = value / max_value * 300
        bars.append(
            f'<rect x="{x}" y="{350 - bar_height:.1f}" width="{bar_width}" '
            f'height="{bar_height:.1f}" fill="#5d9b8a"/><text x="{x}" y="370" '
            f'font-size="11">{escape(system)}</text><text x="{x}" y="{340 - bar_height:.1f}" '
            f'font-size="10">{value:.3f}</text>'
        )
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#0b1110"/>'
           f'<text x="40" y="28" fill="#e6eeec" font-size="16">{escape(metric)}</text>'
           + "".join(bars) + '</svg>\n')
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg)
