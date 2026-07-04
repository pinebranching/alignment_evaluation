from __future__ import annotations

import csv
from numbers import Real
from pathlib import Path
from typing import Any


def export_csv_artifacts(
    output_dir: Path,
    metric_summary_rows: list[dict[str, Any]],
    metric_case_rows: list[dict[str, Any]],
    analysis_case_rows: list[dict[str, Any]],
) -> None:
    _write_csv(output_dir / "metric_summary.csv", metric_summary_rows)
    _write_csv(output_dir / "metric_case_scores.csv", metric_case_rows)
    _write_csv(output_dir / "per_case_analysis.csv", analysis_case_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    formatted_rows = [_format_csv_row(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(formatted_rows)


def _format_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    formatted_row: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            formatted_row[key] = None
            continue

        if (key.endswith("score") or key.endswith("threshold")) and isinstance(value, Real) and not isinstance(value, bool):
            formatted = f"{float(value):.3f}"
            formatted_row[key] = formatted.replace(".", ",")
            continue

        formatted_row[key] = value
    return formatted_row