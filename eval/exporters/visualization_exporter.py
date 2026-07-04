from __future__ import annotations

from collections import defaultdict
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def export_visualization_artifacts(
    output_dir: Path,
    metric_summary_rows: list[dict[str, Any]],
    metric_case_rows: list[dict[str, Any]],
    *,
    max_heatmap_cases: int = 50,
) -> list[Path]:
    """Write compact overview plots for one evaluation result directory."""

    visualizations_dir = output_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    overview_path = _plot_metric_overview(visualizations_dir, metric_summary_rows)
    if overview_path is not None:
        written_paths.append(overview_path)

    distribution_path = _plot_score_distribution(visualizations_dir, metric_case_rows)
    if distribution_path is not None:
        written_paths.append(distribution_path)

    heatmap_path = _plot_case_metric_heatmap(
        visualizations_dir,
        metric_case_rows,
        max_cases=max_heatmap_cases,
    )
    if heatmap_path is not None:
        written_paths.append(heatmap_path)

    return written_paths


def _plot_metric_overview(
    visualizations_dir: Path,
    metric_summary_rows: list[dict[str, Any]],
) -> Path | None:
    rows = [
        row
        for row in metric_summary_rows
        if _as_float(row.get("mean_score")) is not None
        or _as_float(row.get("pass_rate")) is not None
    ]
    if not rows:
        return None

    rows = sorted(rows, key=lambda item: str(item.get("metric", "")))
    labels = [_short_label(str(row.get("metric", "metric"))) for row in rows]
    mean_scores = [_as_float(row.get("mean_score")) for row in rows]
    pass_rates = [_as_float(row.get("pass_rate")) for row in rows]
    y_positions = np.arange(len(rows))
    height = 0.34

    fig_height = max(3.2, min(12.0, 0.48 * len(rows) + 1.8))
    fig, ax = plt.subplots(figsize=(8.6, fig_height))
    ax.barh(
        y_positions - height / 2,
        [_plot_value(value) for value in mean_scores],
        height=height,
        label="Mean score",
        color="#2e86de",
    )
    ax.barh(
        y_positions + height / 2,
        [_plot_value(value) for value in pass_rates],
        height=height,
        label="Pass rate",
        color="#27ae60",
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Metric Overview")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")

    for y_position, mean_score, pass_rate in zip(y_positions, mean_scores, pass_rates):
        _annotate_bar(ax, mean_score, y_position - height / 2)
        _annotate_bar(ax, pass_rate, y_position + height / 2)

    fig.tight_layout()
    path = visualizations_dir / "01_metric_overview.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_score_distribution(
    visualizations_dir: Path,
    metric_case_rows: list[dict[str, Any]],
) -> Path | None:
    metric_scores: dict[str, list[float]] = defaultdict(list)
    for row in metric_case_rows:
        score = _as_float(row.get("score"))
        if score is None:
            continue
        metric_scores[str(row.get("metric", "metric"))].append(score)

    if not metric_scores:
        return None

    metrics = sorted(metric_scores)
    scores = [metric_scores[metric] for metric in metrics]
    labels = [_short_label(metric) for metric in metrics]
    fig_height = max(3.2, min(12.0, 0.5 * len(metrics) + 1.6))
    fig, ax = plt.subplots(figsize=(8.6, fig_height))
    ax.boxplot(scores, orientation="horizontal", tick_labels=labels, patch_artist=True)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Case score")
    ax.set_title("Case Score Distribution")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    path = visualizations_dir / "02_score_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_case_metric_heatmap(
    visualizations_dir: Path,
    metric_case_rows: list[dict[str, Any]],
    *,
    max_cases: int,
) -> Path | None:
    case_metric_values: dict[str, dict[str, float]] = defaultdict(dict)
    for row in metric_case_rows:
        case_id = str(row.get("case_id", "case"))
        metric = str(row.get("metric", "metric"))
        value = _as_float(row.get("score"))
        if value is None:
            value = _success_value(row.get("success"))
        if value is not None:
            case_metric_values[case_id][metric] = value

    if not case_metric_values:
        return None

    metrics = sorted({metric for values in case_metric_values.values() for metric in values})
    if not metrics:
        return None

    ordered_cases = sorted(
        case_metric_values,
        key=lambda case_id: (
            _row_mean(case_metric_values[case_id].values()),
            case_id,
        ),
    )
    if len(ordered_cases) > max_cases:
        ordered_cases = ordered_cases[:max_cases]

    matrix = np.full((len(ordered_cases), len(metrics)), np.nan, dtype=float)
    for row_idx, case_id in enumerate(ordered_cases):
        for col_idx, metric in enumerate(metrics):
            value = case_metric_values[case_id].get(metric)
            if value is not None:
                matrix[row_idx, col_idx] = value

    if np.isnan(matrix).all():
        return None

    masked_matrix = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#eeeeee")

    fig_width = max(5.2, min(16.0, 1.15 * len(metrics) + 2.8))
    fig_height = max(4.2, min(18.0, 0.32 * len(ordered_cases) + 2.2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(masked_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([_short_label(metric, max_length=22) for metric in metrics], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ordered_cases)))
    ax.set_yticklabels([_short_label(case_id, max_length=36) for case_id in ordered_cases])
    title = "Case x Metric Heatmap"
    if len(case_metric_values) > max_cases:
        title += f" (lowest {max_cases} case averages)"
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Score / success")

    fig.tight_layout()
    path = visualizations_dir / "03_case_metric_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    rendered = float(value)
    if not isfinite(rendered):
        return None
    return rendered


def _success_value(value: Any) -> float | None:
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    return None


def _plot_value(value: float | None) -> float:
    if value is None:
        return 0.0
    return min(max(value, 0.0), 1.0)


def _annotate_bar(ax: Any, value: float | None, y_position: float) -> None:
    if value is None:
        return
    x_position = min(_plot_value(value) + 0.015, 0.96)
    ax.text(x_position, y_position, f"{value:.2f}", va="center", fontsize=8)


def _row_mean(values: Any) -> float:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return 1.0
    return sum(numeric_values) / len(numeric_values)


def _short_label(value: str, *, max_length: int = 42) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 1] + "..."
