from __future__ import annotations

import csv
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def export_orchestrator_visualizations(
    run_dir: Path,
    state_payload: dict[str, Any],
    master_config: dict[str, Any],
    *,
    metric_summary_csv_path: Path | None = None,
    max_branches: int = 80,
) -> list[Path]:
    """Write compact cross-branch overview plots for an orchestrator run."""

    branch_rows = (
        _collect_branch_metric_rows_from_csv(metric_summary_csv_path)
        if metric_summary_csv_path is not None and metric_summary_csv_path.exists()
        else _collect_branch_metric_rows(state_payload)
    )
    if not branch_rows:
        return []

    output_dir = run_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    heatmap_path = _plot_branch_metric_heatmap(output_dir, branch_rows, max_branches=max_branches)
    if heatmap_path is not None:
        written_paths.append(heatmap_path)

    delta_rows = _build_base_delta_rows(branch_rows, master_config)
    delta_path = _plot_base_delta_chart(output_dir, delta_rows)
    if delta_path is not None:
        written_paths.append(delta_path)

    return written_paths


def _collect_branch_metric_rows_from_csv(path: Path) -> list[dict[str, Any]]:
    grouped_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            branch_id = str(row.get("branch_id") or "").strip()
            metric_name = str(row.get("metric") or "").strip()
            if not branch_id or not metric_name:
                continue

            value = _as_float(row.get("mean_score"))
            if value is None:
                value = _as_float(row.get("pass_rate"))
            if value is None:
                continue

            key = (
                branch_id,
                str(row.get("candidate_id") or ""),
                str(row.get("evaluation_benchmark_id") or ""),
            )
            grouped_row = grouped_rows.setdefault(
                key,
                {
                    "branch_id": branch_id,
                    "candidate_id": row.get("candidate_id") or _candidate_from_branch_id(branch_id),
                    "candidate_kind": row.get("candidate_kind"),
                    "method": row.get("alignment_technique") or "base",
                    "evaluation_benchmark_id": row.get("evaluation_benchmark_id"),
                    "metric_values": {},
                },
            )
            grouped_row["metric_values"][metric_name] = value

    return list(grouped_rows.values())


def _collect_branch_metric_rows(state_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch in state_payload.get("branches", []):
        if branch.get("status") != "completed":
            continue
        evaluation = branch.get("evaluation") or {}
        if evaluation.get("status") != "completed":
            continue
        summary = evaluation.get("result_summary") or {}
        metrics = summary.get("metrics") or []
        if not metrics:
            continue

        metric_values: dict[str, float] = {}
        for metric in metrics:
            metric_name = str(metric.get("metric", "")).strip()
            if not metric_name:
                continue
            value = _as_float(metric.get("mean_score"))
            if value is None:
                value = _as_float(metric.get("pass_rate"))
            if value is not None:
                metric_values[metric_name] = value

        if not metric_values:
            continue

        rows.append(
            {
                "branch_id": str(branch.get("branch_id", "")),
                "candidate_id": branch.get("candidate_id") or _candidate_from_branch_id(branch.get("branch_id")),
                "candidate_kind": branch.get("candidate_kind"),
                "method": branch.get("method") or "base",
                "evaluation_benchmark_id": branch.get("evaluation_benchmark_id"),
                "metric_values": metric_values,
            }
        )
    return rows


def _plot_branch_metric_heatmap(
    output_dir: Path,
    branch_rows: list[dict[str, Any]],
    *,
    max_branches: int,
) -> Path | None:
    metrics = sorted({metric for row in branch_rows for metric in row["metric_values"]})
    if not metrics:
        return None

    ordered_rows = sorted(
        branch_rows,
        key=lambda row: (_row_mean(row["metric_values"].values()), str(row["branch_id"])),
    )
    if len(ordered_rows) > max_branches:
        ordered_rows = ordered_rows[:max_branches]

    matrix = np.full((len(ordered_rows), len(metrics)), np.nan, dtype=float)
    for row_idx, row in enumerate(ordered_rows):
        for col_idx, metric in enumerate(metrics):
            value = row["metric_values"].get(metric)
            if value is not None:
                matrix[row_idx, col_idx] = value

    if np.isnan(matrix).all():
        return None

    masked_matrix = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#eeeeee")

    fig_width = max(5.4, min(18.0, 1.2 * len(metrics) + 3.0))
    fig_height = max(4.2, min(22.0, 0.36 * len(ordered_rows) + 2.4))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(masked_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([_short_label(metric, max_length=24) for metric in metrics], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ordered_rows)))
    ax.set_yticklabels([_branch_label(row) for row in ordered_rows])
    title = "Branch x Metric Overview"
    if len(branch_rows) > max_branches:
        title += f" (lowest {max_branches} branch averages)"
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Mean score / pass rate")

    fig.tight_layout()
    path = output_dir / "04_branch_metric_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _build_base_delta_rows(
    branch_rows: list[dict[str, Any]],
    master_config: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate_index = {
        str(candidate.get("candidate_id")): candidate
        for candidate in (master_config.get("alignment") or {}).get("candidates", [])
        if candidate.get("candidate_id") is not None
    }
    base_model_ref_by_candidate: dict[str, str] = {}
    base_candidate_by_model_ref: dict[str, str] = {}

    for candidate_id, candidate in candidate_index.items():
        kind = str(candidate.get("kind", "")).strip().lower()
        if kind == "base":
            model_ref = str(candidate.get("model_ref", "")).strip()
            if model_ref:
                base_model_ref_by_candidate[candidate_id] = model_ref
                base_candidate_by_model_ref.setdefault(model_ref, candidate_id)
        elif kind == "aligned":
            base_model_ref = str(candidate.get("base_model_ref", "")).strip()
            if base_model_ref:
                base_model_ref_by_candidate[candidate_id] = base_model_ref

    base_rows: dict[tuple[str, str | None], dict[str, float]] = {}
    for row in branch_rows:
        candidate_id = str(row.get("candidate_id") or "")
        candidate = candidate_index.get(candidate_id, {})
        if str(candidate.get("kind", "")).strip().lower() != "base":
            continue
        model_ref = base_model_ref_by_candidate.get(candidate_id)
        if model_ref:
            base_rows[(model_ref, row.get("evaluation_benchmark_id"))] = row["metric_values"]

    delta_rows: list[dict[str, Any]] = []
    for row in branch_rows:
        candidate_id = str(row.get("candidate_id") or "")
        candidate = candidate_index.get(candidate_id, {})
        if str(candidate.get("kind", "")).strip().lower() != "aligned":
            continue

        base_model_ref = base_model_ref_by_candidate.get(candidate_id)
        base_values = base_rows.get((base_model_ref, row.get("evaluation_benchmark_id")))
        if not base_values:
            continue

        deltas = {}
        for metric, value in row["metric_values"].items():
            base_value = base_values.get(metric)
            if base_value is not None:
                deltas[metric] = value - base_value

        if deltas:
            delta_rows.append(
                {
                    "branch_id": row["branch_id"],
                    "candidate_id": candidate_id,
                    "base_candidate_id": base_candidate_by_model_ref.get(base_model_ref),
                    "method": row.get("method") or str(candidate.get("method", "aligned")),
                    "metric_deltas": deltas,
                }
            )

    return delta_rows


def _plot_base_delta_chart(
    output_dir: Path,
    delta_rows: list[dict[str, Any]],
) -> Path | None:
    if not delta_rows:
        return None

    metrics = sorted({metric for row in delta_rows for metric in row["metric_deltas"]})
    if not metrics:
        return None

    ordered_rows = sorted(
        delta_rows,
        key=lambda row: (_row_mean(row["metric_deltas"].values()), str(row["branch_id"])),
    )
    matrix = np.full((len(ordered_rows), len(metrics)), np.nan, dtype=float)
    for row_idx, row in enumerate(ordered_rows):
        for col_idx, metric in enumerate(metrics):
            value = row["metric_deltas"].get(metric)
            if value is not None:
                matrix[row_idx, col_idx] = value

    if np.isnan(matrix).all():
        return None

    max_abs = float(np.nanmax(np.abs(matrix))) if not np.isnan(matrix).all() else 0.0
    limit = max(0.1, min(1.0, max_abs))
    masked_matrix = np.ma.masked_invalid(matrix)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("#eeeeee")

    fig_width = max(5.4, min(18.0, 1.2 * len(metrics) + 3.0))
    fig_height = max(4.2, min(18.0, 0.42 * len(ordered_rows) + 2.4))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(masked_matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([_short_label(metric, max_length=24) for metric in metrics], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ordered_rows)))
    ax.set_yticklabels([_branch_label(row) for row in ordered_rows])
    ax.set_title("Aligned Candidate Delta vs Base")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Metric delta")

    fig.tight_layout()
    path = output_dir / "05_base_delta_chart.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _as_float(value: Any) -> float | None:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        value = value.replace(",", ".")
        try:
            rendered = float(value)
        except ValueError:
            return None
        return rendered if isfinite(rendered) else None
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    rendered = float(value)
    if not isfinite(rendered):
        return None
    return rendered


def _row_mean(values: Any) -> float:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return 1.0
    return sum(numeric_values) / len(numeric_values)


def _branch_label(row: dict[str, Any]) -> str:
    candidate_id = str(row.get("candidate_id") or row.get("branch_id") or "branch")
    method = str(row.get("method") or "base")
    label = candidate_id if method == "base" else f"{candidate_id} ({method})"
    return _short_label(label, max_length=44)


def _candidate_from_branch_id(branch_id: Any) -> str | None:
    if branch_id is None:
        return None
    return str(branch_id).split("--", 1)[0] or None


def _short_label(value: str, *, max_length: int = 42) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 1] + "..."
