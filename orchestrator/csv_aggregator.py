from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


CSV_EXPORTS = {
    "metric_summary.csv": "all_metric_summary.csv",
    "per_case_analysis.csv": "all_per_case_analysis.csv",
    "metric_case_scores.csv": "all_metric_case_scores.csv",
}

METADATA_FIELDS = [
    "row_uid",
    "run_id",
    "run_name",
    "branch_id",
    "branch_name",
    "candidate_id",
    "candidate_kind",
    "base_model",
    "base_model_ref",
    "alignment_technique",
    "train_dataset_ref",
    "train_dataset_path",
    "test_dataset_ref",
    "test_dataset_path",
    "inference_benchmark_id",
    "evaluation_benchmark_id",
    "source_csv",
    "source_row_number",
]


def export_orchestrator_csv_aggregates(
    run_dir: Path,
    state_payload: dict[str, Any],
    master_config: dict[str, Any],
    *,
    config_dir: Path,
) -> list[Path]:
    """Merge branch-level evaluation CSVs into run-level analysis CSVs."""

    written_paths: list[Path] = []
    for source_name, output_name in CSV_EXPORTS.items():
        rows = _collect_rows(
            run_dir=run_dir,
            state_payload=state_payload,
            master_config=master_config,
            config_dir=config_dir,
            source_name=source_name,
        )
        if not rows:
            continue
        output_path = run_dir / output_name
        _write_csv(output_path, rows)
        written_paths.append(output_path)
    return written_paths


def _collect_rows(
    *,
    run_dir: Path,
    state_payload: dict[str, Any],
    master_config: dict[str, Any],
    config_dir: Path,
    source_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch in state_payload.get("branches", []):
        if branch.get("status") != "completed":
            continue
        evaluation = branch.get("evaluation") or {}
        if evaluation.get("status") != "completed":
            continue

        branch_id = str(branch.get("branch_id", "")).strip()
        if not branch_id:
            continue
        source_path = run_dir / "branches" / branch_id / "evaluation" / source_name
        if not source_path.exists():
            continue

        branch_metadata = _build_branch_metadata(
            state_payload=state_payload,
            master_config=master_config,
            config_dir=config_dir,
            branch=branch,
            source_name=source_name,
        )
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            for source_row_number, row in enumerate(reader, start=1):
                row_uid = (
                    f"{branch_metadata['run_id']}:{branch_id}:"
                    f"{source_name}:{source_row_number}"
                )
                rows.append(
                    {
                        **branch_metadata,
                        "row_uid": row_uid,
                        "source_row_number": source_row_number,
                        **_without_metadata_keys(row),
                    }
                )
    return rows


def _build_branch_metadata(
    *,
    state_payload: dict[str, Any],
    master_config: dict[str, Any],
    config_dir: Path,
    branch: dict[str, Any],
    source_name: str,
) -> dict[str, Any]:
    candidate_id = str(branch.get("candidate_id") or _candidate_from_branch_id(branch.get("branch_id")) or "")
    candidate = _candidate_index(master_config).get(candidate_id, {})
    candidate_kind = str(branch.get("candidate_kind") or candidate.get("kind") or "").strip().lower()
    alignment_technique = str(branch.get("method") or candidate.get("method") or "base")
    base_model_ref = _base_model_ref(candidate)
    base_model = _model_name(master_config, base_model_ref)
    train_dataset_ref = str(candidate.get("dataset_ref", "")).strip() or None
    test_dataset_ref = _test_dataset_ref(master_config, branch)

    return {
        "row_uid": None,
        "run_id": state_payload.get("run_id"),
        "run_name": state_payload.get("run_name"),
        "branch_id": branch.get("branch_id"),
        "branch_name": branch.get("branch_id"),
        "candidate_id": candidate_id or None,
        "candidate_kind": candidate_kind or None,
        "base_model": base_model,
        "base_model_ref": base_model_ref,
        "alignment_technique": alignment_technique,
        "train_dataset_ref": train_dataset_ref,
        "train_dataset_path": _dataset_path(master_config, config_dir, train_dataset_ref),
        "test_dataset_ref": test_dataset_ref,
        "test_dataset_path": _dataset_path(master_config, config_dir, test_dataset_ref),
        "inference_benchmark_id": branch.get("inference_benchmark_id"),
        "evaluation_benchmark_id": branch.get("evaluation_benchmark_id"),
        "source_csv": source_name,
        "source_row_number": None,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames = list(METADATA_FIELDS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _without_metadata_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in METADATA_FIELDS}


def _candidate_index(master_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(candidate.get("candidate_id")): candidate
        for candidate in (master_config.get("alignment") or {}).get("candidates", [])
        if candidate.get("candidate_id") is not None
    }


def _base_model_ref(candidate: dict[str, Any]) -> str | None:
    kind = str(candidate.get("kind", "")).strip().lower()
    if kind == "base":
        value = str(candidate.get("model_ref", "")).strip()
    else:
        value = str(candidate.get("base_model_ref", "")).strip()
    return value or None


def _model_name(master_config: dict[str, Any], model_ref: str | None) -> str | None:
    if not model_ref:
        return None
    model = ((master_config.get("catalog") or {}).get("models") or {}).get(model_ref) or {}
    value = str(model.get("model_name_or_path", "")).strip()
    return value or model_ref


def _test_dataset_ref(master_config: dict[str, Any], branch: dict[str, Any]) -> str | None:
    evaluation_benchmark_id = str(branch.get("evaluation_benchmark_id") or "").strip()
    benchmark = _benchmark_index(master_config, "evaluation").get(evaluation_benchmark_id)
    if benchmark is not None:
        value = str(benchmark.get("dataset_ref", "")).strip()
        if value:
            return value

    inference_benchmark_id = str(branch.get("inference_benchmark_id") or "").strip()
    benchmark = _benchmark_index(master_config, "inference").get(inference_benchmark_id)
    if benchmark is not None:
        value = str(benchmark.get("dataset_ref", "")).strip()
        if value:
            return value
    return None


def _benchmark_index(master_config: dict[str, Any], section: str) -> dict[str, dict[str, Any]]:
    benchmarks = ((master_config.get(section) or {}).get("benchmarks") or [])
    return {
        str(benchmark.get("benchmark_id")): benchmark
        for benchmark in benchmarks
        if benchmark.get("benchmark_id") is not None
    }


def _dataset_path(
    master_config: dict[str, Any],
    config_dir: Path,
    dataset_ref: str | None,
) -> str | None:
    if not dataset_ref:
        return None
    dataset = ((master_config.get("catalog") or {}).get("datasets") or {}).get(dataset_ref) or {}
    raw_path = dataset.get("path")
    if raw_path in (None, ""):
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def _candidate_from_branch_id(branch_id: Any) -> str | None:
    if branch_id is None:
        return None
    return str(branch_id).split("--", 1)[0] or None
