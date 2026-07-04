from __future__ import annotations

import json
from typing import Any

from ..policies.metric_requirements import collect_required_fields
from ..schemas import DatasetConfig, EvalConfig, ModelOutputsConfig


def prepare_eval_records(config: EvalConfig) -> list[dict[str, Any]]:
    dataset_records = load_dataset(config.dataset)
    output_records = load_outputs(config.model_outputs)

    dataset_by_id: dict[str, dict[str, Any]] = {}
    for record in dataset_records:
        value = record.get(config.dataset.id_field)
        if value in (None, ""):
            raise ValueError(f"Missing id value for field '{config.dataset.id_field}'.")
        record_id = str(value)
        if record_id in dataset_by_id:
            raise ValueError(f"Duplicate dataset id detected: {record_id}")
        dataset_by_id[record_id] = record

    output_id_field = config.model_outputs.id_field or config.dataset.id_field
    output_by_id: dict[str, dict[str, Any]] = {}
    for record in output_records:
        value = record.get(output_id_field)
        if value in (None, ""):
            raise ValueError(f"Missing id value for field '{output_id_field}'.")
        record_id = str(value)
        if record_id in output_by_id:
            raise ValueError(f"Duplicate model output id detected: {record_id}")
        output_by_id[record_id] = record

    missing_output_ids = [record_id for record_id in dataset_by_id if record_id not in output_by_id]
    if missing_output_ids:
        preview = ", ".join(missing_output_ids[:5])
        raise ValueError(
            f"Model outputs are missing for {len(missing_output_ids)} dataset records. "
            f"First ids: {preview}"
        )

    joined_records: list[dict[str, Any]] = []
    required_fields = collect_required_fields(config.metrics)
    for record_id, dataset_record in dataset_by_id.items():
        output_record = output_by_id[record_id]
        joined = {
            "id": record_id,
            "input": _required_text(dataset_record, config.dataset.input_field, "dataset input"),
            "actual_output": _required_text(
                output_record,
                config.model_outputs.actual_output_field,
                "model output",
            ),
            "expected_output": _optional_text(
                dataset_record,
                config.dataset.expected_output_field,
            ),
            "context": _optional_list(dataset_record, config.dataset.context_field),
            "retrieval_context": _optional_list(dataset_record, config.dataset.retrieval_context_field),
            "additional_metadata": _optional_metadata(dataset_record, config.dataset.metadata_field),
            "model_name": _optional_text(output_record, "model_name"),
            "backend": _optional_text(output_record, "backend"),
            "latency_ms": _optional_number(output_record, "latency_ms"),
            "prompt_tokens": _optional_number(output_record, "prompt_tokens"),
            "completion_tokens": _optional_number(output_record, "completion_tokens"),
            "finish_reason": _optional_text(output_record, "finish_reason"),
            "inference_error": _optional_text(output_record, "error"),
        }
        missing = [field for field in required_fields if joined.get(field) in (None, "")]
        if missing:
            raise ValueError(f"Record '{record_id}' is missing required eval fields: {sorted(missing)}")
        joined_records.append(joined)
    return joined_records


def load_dataset(dataset_config: DatasetConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with dataset_config.path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in dataset at line {line_number}: {dataset_config.path}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Dataset line {line_number} must decode to an object.")
            records.append(record)
    if dataset_config.max_samples is not None:
        return records[: dataset_config.max_samples]
    return records


def load_outputs(model_outputs_config: ModelOutputsConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with model_outputs_config.path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in model outputs at line {line_number}: {model_outputs_config.path}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Model output line {line_number} must decode to an object.")
            records.append(record)
    if not records:
        raise ValueError(f"No model outputs were loaded from {model_outputs_config.path}")
    return records


def _required_text(record: dict[str, Any], field_name: str | None, label: str) -> str:
    if not field_name:
        raise ValueError(f"A field name is required for {label}.")
    value = record.get(field_name)
    if value in (None, ""):
        record_id = record.get("id", "<unknown>")
        detail = ""
        if label == "model output":
            raw_error = record.get("error")
            if raw_error not in (None, ""):
                detail = f" Upstream inference error for id '{record_id}': {raw_error}"
        raise ValueError(
            f"Missing required field '{field_name}' for {label} in record '{record_id}'.{detail}"
        )
    return str(value)


def _optional_text(record: dict[str, Any], field_name: str | None) -> str | None:
    if not field_name:
        return None
    value = record.get(field_name)
    if value in (None, ""):
        return None
    return str(value)


def _optional_list(record: dict[str, Any], field_name: str | None) -> list[str] | None:
    if not field_name:
        return None
    value = record.get(field_name)
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _optional_metadata(record: dict[str, Any], field_name: str | None) -> dict[str, Any] | None:
    if not field_name:
        return None
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Metadata field '{field_name}' must contain an object when provided.")
    return value


def _optional_number(record: dict[str, Any], field_name: str) -> int | float | None:
    value = record.get(field_name)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    return None
