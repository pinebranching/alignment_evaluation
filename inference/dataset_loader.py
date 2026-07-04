from __future__ import annotations

import json
from typing import Any

from .schemas import InferenceDatasetConfig, InferenceInputRecord


def load_dataset_records(dataset_config: InferenceDatasetConfig) -> list[InferenceInputRecord]:
    records: list[InferenceInputRecord] = []
    with dataset_config.path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in dataset at line {line_number}: {dataset_config.path}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Dataset line {line_number} must decode to an object.")
            records.append(_to_input_record(row, dataset_config))

    if not records:
        raise ValueError(f"No dataset records were loaded from {dataset_config.path}")
    if dataset_config.max_samples is not None:
        return records[: dataset_config.max_samples]
    return records


def _to_input_record(
    row: dict[str, Any], dataset_config: InferenceDatasetConfig
) -> InferenceInputRecord:
    raw_id = row.get(dataset_config.id_field)
    if raw_id in (None, ""):
        raise ValueError(f"Missing dataset id field '{dataset_config.id_field}'.")
    raw_input = row.get(dataset_config.input_field)
    if raw_input in (None, ""):
        raise ValueError(f"Missing dataset input field '{dataset_config.input_field}'.")

    return InferenceInputRecord(
        id=str(raw_id),
        input=str(raw_input),
        expected_output=_optional_text(row, dataset_config.expected_output_field),
        context=_optional_list(row, dataset_config.context_field),
        retrieval_context=_optional_list(row, dataset_config.retrieval_context_field),
        additional_metadata=_optional_metadata(row, dataset_config.metadata_field),
    )


def _optional_text(row: dict[str, Any], field_name: str | None) -> str | None:
    if not field_name:
        return None
    value = row.get(field_name)
    if value in (None, ""):
        return None
    return str(value)

# todo: remove method as only used once and add inline
def _optional_list(row: dict[str, Any], field_name: str | None) -> list[str] | None:
    if not field_name:
        return None
    value = row.get(field_name)
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _optional_metadata(row: dict[str, Any], field_name: str | None) -> dict[str, Any] | None:
    if not field_name:
        return None
    raw_meta = row.get(field_name)
    if raw_meta is None:
        return None
    if not isinstance(raw_meta, dict):
        raise ValueError(
            f"Dataset metadata field '{field_name}' must contain an object."
        )
    return raw_meta

