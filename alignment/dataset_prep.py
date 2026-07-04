"""Materialize a local LLaMA-Factory dataset bundle for one alignment run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .schemas import AlignmentConfig, PreparedDataset


def prepare_alignment_dataset(config: AlignmentConfig, run_dir: Path) -> PreparedDataset:
    """Copy the source dataset into the run folder and write dataset_info.json."""

    dataset_dir = run_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    materialized_data_path = dataset_dir / config.dataset.path.name
    shutil.copy2(config.dataset.path, materialized_data_path)

    dataset_name = _build_dataset_name(config.run_name, config.method)
    dataset_info_path = dataset_dir / "dataset_info.json"
    payload = {
        dataset_name: _build_dataset_info_entry(config, materialized_data_path.name),
    }
    dataset_info_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    return PreparedDataset(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        dataset_info_path=dataset_info_path,
        materialized_data_path=materialized_data_path,
    )


def _build_dataset_name(run_name: str, method: str) -> str:
    """Create a stable local dataset identifier for this run."""

    raw_name = f"{run_name}_{method}"
    normalized = "".join(char if char.isalnum() else "_" for char in raw_name.lower())
    return normalized.strip("_") or "alignment_dataset"


def _build_dataset_info_entry(config: AlignmentConfig, file_name: str) -> dict[str, object]:
    """Build the dataset_info.json entry that matches the selected dataset schema."""

    columns: dict[str, str] = {"prompt": config.dataset.prompt_field}

    if config.dataset.schema == "prompt_response":
        assert config.dataset.response_field is not None
        columns["response"] = config.dataset.response_field
    elif config.dataset.schema == "prompt_response_label":
        assert config.dataset.response_field is not None
        assert config.dataset.label_field is not None
        columns["response"] = config.dataset.response_field
        columns["kto_tag"] = config.dataset.label_field
    elif config.dataset.schema == "prompt_chosen_rejected":
        assert config.dataset.chosen_field is not None
        assert config.dataset.rejected_field is not None
        columns["chosen"] = config.dataset.chosen_field
        columns["rejected"] = config.dataset.rejected_field

    if config.dataset.system_field:
        columns["system"] = config.dataset.system_field

    entry: dict[str, object] = {
        "file_name": file_name,
        "columns": columns,
    }

    if config.dataset.schema == "prompt_chosen_rejected":
        entry["ranking"] = True

    return entry
