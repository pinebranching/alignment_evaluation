"""Persist run manifests and resolve the canonical alignment artifact paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import AlignmentConfig, AlignmentResult, CommandResult, PreparedDataset


def build_run_manifest(
    config: AlignmentConfig,
    result: AlignmentResult,
    prepared_dataset: PreparedDataset,
    train_result: CommandResult,
    export_result: CommandResult | None,
) -> dict[str, Any]:
    """Build the manifest that downstream modules and users can consume."""

    return {
        "backend": config.backend,
        "method": config.method,
        "run_name": config.run_name,
        "config_path": str(config.config_path),
        "run_dir": str(result.run_dir),
        "dataset": {
            "source_path": str(config.dataset.path),
            "materialized_path": str(prepared_dataset.materialized_data_path),
            "dataset_info_path": str(prepared_dataset.dataset_info_path),
            "schema": config.dataset.schema,
            "dataset_name": prepared_dataset.dataset_name,
        },
        "model": {
            "base_model_name_or_path": config.model.base_model_name_or_path,
            "template": config.model.template,
            "finetuning_type": config.tuning.finetuning_type,
        },
        "artifacts": {
            "adapter_path": str(result.adapter_path),
            "exported_model_path": str(result.exported_model_path)
            if result.exported_model_path is not None
            else None,
            "train_config_path": str(result.train_config_path),
            "export_config_path": str(result.export_config_path)
            if result.export_config_path is not None
            else None,
        },
        "commands": {
            "train": {
                "command": train_result.command,
                "log_path": str(train_result.log_path),
                "return_code": train_result.return_code,
                "duration_seconds": train_result.duration_seconds,
                "launcher_python": train_result.launcher_python,
            },
            "export": {
                "command": export_result.command,
                "log_path": str(export_result.log_path),
                "return_code": export_result.return_code,
                "duration_seconds": export_result.duration_seconds,
                "launcher_python": export_result.launcher_python,
            }
            if export_result is not None
            else None,
        },
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write the manifest JSON file for one alignment run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def resolve_export_path(run_dir: Path) -> Path:
    """Return the canonical merged-model export directory for a run."""

    return run_dir / "artifacts" / "exported_model"


def resolve_adapter_path(run_dir: Path) -> Path:
    """Return the canonical adapter/checkpoint directory for a run."""

    return run_dir / "artifacts" / "adapter"
