"""Coordinate the full alignment flow from config loading through training and export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import (
    build_run_manifest,
    resolve_adapter_path,
    resolve_export_path,
    write_manifest,
)
from .config import load_alignment_config, validate_alignment_config
from .dataset_prep import prepare_alignment_dataset
from .llama_factory_adapter import build_export_config, build_train_config
from .runner import build_export_command, build_train_command, run_command, write_yaml
from .schemas import AlignmentResult


PPO_COMPATIBILITY_SKIP_REASON = (
    "PPO execution through the vendored LLaMA-Factory backend is temporarily skipped. "
    "This LLaMA-Factory copy has incompatible TRL expectations between its declared "
    "requirements and its vendored PPO trainer. To resolve this, please refer to the "
    "README file."
)


def run_alignment(config_path: str | Path) -> dict[str, Any]:
    """Run one alignment task and return the key artifact locations."""

    config = load_alignment_config(config_path)
    validate_alignment_config(config)

    run_dir = _prepare_run_dir(config.output.root_dir / config.run_name, config.output.overwrite_output_dir)
    prepared_dataset = prepare_alignment_dataset(config, run_dir)

    generated_dir = run_dir / "generated_configs"
    logs_dir = run_dir / "logs"
    adapter_path = resolve_adapter_path(run_dir)
    train_config_path = generated_dir / "train.yaml"

    train_payload = build_train_config(config, prepared_dataset, adapter_path)
    write_yaml(train_config_path, train_payload)

    train_command = build_train_command(config, train_config_path)
    if config.method == "ppo":
        return _skip_ppo_alignment(
            config=config,
            run_dir=run_dir,
            adapter_path=adapter_path,
            train_config_path=train_config_path,
            train_command=train_command,
            logs_dir=logs_dir,
            prepared_dataset=prepared_dataset,
        )

    train_result = run_command(train_command, logs_dir / "train.log", config, "train")

    export_config_path: Path | None = None
    export_result = None
    exported_model_path: Path | None = None

    if config.output.export_merged_model:
        exported_model_path = resolve_export_path(run_dir)
        export_config_path = generated_dir / "export.yaml"
        export_payload = build_export_config(config, adapter_path, exported_model_path)
        write_yaml(export_config_path, export_payload)
        export_command = build_export_command(config, export_config_path)
        export_result = run_command(export_command, logs_dir / "export.log", config, "export")

    manifest_path = run_dir / "manifest.json"
    result = AlignmentResult(
        run_dir=run_dir,
        adapter_path=adapter_path,
        exported_model_path=exported_model_path,
        train_config_path=train_config_path,
        export_config_path=export_config_path,
        manifest_path=manifest_path,
        method=config.method,
        backend=config.backend,
    )
    manifest_payload = build_run_manifest(config, result, prepared_dataset, train_result, export_result)
    write_manifest(manifest_path, manifest_payload)

    primary_artifact_path = result.exported_model_path or result.adapter_path
    return {
        "status": "completed",
        "run_name": config.run_name,
        "method": result.method,
        "backend": result.backend,
        "base_model": config.model.base_model_name_or_path,
        "primary_artifact_path": str(primary_artifact_path),
        "artifacts": {
            "adapter_path": str(result.adapter_path),
            "exported_model_path": str(result.exported_model_path)
            if result.exported_model_path is not None
            else None,
            "manifest_path": str(result.manifest_path),
        },
        "logs": {
            "train_log_path": str(train_result.log_path),
            "export_log_path": str(export_result.log_path) if export_result is not None else None,
        },
    }


def _prepare_run_dir(run_dir: Path, overwrite_output_dir: bool) -> Path:
    """Create the run directory or fail if the target already exists without overwrite."""

    if run_dir.exists() and not overwrite_output_dir:
        raise ValueError(
            f"Run directory already exists and overwrite_output_dir=false: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _skip_ppo_alignment(
    *,
    config,
    run_dir: Path,
    adapter_path: Path,
    train_config_path: Path,
    train_command: list[str],
    logs_dir: Path,
    prepared_dataset,
) -> dict[str, Any]:
    """Record a deliberate PPO skip while preserving generated backend config."""

    logs_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = logs_dir / "train.log"
    log_lines = [
        "[alignment] PPO LLaMA-Factory execution skipped.",
        f"[alignment] Reason: {PPO_COMPATIBILITY_SKIP_REASON}",
        f"[alignment] Generated train config: {train_config_path}",
        f"[alignment] Command not executed: {' '.join(train_command)}",
    ]
    train_log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    for line in log_lines:
        print(line)

    manifest_path = run_dir / "manifest.json"
    write_manifest(
        manifest_path,
        {
            "status": "skipped",
            "reason": PPO_COMPATIBILITY_SKIP_REASON,
            "backend": config.backend,
            "method": config.method,
            "run_name": config.run_name,
            "config_path": str(config.config_path),
            "run_dir": str(run_dir),
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
                "adapter_path": str(adapter_path),
                "exported_model_path": None,
                "train_config_path": str(train_config_path),
                "export_config_path": None,
            },
            "commands": {
                "train": {
                    "command": train_command,
                    "log_path": str(train_log_path),
                    "return_code": None,
                    "duration_seconds": 0,
                    "launcher_python": config.runtime.python_executable,
                    "executed": False,
                },
                "export": None,
            },
        },
    )

    return {
        "status": "skipped",
        "reason": PPO_COMPATIBILITY_SKIP_REASON,
        "run_name": config.run_name,
        "method": config.method,
        "backend": config.backend,
        "base_model": config.model.base_model_name_or_path,
        "primary_artifact_path": None,
        "artifacts": {
            "adapter_path": str(adapter_path),
            "exported_model_path": None,
            "manifest_path": str(manifest_path),
            "train_config_path": str(train_config_path),
        },
        "logs": {
            "train_log_path": str(train_log_path),
            "export_log_path": None,
        },
    }
