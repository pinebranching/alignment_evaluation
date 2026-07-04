from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import build_run_manifest, write_jsonl, write_manifest
from .backend_factory import build_backend
from .config import load_inference_config, validate_inference_config
from .dataset_loader import load_dataset_records
from .huggingface_runtime import temporary_runtime_env
from .io import load_model_outputs, save_model_outputs
from .runner import run_inference_records


def run_inference(config_path: str | Path) -> dict[str, Any]:
    print(f"[inference] Loading config: {config_path}")
    config = load_inference_config(config_path)
    validate_inference_config(config)

    records = load_dataset_records(config.dataset)
    print(
        f"[inference] Running {len(records)} records with backend={config.model.backend} "
        f"model={config.model.model_name_or_path}"
    )
    with temporary_runtime_env(
        target=config.runtime.target,
        cuda_visible_devices=config.runtime.cuda_visible_devices,
        cpu_threads=config.runtime.cpu_threads,
    ):
        backend = build_backend(config.model, config.generation)
        output_rows, counters = run_inference_records(config, records, backend)

    write_jsonl(config.output.path, output_rows)

    manifest_path: str | None = None
    if config.output.save_manifest:
        manifest_payload = build_run_manifest(
            config_path=config.config_path,
            dataset_path=config.dataset.path,
            output_path=config.output.path,
            model_name=config.model.model_name_or_path,
            backend=config.model.backend,
            total_records=len(records),
            succeeded=counters["succeeded"],
            failed=counters["failed"],
        )
        manifest_file = config.output.path.parent / f"{config.output.path.stem}.manifest.json"
        write_manifest(manifest_file, manifest_payload)
        manifest_path = str(manifest_file)

    result = {
        "output_path": str(config.output.path),
        "manifest_path": manifest_path,
        "total_records": len(records),
        "succeeded": counters["succeeded"],
        "failed": counters["failed"],
    }
    print(
        f"[inference] Finished. output={result['output_path']} "
        f"succeeded={result['succeeded']} failed={result['failed']}"
    )
    return result


__all__ = ["run_inference", "load_model_outputs", "save_model_outputs"]
