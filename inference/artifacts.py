from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return path


def write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return path


def build_run_manifest(
    *,
    config_path: Path,
    dataset_path: Path,
    output_path: Path,
    model_name: str,
    backend: str,
    total_records: int,
    succeeded: int,
    failed: int,
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "model_name": model_name,
        "backend": backend,
        "total_records": total_records,
        "succeeded": succeeded,
        "failed": failed,
    }

