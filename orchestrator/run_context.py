from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .schemas import MasterConfig, PipelineState


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_run_id(run_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    normalized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in run_name)
    normalized = normalized.strip("-") or "run"
    return f"{normalized}-{timestamp}"


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def prepare_run_dir(config: MasterConfig) -> tuple[str, Path]:
    run_id = build_run_id(config.run_name)
    run_dir = config.output_root / run_id
    if run_dir.exists() and not config.overwrite:
        raise ValueError(
            f"Orchestrator run directory already exists and overwrite=false: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def initialize_pipeline_state(
    config: MasterConfig,
    run_id: str,
    run_dir: Path,
) -> PipelineState:
    state_path = run_dir / "pipeline_state.json"
    payload = {
        "run_id": run_id,
        "run_name": config.run_name,
        "status": "pending",
        "master_config_path": str(config.path),
        "run_dir": str(run_dir),
        "created_at_utc": utc_now(),
        "execute": {
            "alignment": config.execute_alignment,
            "inference": config.execute_inference,
            "evaluation": config.execute_evaluation,
        },
        "methods": config.methods,
        "branches": [],
    }
    return PipelineState(
        run_id=run_id,
        run_dir=run_dir,
        state_path=state_path,
        payload=payload,
    )


def persist_state(state: PipelineState, enabled: bool) -> None:
    if not enabled:
        return
    write_json(state.state_path, state.payload)
