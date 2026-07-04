from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MasterConfig:
    """Store the user-facing orchestrator config and resolved run settings."""

    path: Path
    raw: dict[str, Any]
    run_name: str
    methods: list[str]
    execute_alignment: bool
    execute_inference: bool
    execute_evaluation: bool
    output_root: Path
    overwrite: bool
    save_generated_stage_configs: bool
    save_pipeline_state: bool
    stop_on_error: bool


@dataclass(slots=True)
class StagePlan:
    """Describe one concrete branch that the orchestrator will execute."""

    branch_id: str
    candidate_id: str | None
    candidate_kind: str | None
    method: str | None
    run_name: str
    inference_benchmark_id: str | None = None
    evaluation_benchmark_id: str | None = None
    external_model_outputs_path: str | None = None


@dataclass(slots=True)
class PipelineState:
    """Persist the global orchestration status and stage handoff data."""

    run_id: str
    run_dir: Path
    state_path: Path
    payload: dict[str, Any]
