from __future__ import annotations

from pathlib import Path
from typing import Any

from alignment import run_alignment
from eval import run_evaluation
from inference import run_inference


def run_alignment_stage(config_path: str | Path) -> dict[str, Any]:
    return run_alignment(config_path)


def run_inference_stage(config_path: str | Path) -> dict[str, Any]:
    return run_inference(config_path)


def run_evaluation_stage(config_path: str | Path) -> dict[str, Any]:
    return run_evaluation(config_path)
