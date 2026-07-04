from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DatasetConfig:
    path: Path
    format: str
    input_field: str
    expected_output_field: str | None
    id_field: str
    max_samples: int | None = None
    context_field: str | None = None
    retrieval_context_field: str | None = None
    metadata_field: str | None = None


@dataclass(slots=True)
class ModelOutputsConfig:
    source: str
    path: Path
    actual_output_field: str
    id_field: str | None = None


@dataclass(slots=True)
class JudgeModelConfig:
    backend: str
    model: str
    tokenizer_name_or_path: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    torch_dtype: str | None = None
    device: str = "auto"
    temperature: float | None = 0.0
    generation_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricConfig:
    name: str
    threshold: float | None = None
    include_reason: bool = True
    strict_mode: bool = False
    verbose_mode: bool = False


@dataclass(slots=True)
class ExecutionConfig:
    batch_size: int = 0
    max_concurrency: int = 5


@dataclass(slots=True)
class EvalRuntimeConfig:
    target: str = "cpu"
    cpu_threads: int = 0
    cuda_visible_devices: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OutputConfig:
    dir: Path
    save_json: bool = True
    save_csv: bool = True
    save_visualizations: bool = True
    save_config_snapshot: bool = True


@dataclass(slots=True)
class EvalConfig:
    mode: str
    dataset: DatasetConfig
    model_outputs: ModelOutputsConfig
    judge_model: JudgeModelConfig | None
    metrics: list[MetricConfig]
    execution: ExecutionConfig
    runtime: EvalRuntimeConfig
    output: OutputConfig
    config_path: Path
    raw_eval_config: dict[str, Any]
