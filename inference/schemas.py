from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class InferenceDatasetConfig:
    path: Path
    format: str
    id_field: str
    input_field: str
    max_samples: int | None = None
    expected_output_field: str | None = None
    context_field: str | None = None
    retrieval_context_field: str | None = None
    metadata_field: str | None = None


@dataclass(slots=True)
class InferenceModelConfig:
    backend: str
    model_name_or_path: str
    adapter_name_or_path: str | None = None
    tokenizer_name_or_path: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    torch_dtype: str | None = None
    device: str = "auto"


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool | None = None
    repetition_penalty: float = 1.2  # Penalizes repeated tokens to avoid degenerate repetitive output. 1.0 = no penalty, >1.0 = penalty strength.
    stop_strings: list[str] = field(default_factory=list)
    add_generation_prompt: bool = True
    enable_thinking: bool | None = None


@dataclass(slots=True)
class InferenceExecutionConfig:
    batch_size: int = 1
    max_concurrency: int = 1


@dataclass(slots=True)
class InferenceRuntimeConfig:
    target: str = "cpu"
    cpu_threads: int = 0
    cuda_visible_devices: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InferenceOutputConfig:
    path: Path
    overwrite: bool = True
    save_manifest: bool = True


@dataclass(slots=True)
class InferenceConfig:
    mode: str
    dataset: InferenceDatasetConfig
    model: InferenceModelConfig
    generation: GenerationConfig
    execution: InferenceExecutionConfig
    runtime: InferenceRuntimeConfig
    output: InferenceOutputConfig
    config_path: Path
    raw_inference_config: dict[str, Any]


@dataclass(slots=True)
class InferenceInputRecord:
    id: str
    input: str
    expected_output: str | None = None
    context: list[str] | None = None
    retrieval_context: list[str] | None = None
    additional_metadata: dict[str, Any] | None = None

