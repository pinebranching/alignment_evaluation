from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schemas import (
    GenerationConfig,
    InferenceConfig,
    InferenceDatasetConfig,
    InferenceExecutionConfig,
    InferenceModelConfig,
    InferenceOutputConfig,
    InferenceRuntimeConfig,
)


def load_inference_config(config_path: str | Path) -> InferenceConfig:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root in {path} must be a dict.")
    inf_section = raw.get("inference", raw)
    if not isinstance(inf_section, dict):
        raise ValueError("The inference config must be a dict.")

    base_dir = path.parent
    dataset_raw = _require_mapping(inf_section.get("dataset"), "dataset")
    model_raw = _require_mapping(inf_section.get("model"), "model")
    generation_raw = _optional_mapping(inf_section.get("generation"), "generation")
    execution_raw = _optional_mapping(inf_section.get("execution"), "execution")
    runtime_raw = _optional_mapping(inf_section.get("runtime"), "runtime")
    output_raw = _require_mapping(inf_section.get("output"), "output")

    return InferenceConfig(
        mode=str(inf_section.get("mode", "offline")),
        dataset=InferenceDatasetConfig(
            path=_resolve_path(base_dir, dataset_raw.get("path")),
            format=str(dataset_raw.get("format", "jsonl")),
            id_field=str(dataset_raw.get("id_field", "id")),
            input_field=str(dataset_raw.get("input_field", "input")),
            max_samples=int(v) if (v := dataset_raw.get("max_samples")) is not None else None,
            expected_output_field=str(v)
            if (v := dataset_raw.get("expected_output_field")) is not None
            else None,
            context_field=str(v) if (v := dataset_raw.get("context_field")) is not None else None,
            retrieval_context_field=str(v)
            if (v := dataset_raw.get("retrieval_context_field")) is not None
            else None,
            metadata_field=str(v) if (v := dataset_raw.get("metadata_field")) is not None else None,
        ),
        model=InferenceModelConfig(
            backend=str(model_raw.get("backend", "transformers_local")),
            model_name_or_path=str(model_raw.get("model_name_or_path", "")),
            adapter_name_or_path=str(v)
            if (v := model_raw.get("adapter_name_or_path")) is not None
            else None,
            tokenizer_name_or_path=str(v)
            if (v := model_raw.get("tokenizer_name_or_path")) is not None
            else None,
            revision=str(v) if (v := model_raw.get("revision")) is not None else None,
            trust_remote_code=bool(model_raw.get("trust_remote_code", False)),
            torch_dtype=str(v) if (v := model_raw.get("torch_dtype")) is not None else None,
            device=str(model_raw.get("device", "auto")),
        ),
        generation=GenerationConfig(
            max_new_tokens=int(generation_raw.get("max_new_tokens", 256)),
            temperature=float(generation_raw.get("temperature", 0.0)),
            top_p=float(generation_raw.get("top_p", 1.0)),
            do_sample=bool(v) if (v := generation_raw.get("do_sample")) is not None else None,
            repetition_penalty=float(generation_raw.get("repetition_penalty", 1.0)),
            stop_strings=_as_string_list(generation_raw.get("stop_strings", [])),
            add_generation_prompt=bool(generation_raw.get("add_generation_prompt", True)),
            enable_thinking=bool(v)
            if (v := generation_raw.get("enable_thinking")) is not None
            else None,
        ),
        execution=InferenceExecutionConfig(
            batch_size=int(execution_raw.get("batch_size", 1)),
            max_concurrency=int(execution_raw.get("max_concurrency", 1)),
        ),
        runtime=InferenceRuntimeConfig(
            target=str(runtime_raw.get("target", "cpu")),
            cpu_threads=int(runtime_raw.get("cpu_threads", 0)),
            cuda_visible_devices=_as_string_list(runtime_raw.get("cuda_visible_devices", [])),
        ),
        output=InferenceOutputConfig(
            path=_resolve_path(base_dir, output_raw.get("path")),
            overwrite=bool(output_raw.get("overwrite", True)),
            save_manifest=bool(output_raw.get("save_manifest", True)),
        ),
        config_path=path,
        raw_inference_config=inf_section,
    )


def validate_inference_config(config: InferenceConfig) -> None:
    if config.mode != "offline":
        raise ValueError("Only inference.mode='offline' is supported right now.")
    if config.dataset.format.lower() != "jsonl":
        raise ValueError("Only dataset.format='jsonl' is supported right now.")
    if config.model.backend.strip().lower() != "transformers_local":
        raise ValueError("Only model.backend='transformers_local' is supported.")
    if not config.model.model_name_or_path:
        raise ValueError("model.model_name_or_path must not be empty.")
    if not config.dataset.path.exists():
        raise ValueError(f"Dataset file does not exist: {config.dataset.path}")
    if config.dataset.max_samples is not None and config.dataset.max_samples <= 0:
        raise ValueError("dataset.max_samples must be > 0 when provided.")
    if config.execution.batch_size <= 0:
        raise ValueError("execution.batch_size must be > 0.")
    if config.execution.max_concurrency <= 0:
        raise ValueError("execution.max_concurrency must be > 0.")
    if config.runtime.target not in {"cpu", "single_gpu", "multi_gpu"}:
        raise ValueError("runtime.target must be one of: cpu, single_gpu, multi_gpu.")
    if config.runtime.cpu_threads < 0:
        raise ValueError("runtime.cpu_threads must be >= 0.")
    if config.runtime.target == "cpu" and config.runtime.cuda_visible_devices:
        raise ValueError("runtime.cuda_visible_devices must be empty when runtime.target='cpu'.")
    if config.generation.max_new_tokens <= 0:
        raise ValueError("generation.max_new_tokens must be > 0.")
    if config.generation.temperature < 0:
        raise ValueError("generation.temperature must be >= 0.")
    if not 0 < config.generation.top_p <= 1:
        raise ValueError("generation.top_p must be in (0, 1].")
    if config.generation.repetition_penalty <= 0:
        raise ValueError("generation.repetition_penalty must be > 0.")
    if config.output.path.exists() and not config.output.overwrite:
        raise ValueError(
            f"Output path already exists and overwrite=false: {config.output.path}"
        )


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None or not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _optional_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping when provided.")
    return value


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("generation.stop_strings must be a list when provided.")
    return [str(item) for item in value]


def _resolve_path(base_dir: Path, raw_path: Any) -> Path:
    if raw_path in (None, ""):
        raise ValueError("A required path field is missing or empty.")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path

