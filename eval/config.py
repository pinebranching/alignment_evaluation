from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schemas import (
    DatasetConfig,
    EvalConfig,
    ExecutionConfig,
    EvalRuntimeConfig,
    JudgeModelConfig,
    MetricConfig,
    ModelOutputsConfig,
    OutputConfig,
)


def load_eval_config(config_path: str | Path) -> EvalConfig:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root in {path} must be a dict.")
    eval_section = raw.get("eval", raw)
    if not isinstance(eval_section, dict):
        raise ValueError("The evaluation config must be a dict.")

    base_dir = path.parent

    # check types of nested sections if they exist, to provide better error messages
    dataset_raw = eval_section.get("dataset")
    if dataset_raw and not isinstance(dataset_raw, dict):
        raise ValueError(f"dataset must be a dict if provided.")
    
    model_outputs_raw = eval_section.get("model_outputs")
    if model_outputs_raw and not isinstance(model_outputs_raw, dict):
        raise ValueError(f"model_outputs must be a dict if provided.")
    
    output_raw = eval_section.get("output")
    if output_raw and not isinstance(output_raw, dict):
        raise ValueError(f"output must be a dict if provided.")

    execution_raw = eval_section.get("execution", {})
    if execution_raw and not isinstance(execution_raw, dict):
        raise ValueError("execution must be a dict if provided.")

    runtime_raw = eval_section.get("runtime", {})
    if runtime_raw and not isinstance(runtime_raw, dict):
        raise ValueError("runtime must be a dict if provided.")

    # instantiate EvalConfig with defaults, then override with any provided config values
    config = EvalConfig(
        mode=str(eval_section.get("mode", "offline")),
        dataset=DatasetConfig(
            path=_resolve_path(base_dir, dataset_raw.get("path")),
            format=str(dataset_raw.get("format", "jsonl")),
            input_field=str(dataset_raw.get("input_field", "input")),
            expected_output_field=str(v) if (v := dataset_raw.get("expected_output_field")) is not None else None,
            id_field=str(dataset_raw.get("id_field", "id")),
            max_samples=int(v) if (v := dataset_raw.get("max_samples")) is not None else None,
            context_field=str(v) if (v := dataset_raw.get("context_field")) is not None else None,
            retrieval_context_field=str(v) if (v := dataset_raw.get("retrieval_context_field")) is not None else None,
            metadata_field=str(v) if (v := dataset_raw.get("metadata_field")) is not None else None,
        ),
        model_outputs=ModelOutputsConfig(
            source=str(model_outputs_raw.get("source", "file")),
            path=_resolve_path(base_dir, model_outputs_raw.get("path")),
            actual_output_field=str(model_outputs_raw.get("actual_output_field", "actual_output")),
            id_field=str(v) if (v := model_outputs_raw.get("id_field")) is not None else None,
        ),
        judge_model=_build_judge_model_config(eval_section.get("judge_model")),
        metrics=_build_metrics(eval_section.get("metrics", [])),
        execution=ExecutionConfig(
            batch_size=int(execution_raw.get("batch_size", 0)),
            max_concurrency=int(execution_raw.get("max_concurrency", 20)),
        ),
        runtime=EvalRuntimeConfig(
            target=str(runtime_raw.get("target", "cpu")),
            cpu_threads=int(runtime_raw.get("cpu_threads", 0)),
            cuda_visible_devices=_as_string_list(runtime_raw.get("cuda_visible_devices", [])),
        ),
        output=OutputConfig(
            dir=_resolve_path(base_dir, output_raw.get("dir")),
            save_json=bool(output_raw.get("save_json", True)),
            save_csv=bool(output_raw.get("save_csv", True)),
            save_visualizations=bool(output_raw.get("save_visualizations", True)),
            save_config_snapshot=bool(output_raw.get("save_config_snapshot", True)),
        ),
        config_path=path,
        raw_eval_config=eval_section,
    )
    return config


def validate_eval_config(config: EvalConfig) -> None:
    if config.mode != "offline":
        raise ValueError("Only eval.mode='offline' is supported in this implementation.")

    if config.dataset.format.lower() != "jsonl":
        raise ValueError("Only dataset.format='jsonl' is supported right now.")

    if config.model_outputs.source != "file":
        raise ValueError("Only model_outputs.source='file' is supported right now.")

    if not config.dataset.path.exists():
        raise ValueError(f"Dataset file does not exist: {config.dataset.path}")

    if config.dataset.max_samples is not None and config.dataset.max_samples <= 0:
        raise ValueError("dataset.max_samples must be > 0 when provided.")

    if not config.model_outputs.path.exists():
        raise ValueError(f"Model outputs file does not exist: {config.model_outputs.path}")

    if config.execution.batch_size < 0:
        raise ValueError("execution.batch_size must be >= 0")
    if config.execution.max_concurrency <= 0:
        raise ValueError("execution.max_concurrency must be > 0")
    if config.runtime.target not in {"cpu", "single_gpu", "multi_gpu"}:
        raise ValueError("runtime.target must be one of: cpu, single_gpu, multi_gpu.")
    if config.runtime.cpu_threads < 0:
        raise ValueError("runtime.cpu_threads must be >= 0.")
    if config.runtime.target == "cpu" and config.runtime.cuda_visible_devices:
        raise ValueError("runtime.cuda_visible_devices must be empty when runtime.target='cpu'.")

    if not config.metrics:
        raise ValueError("At least one metric must be configured.")

    metric_names: set[str] = set()
    for metric in config.metrics:
        normalized_name = metric.name.strip().lower()
        if normalized_name in metric_names:
            raise ValueError(f"Duplicate metric configured: {metric.name}")
        metric_names.add(normalized_name)
        if metric.threshold is not None and not 0 <= metric.threshold <= 1:
            raise ValueError(f"Metric threshold for '{metric.name}' must be between 0 and 1.")

    from .policies.metric_requirements import metrics_require_judge

    if metrics_require_judge(config.metrics) and config.judge_model is None:
        raise ValueError("A judge_model must be configured when using LLM-as-a-judge metrics.")

    if config.judge_model is not None and not config.judge_model.model:
        raise ValueError("judge_model.model must not be empty.")
    if config.judge_model is not None:
        backend = config.judge_model.backend.strip().lower()
        if backend not in {"transformers_local", "huggingface", "hf"}:
            raise ValueError(
                "judge_model.backend must be one of: transformers_local, huggingface, hf."
            )


def _build_judge_model_config(raw_value: Any) -> JudgeModelConfig | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("judge_model must be a mapping.")
    
    generation_kwargs = raw_value.get("generation_kwargs")
    if generation_kwargs is None:
        generation_kwargs = {}
    elif not isinstance(generation_kwargs, dict):
        raise ValueError("generation_kwargs must be a mapping when provided.")
    
    return JudgeModelConfig(
        backend=str(raw_value.get("backend", "transformers_local")),
        model=str(raw_value.get("model", raw_value.get("model_name_or_path", ""))),
        tokenizer_name_or_path=str(v)
        if (v := raw_value.get("tokenizer_name_or_path")) is not None
        else None,
        revision=str(v) if (v := raw_value.get("revision")) is not None else None,
        trust_remote_code=bool(raw_value.get("trust_remote_code", False)),
        torch_dtype=str(v) if (v := raw_value.get("torch_dtype")) is not None else None,
        device=str(raw_value.get("device", "auto")),
        temperature=float(v) if (v := raw_value.get("temperature")) is not None else None,
        generation_kwargs=generation_kwargs,
    )


def _build_metrics(metric_items: Any) -> list[MetricConfig]:
    if not isinstance(metric_items, list):
        raise ValueError("metrics must be a list.")
    metrics: list[MetricConfig] = []
    for item in metric_items:
        if isinstance(item, str):
            metrics.append(MetricConfig(name=item))
            continue
        if not isinstance(item, dict):
            raise ValueError("Each metric must be a string or mapping.")
        if "judge_model" in item:
            raise ValueError("Per-metric judge_model overrides are not supported in this version.")
        metrics.append(
            MetricConfig(
                name=str(item.get("name", "")),
                threshold=float(v) if (v := item.get("threshold")) is not None else None,
                include_reason=bool(item.get("include_reason", True)),
                strict_mode=bool(item.get("strict_mode", False)),
                verbose_mode=bool(item.get("verbose_mode", False)),
            )
        )
    return metrics


def _resolve_path(base_dir: Path, raw_path: Any) -> Path:
    if raw_path in (None, ""):
        raise ValueError("A required path field is missing or empty.")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("runtime.cuda_visible_devices must be a list when provided.")
    return [str(item) for item in value]
