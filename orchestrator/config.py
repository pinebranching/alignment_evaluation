from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from alignment.method_registry import get_method_spec
from eval.policies.metric_requirements import metrics_require_judge
from eval.schemas import MetricConfig

from .schemas import MasterConfig


def load_master_config(config_path: str | Path) -> MasterConfig:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root in {path} must be a mapping.")

    require_mapping(raw.get("orchestrator"), "orchestrator")
    require_mapping(raw.get("defaults"), "defaults")
    require_mapping(raw.get("catalog"), "catalog")
    require_mapping(raw.get("alignment"), "alignment")
    require_mapping(raw.get("inference"), "inference")
    require_mapping(raw.get("evaluation"), "evaluation")

    orchestrator_raw = raw["orchestrator"]
    execute_raw = optional_mapping(orchestrator_raw.get("execute"), "orchestrator.execute")
    output_raw = require_mapping(orchestrator_raw.get("output"), "orchestrator.output")

    candidate_items = require_list(raw["alignment"].get("candidates"), "alignment.candidates")
    methods: list[str] = []
    for item in candidate_items:
        if not isinstance(item, dict):
            raise ValueError("Each alignment candidate must be a mapping.")
        if str(item.get("kind", "")).strip().lower() != "aligned":
            continue
        method = str(item.get("method", "")).strip().lower()
        if method and method not in methods:
            methods.append(method)

    return MasterConfig(
        path=path,
        raw=raw,
        run_name=str(orchestrator_raw.get("run_name", "")).strip(),
        methods=methods,
        execute_alignment=bool(execute_raw.get("alignment", True)),
        execute_inference=bool(execute_raw.get("inference", True)),
        execute_evaluation=bool(execute_raw.get("evaluation", True)),
        output_root=resolve_path(path.parent, output_raw.get("root_dir")),
        overwrite=bool(output_raw.get("overwrite", False)),
        save_generated_stage_configs=bool(output_raw.get("save_generated_stage_configs", True)),
        save_pipeline_state=bool(output_raw.get("save_pipeline_state", True)),
        stop_on_error=bool(orchestrator_raw.get("stop_on_error", True)),
    )


def validate_master_config(config: MasterConfig) -> None:
    if not config.run_name:
        raise ValueError("orchestrator.run_name must not be empty.")

    if not any(
        [
            config.execute_alignment,
            config.execute_inference,
            config.execute_evaluation,
        ]
    ):
        raise ValueError("At least one stage must be enabled in orchestrator.execute.")
    if config.execute_alignment and config.execute_evaluation and not config.execute_inference:
        raise ValueError("Alignment -> evaluation without inference is not supported.")

    defaults_raw = require_mapping(config.raw.get("defaults"), "defaults")
    require_mapping(defaults_raw.get("alignment"), "defaults.alignment")
    require_mapping(defaults_raw.get("inference"), "defaults.inference")
    require_mapping(defaults_raw.get("evaluation"), "defaults.evaluation")
    catalog_raw = require_mapping(config.raw.get("catalog"), "catalog")

    validate_shared_runtime(require_mapping(defaults_raw.get("runtime"), "defaults.runtime"))

    candidate_index = get_candidate_index(config.raw)
    if not candidate_index:
        raise ValueError("alignment.candidates must be a non-empty list.")

    model_index = get_catalog_index(catalog_raw, "models", "catalog.models")
    dataset_index = get_catalog_index(catalog_raw, "datasets", "catalog.datasets")
    judge_models = get_catalog_index(catalog_raw, "judge_models", "catalog.judge_models")
    training_profiles = optional_mapping(
        catalog_raw.get("training_profiles"),
        "catalog.training_profiles",
    )
    metric_sets = optional_mapping(catalog_raw.get("metric_sets"), "catalog.metric_sets")

    for candidate_id, candidate in candidate_index.items():
        kind = str(candidate.get("kind", "")).strip().lower()
        if kind not in {"base", "aligned"}:
            raise ValueError(
                f"alignment.candidates[{candidate_id}].kind must be 'base' or 'aligned'."
            )

        if kind == "base":
            model_ref = str(candidate.get("model_ref", "")).strip()
            if not model_ref:
                raise ValueError(f"Base candidate '{candidate_id}' must define model_ref.")
            model_entry = require_mapping(
                model_index.get(model_ref),
                f"catalog.models.{model_ref}",
            )
            if not str(model_entry.get("model_name_or_path", "")).strip():
                raise ValueError(
                    f"catalog.models.{model_ref}.model_name_or_path must not be empty."
                )
            continue

        if not config.execute_alignment:
            raise ValueError(
                f"Aligned candidate '{candidate_id}' requires orchestrator.execute.alignment=true. "
                "For inference-only or evaluation-only runs, model the exported model as kind='base'."
            )

        method = str(candidate.get("method", "")).strip().lower()
        if not method:
            raise ValueError(f"Aligned candidate '{candidate_id}' must define method.")
        method_spec = get_method_spec(method)
        if not bool(method_spec["implemented"]):
            raise ValueError(
                f"Aligned candidate '{candidate_id}' uses method '{method}', which is not implemented yet."
            )
        method_config = optional_mapping(
            candidate.get("method_config"),
            f"alignment.candidates[{candidate_id}].method_config",
        )
        if method == "ppo":
            ppo_config = optional_mapping(
                method_config.get("ppo"),
                f"alignment.candidates[{candidate_id}].method_config.ppo",
            )
            reward_model_candidate_ref = str(ppo_config.get("reward_model_candidate_ref", "")).strip()
            if not str(
                ppo_config.get("reward_model_path", ppo_config.get("reward_model", ""))
            ).strip() and not reward_model_candidate_ref:
                raise ValueError(
                    f"Aligned candidate '{candidate_id}' uses PPO and must define "
                    "method_config.ppo.reward_model_path or method_config.ppo.reward_model_candidate_ref."
                )
            if reward_model_candidate_ref:
                reward_candidate = require_mapping(
                    candidate_index.get(reward_model_candidate_ref),
                    f"alignment.candidates[{candidate_id}].method_config.ppo.reward_model_candidate_ref",
                )
                if str(reward_candidate.get("kind", "")).strip().lower() != "aligned":
                    raise ValueError(
                        f"PPO candidate '{candidate_id}' references reward model candidate "
                        f"'{reward_model_candidate_ref}', which must be kind='aligned'."
                    )
                if str(reward_candidate.get("method", "")).strip().lower() != "rm":
                    raise ValueError(
                        f"PPO candidate '{candidate_id}' references reward model candidate "
                        f"'{reward_model_candidate_ref}', which must use method='rm'."
                    )
                if str(reward_candidate.get("base_model_ref", "")).strip() != str(
                    candidate.get("base_model_ref", "")
                ).strip():
                    raise ValueError(
                        f"PPO candidate '{candidate_id}' and reward model candidate "
                        f"'{reward_model_candidate_ref}' must share base_model_ref."
                    )

        base_model_ref = str(candidate.get("base_model_ref", "")).strip()
        if not base_model_ref:
            raise ValueError(f"Aligned candidate '{candidate_id}' must define base_model_ref.")
        model_entry = require_mapping(
            model_index.get(base_model_ref),
            f"catalog.models.{base_model_ref}",
        )
        if not str(model_entry.get("model_name_or_path", "")).strip():
            raise ValueError(
                f"catalog.models.{base_model_ref}.model_name_or_path must not be empty."
            )
        if not str(model_entry.get("template", "")).strip():
            raise ValueError(
                f"catalog.models.{base_model_ref}.template must not be empty for aligned candidates."
            )

        dataset_ref = str(candidate.get("dataset_ref", "")).strip()
        if not dataset_ref:
            raise ValueError(f"Aligned candidate '{candidate_id}' must define dataset_ref.")
        dataset_entry = require_mapping(
            dataset_index.get(dataset_ref),
            f"catalog.datasets.{dataset_ref}",
        )
        validate_alignment_dataset_entry(config.path.parent, dataset_ref, dataset_entry, method)

        profile_ref = candidate.get("profile_ref")
        profile_payload: dict[str, Any] = {}
        if profile_ref is not None:
            profile_key = str(profile_ref).strip()
            if not profile_key:
                raise ValueError(f"Aligned candidate '{candidate_id}' has an empty profile_ref.")
            profile_payload = require_mapping(
                training_profiles.get(profile_key),
                f"catalog.training_profiles.{profile_key}",
            )
        resolved_training = merge_mappings(
            require_mapping(defaults_raw["alignment"].get("training"), "defaults.alignment.training"),
            optional_mapping(profile_payload.get("training"), f"catalog.training_profiles.{profile_ref}.training")
            if profile_ref is not None
            else {},
            optional_mapping(candidate.get("training"), f"alignment.candidates[{candidate_id}].training"),
        )
        validate_alignment_training_payload(resolved_training, candidate_id)

        resolved_output = merge_mappings(
            require_mapping(defaults_raw["alignment"].get("output"), "defaults.alignment.output"),
            optional_mapping(candidate.get("output"), f"alignment.candidates[{candidate_id}].output"),
        )
        if (
            config.execute_inference
            and method != "rm"
            and not bool(resolved_output.get("export_merged_model", False))
        ):
            raise ValueError(
                f"Aligned candidate '{candidate_id}' must enable export_merged_model when inference is executed."
            )
        if method == "rm" and bool(resolved_output.get("export_merged_model", False)):
            raise ValueError(
                f"Reward model candidate '{candidate_id}' must set output.export_merged_model=false."
            )

        resolved_tuning = merge_mappings(
            require_mapping(defaults_raw["alignment"].get("tuning"), "defaults.alignment.tuning"),
            optional_mapping(candidate.get("tuning"), f"alignment.candidates[{candidate_id}].tuning"),
        )
        if not str(resolved_tuning.get("finetuning_type", "")).strip():
            raise ValueError(
                f"Aligned candidate '{candidate_id}' resolved tuning.finetuning_type must not be empty."
            )

    if config.execute_inference:
        inference_benchmarks = get_inference_benchmark_index(config.raw)
        if not inference_benchmarks:
            raise ValueError("inference.benchmarks must be a non-empty list when inference is enabled.")
        for benchmark_id, benchmark in inference_benchmarks.items():
            validate_inference_benchmark(
                config=config,
                benchmark_id=benchmark_id,
                benchmark=benchmark,
                candidate_index=candidate_index,
                dataset_index=dataset_index,
                defaults_raw=defaults_raw,
            )

    if config.execute_evaluation:
        evaluation_benchmarks = get_evaluation_benchmark_index(config.raw)
        if not evaluation_benchmarks:
            raise ValueError("evaluation.benchmarks must be a non-empty list when evaluation is enabled.")
        inference_benchmarks = (
            get_inference_benchmark_index(config.raw) if config.execute_inference else {}
        )
        for benchmark_id, benchmark in evaluation_benchmarks.items():
            validate_evaluation_benchmark(
                config=config,
                benchmark_id=benchmark_id,
                benchmark=benchmark,
                candidate_index=candidate_index,
                dataset_index=dataset_index,
                metric_sets=metric_sets,
                judge_models=judge_models,
                defaults_raw=defaults_raw,
                inference_benchmarks=inference_benchmarks,
            )


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None or not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return value


def optional_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping when provided.")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if value is None or not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return value


def resolve_path(base_dir: Path, raw_path: Any) -> Path:
    if raw_path in (None, ""):
        raise ValueError("A required path field is missing or empty.")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def build_metric_configs(metric_items: list[Any]) -> list[MetricConfig]:
    metrics: list[MetricConfig] = []
    for item in metric_items:
        if isinstance(item, str):
            metrics.append(MetricConfig(name=item))
        elif isinstance(item, dict):
            metrics.append(
                MetricConfig(
                    name=str(item.get("name", "")),
                    threshold=float(item["threshold"]) if item.get("threshold") is not None else None,
                    include_reason=bool(item.get("include_reason", True)),
                    strict_mode=bool(item.get("strict_mode", False)),
                    verbose_mode=bool(item.get("verbose_mode", False)),
                )
            )
        else:
            raise ValueError("Each evaluation metric must be a string or mapping.")
    return metrics


def merge_mappings(*payloads: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        merged.update(payload)
    return merged


def get_catalog_index(raw_config: dict[str, Any], section: str, label: str) -> dict[str, dict[str, Any]]:
    raw_section = optional_mapping(raw_config.get(section), label)
    index: dict[str, dict[str, Any]] = {}
    for key, value in raw_section.items():
        if not isinstance(value, dict):
            raise ValueError(f"{label}.{key} must be a mapping.")
        index[str(key)] = value
    return index


def get_candidate_index(raw_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    alignment_raw = require_mapping(raw_config.get("alignment"), "alignment")
    items = require_list(alignment_raw.get("candidates"), "alignment.candidates")
    index: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each alignment candidate must be a mapping.")
        candidate_id = str(item.get("candidate_id", "")).strip()
        if not candidate_id:
            raise ValueError("Each alignment candidate must define candidate_id.")
        if candidate_id in index:
            raise ValueError(f"Duplicate alignment candidate_id: {candidate_id}")
        index[candidate_id] = item
    return index


def get_inference_benchmark_index(raw_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    inference_raw = require_mapping(raw_config.get("inference"), "inference")
    items = require_list(inference_raw.get("benchmarks"), "inference.benchmarks")
    index: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each inference benchmark must be a mapping.")
        benchmark_id = str(item.get("benchmark_id", "")).strip()
        if not benchmark_id:
            raise ValueError("Each inference benchmark must define benchmark_id.")
        if benchmark_id in index:
            raise ValueError(f"Duplicate inference benchmark_id: {benchmark_id}")
        index[benchmark_id] = item
    return index


def get_evaluation_benchmark_index(raw_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evaluation_raw = require_mapping(raw_config.get("evaluation"), "evaluation")
    items = require_list(evaluation_raw.get("benchmarks"), "evaluation.benchmarks")
    index: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each evaluation benchmark must be a mapping.")
        benchmark_id = str(item.get("benchmark_id", "")).strip()
        if not benchmark_id:
            raise ValueError("Each evaluation benchmark must define benchmark_id.")
        if benchmark_id in index:
            raise ValueError(f"Duplicate evaluation benchmark_id: {benchmark_id}")
        index[benchmark_id] = item
    return index


def select_candidate_ids(
    candidate_index: dict[str, dict[str, Any]],
    selected_items: Any,
    label: str,
) -> list[str]:
    if selected_items is None:
        return list(candidate_index)
    items = require_list(selected_items, label)
    selected: list[str] = []
    for item in items:
        candidate_id = str(item).strip()
        if not candidate_id:
            raise ValueError(f"{label} must not contain empty candidate ids.")
        if candidate_id not in candidate_index:
            raise ValueError(f"{label} references unknown candidate '{candidate_id}'.")
        if candidate_id not in selected:
            selected.append(candidate_id)
    return selected


def get_resolved_eval_metrics(
    benchmark: dict[str, Any],
    metric_sets: dict[str, dict[str, Any]],
    benchmark_id: str,
) -> list[Any]:
    metric_set_ref = benchmark.get("metric_set_ref")
    if metric_set_ref is not None:
        metric_set_key = str(metric_set_ref).strip()
        if not metric_set_key:
            raise ValueError(f"Evaluation benchmark '{benchmark_id}' has an empty metric_set_ref.")
        metric_set_entry = require_mapping(
            metric_sets.get(metric_set_key),
            f"catalog.metric_sets.{metric_set_key}",
        )
        metrics = metric_set_entry.get("metrics")
    else:
        metrics = benchmark.get("metrics", [])
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(
            f"Evaluation benchmark '{benchmark_id}' must define a non-empty metrics list or metric_set_ref."
        )
    return metrics


def get_resolved_metric_set(
    benchmark: dict[str, Any],
    metric_sets: dict[str, dict[str, Any]],
    benchmark_id: str,
) -> dict[str, Any]:
    metric_set_ref = benchmark.get("metric_set_ref")
    if metric_set_ref is None:
        return {}
    metric_set_key = str(metric_set_ref).strip()
    if not metric_set_key:
        raise ValueError(f"Evaluation benchmark '{benchmark_id}' has an empty metric_set_ref.")
    return require_mapping(
        metric_sets.get(metric_set_key),
        f"catalog.metric_sets.{metric_set_key}",
    )


def resolve_judge_payload(
    *,
    metric_set: dict[str, Any],
    benchmark: dict[str, Any],
    benchmark_id: str,
    judge_models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metric_set_ref = metric_set.get("judge_model_ref")
    benchmark_ref = benchmark.get("judge_model_ref")
    judge_ref = benchmark_ref if benchmark_ref is not None else metric_set_ref

    referenced_payload: dict[str, Any] = {}
    if judge_ref is not None:
        judge_key = str(judge_ref).strip()
        if not judge_key:
            raise ValueError(f"Evaluation benchmark '{benchmark_id}' has an empty judge_model_ref.")
        referenced_payload = require_mapping(
            judge_models.get(judge_key),
            f"catalog.judge_models.{judge_key}",
        )

    return merge_mappings(
        referenced_payload,
        optional_mapping(
            benchmark.get("judge_model"),
            f"evaluation.benchmarks[{benchmark_id}].judge_model",
        ),
    )


def validate_shared_runtime(runtime_raw: dict[str, Any]) -> None:
    target = str(runtime_raw.get("target", "")).strip().lower()
    if target not in {"cpu", "single_gpu", "multi_gpu"}:
        raise ValueError("runtime.target must be one of: cpu, single_gpu, multi_gpu.")
    device = str(runtime_raw.get("device", "")).strip()
    if not device:
        raise ValueError("runtime.device must not be empty.")
    mixed_precision = str(runtime_raw.get("mixed_precision", "")).strip().lower()
    if mixed_precision not in {"none", "bf16", "fp16"}:
        raise ValueError("runtime.mixed_precision must be one of: none, bf16, fp16.")
    cpu_threads = runtime_raw.get("cpu_threads", 0)
    if not isinstance(cpu_threads, int) or cpu_threads < 0:
        raise ValueError("runtime.cpu_threads must be an integer >= 0.")
    cuda_visible_devices = runtime_raw.get("cuda_visible_devices", [])
    if not isinstance(cuda_visible_devices, list):
        raise ValueError("runtime.cuda_visible_devices must be a list.")


def validate_alignment_dataset_entry(
    base_dir: Path,
    dataset_ref: str,
    dataset_entry: dict[str, Any],
    method: str,
) -> None:
    stage = str(dataset_entry.get("stage", "")).strip().lower()
    if stage != "alignment":
        raise ValueError(f"catalog.datasets.{dataset_ref}.stage must be 'alignment'.")
    path = resolve_path(base_dir, dataset_entry.get("path"))
    if not path.exists():
        raise ValueError(f"Alignment dataset file does not exist: {path}")
    if not str(dataset_entry.get("format", "")).strip():
        raise ValueError(f"catalog.datasets.{dataset_ref}.format must not be empty.")
    expected_schema = str(get_method_spec(method)["dataset_schema"])
    schema = str(dataset_entry.get("schema", "")).strip()
    if schema != expected_schema:
        raise ValueError(
            f"catalog.datasets.{dataset_ref}.schema must be '{expected_schema}' for method '{method}'."
        )
    fields = require_mapping(dataset_entry.get("fields"), f"catalog.datasets.{dataset_ref}.fields")
    if not str(fields.get("prompt", "")).strip():
        raise ValueError(f"catalog.datasets.{dataset_ref}.fields.prompt must not be empty.")
    if schema == "prompt_response" and not str(fields.get("response", "")).strip():
        raise ValueError(
            f"catalog.datasets.{dataset_ref}.fields.response must not be empty for prompt_response datasets."
        )
    if schema == "prompt_response_label":
        if not str(fields.get("response", "")).strip() or not str(fields.get("label", "")).strip():
            raise ValueError(
                f"catalog.datasets.{dataset_ref}.fields.response and fields.label must not be empty "
                "for prompt_response_label datasets."
            )
    if schema == "prompt_chosen_rejected":
        if not str(fields.get("chosen", "")).strip() or not str(fields.get("rejected", "")).strip():
            raise ValueError(
                f"catalog.datasets.{dataset_ref}.fields.chosen and fields.rejected must not be empty "
                "for prompt_chosen_rejected datasets."
            )


def validate_alignment_training_payload(training_raw: dict[str, Any], label: str) -> None:
    if int(training_raw.get("cutoff_len", 0)) <= 0:
        raise ValueError(f"{label}.training.cutoff_len must be > 0.")
    if int(training_raw.get("per_device_train_batch_size", 0)) <= 0:
        raise ValueError(f"{label}.training.per_device_train_batch_size must be > 0.")
    if int(training_raw.get("gradient_accumulation_steps", 0)) <= 0:
        raise ValueError(f"{label}.training.gradient_accumulation_steps must be > 0.")
    if float(training_raw.get("learning_rate", 0)) <= 0:
        raise ValueError(f"{label}.training.learning_rate must be > 0.")
    if float(training_raw.get("num_train_epochs", 0)) <= 0:
        raise ValueError(f"{label}.training.num_train_epochs must be > 0.")


def validate_inference_benchmark(
    *,
    config: MasterConfig,
    benchmark_id: str,
    benchmark: dict[str, Any],
    candidate_index: dict[str, dict[str, Any]],
    dataset_index: dict[str, dict[str, Any]],
    defaults_raw: dict[str, Any],
) -> None:
    dataset_ref = str(benchmark.get("dataset_ref", "")).strip()
    if not dataset_ref:
        raise ValueError(f"Inference benchmark '{benchmark_id}' must define dataset_ref.")
    dataset_entry = require_mapping(
        dataset_index.get(dataset_ref),
        f"catalog.datasets.{dataset_ref}",
    )
    stage = str(dataset_entry.get("stage", "")).strip().lower()
    if stage != "inference_eval":
        raise ValueError(
            f"catalog.datasets.{dataset_ref}.stage must be 'inference_eval' for inference benchmarks."
        )
    dataset_path = resolve_path(config.path.parent, dataset_entry.get("path"))
    if not dataset_path.exists():
        raise ValueError(f"Inference dataset file does not exist: {dataset_path}")
    fields = require_mapping(dataset_entry.get("fields"), f"catalog.datasets.{dataset_ref}.fields")
    if not str(fields.get("id", "")).strip():
        raise ValueError(f"catalog.datasets.{dataset_ref}.fields.id must not be empty.")
    if not str(fields.get("input", "")).strip():
        raise ValueError(f"catalog.datasets.{dataset_ref}.fields.input must not be empty.")

    candidate_ids = select_candidate_ids(
        candidate_index,
        benchmark.get("candidate_refs"),
        f"inference.benchmarks[{benchmark_id}].candidate_refs",
    )
    for candidate_id in candidate_ids:
        candidate = candidate_index[candidate_id]
        if str(candidate.get("method", "")).strip().lower() == "rm":
            raise ValueError(
                f"Inference benchmark '{benchmark_id}' cannot select reward model candidate "
                f"'{candidate_id}'. Use it through PPO method_config.ppo.reward_model_candidate_ref instead."
            )

    defaults_inference = require_mapping(defaults_raw.get("inference"), "defaults.inference")
    model_payload = merge_mappings(
        require_mapping(defaults_inference.get("model"), "defaults.inference.model"),
        optional_mapping(benchmark.get("model"), f"inference.benchmarks[{benchmark_id}].model"),
    )
    if not str(model_payload.get("backend", "")).strip():
        raise ValueError(f"Inference benchmark '{benchmark_id}' resolved model.backend must not be empty.")

    generation_payload = merge_mappings(
        require_mapping(defaults_inference.get("generation"), "defaults.inference.generation"),
        optional_mapping(
            benchmark.get("generation"),
            f"inference.benchmarks[{benchmark_id}].generation",
        ),
    )
    if int(generation_payload.get("max_new_tokens", 0)) <= 0:
        raise ValueError(
            f"Inference benchmark '{benchmark_id}' resolved generation.max_new_tokens must be > 0."
        )
    top_p = float(generation_payload.get("top_p", 0))
    if top_p <= 0 or top_p > 1:
        raise ValueError(
            f"Inference benchmark '{benchmark_id}' resolved generation.top_p must be in (0, 1]."
        )

    execution_payload = merge_mappings(
        require_mapping(defaults_inference.get("execution"), "defaults.inference.execution"),
        optional_mapping(
            benchmark.get("execution"),
            f"inference.benchmarks[{benchmark_id}].execution",
        ),
    )
    if int(execution_payload.get("batch_size", 0)) <= 0:
        raise ValueError(
            f"Inference benchmark '{benchmark_id}' resolved execution.batch_size must be > 0."
        )
    if int(execution_payload.get("max_concurrency", 0)) <= 0:
        raise ValueError(
            f"Inference benchmark '{benchmark_id}' resolved execution.max_concurrency must be > 0."
        )


def validate_evaluation_benchmark(
    *,
    config: MasterConfig,
    benchmark_id: str,
    benchmark: dict[str, Any],
    candidate_index: dict[str, dict[str, Any]],
    dataset_index: dict[str, dict[str, Any]],
    metric_sets: dict[str, dict[str, Any]],
    judge_models: dict[str, dict[str, Any]],
    defaults_raw: dict[str, Any],
    inference_benchmarks: dict[str, dict[str, Any]],
) -> None:
    dataset_ref = str(benchmark.get("dataset_ref", "")).strip()
    if not dataset_ref:
        raise ValueError(f"Evaluation benchmark '{benchmark_id}' must define dataset_ref.")
    dataset_entry = require_mapping(
        dataset_index.get(dataset_ref),
        f"catalog.datasets.{dataset_ref}",
    )
    stage = str(dataset_entry.get("stage", "")).strip().lower()
    if stage != "inference_eval":
        raise ValueError(
            f"catalog.datasets.{dataset_ref}.stage must be 'inference_eval' for evaluation benchmarks."
        )
    dataset_path = resolve_path(config.path.parent, dataset_entry.get("path"))
    if not dataset_path.exists():
        raise ValueError(f"Evaluation dataset file does not exist: {dataset_path}")
    fields = require_mapping(dataset_entry.get("fields"), f"catalog.datasets.{dataset_ref}.fields")
    if not str(fields.get("id", "")).strip():
        raise ValueError(f"catalog.datasets.{dataset_ref}.fields.id must not be empty.")
    if not str(fields.get("input", "")).strip():
        raise ValueError(f"catalog.datasets.{dataset_ref}.fields.input must not be empty.")

    metric_set = get_resolved_metric_set(benchmark, metric_sets, benchmark_id)
    metrics_raw = get_resolved_eval_metrics(benchmark, metric_sets, benchmark_id)
    metrics = build_metric_configs(metrics_raw)

    defaults_eval = require_mapping(defaults_raw.get("evaluation"), "defaults.evaluation")
    judge_payload = resolve_judge_payload(
        metric_set=metric_set,
        benchmark=benchmark,
        benchmark_id=benchmark_id,
        judge_models=judge_models,
    )
    if metrics_require_judge(metrics):
        if not str(judge_payload.get("model", judge_payload.get("model_name_or_path", ""))).strip():
            raise ValueError(
                f"Evaluation benchmark '{benchmark_id}' requires a configured judge model."
            )
        backend = str(judge_payload.get("backend", "transformers_local")).strip().lower()
        if backend not in {"transformers_local", "huggingface", "hf"}:
            raise ValueError(
                f"Evaluation benchmark '{benchmark_id}' resolved judge_model.backend must be "
                "one of: transformers_local, huggingface, hf."
            )

    execution_payload = merge_mappings(
        require_mapping(defaults_eval.get("execution"), "defaults.evaluation.execution"),
        optional_mapping(
            benchmark.get("execution"),
            f"evaluation.benchmarks[{benchmark_id}].execution",
        ),
    )
    if int(execution_payload.get("batch_size", 0)) < 0:
        raise ValueError(
            f"Evaluation benchmark '{benchmark_id}' resolved execution.batch_size must be >= 0."
        )
    if int(execution_payload.get("max_concurrency", 0)) <= 0:
        raise ValueError(
            f"Evaluation benchmark '{benchmark_id}' resolved execution.max_concurrency must be > 0."
        )

    if config.execute_inference:
        source_id = str(benchmark.get("source_inference_benchmark", "")).strip()
        if not source_id:
            raise ValueError(
                f"Evaluation benchmark '{benchmark_id}' must define source_inference_benchmark when inference is executed."
            )
        if source_id not in inference_benchmarks:
            raise ValueError(
                f"Evaluation benchmark '{benchmark_id}' references unknown source_inference_benchmark '{source_id}'."
            )
        source_benchmark = inference_benchmarks[source_id]
        source_candidate_ids = select_candidate_ids(
            candidate_index,
            source_benchmark.get("candidate_refs"),
            f"inference.benchmarks[{source_id}].candidate_refs",
        )
        if benchmark.get("candidate_refs") is not None:
            eval_candidate_ids = select_candidate_ids(
                candidate_index,
                benchmark.get("candidate_refs"),
                f"evaluation.benchmarks[{benchmark_id}].candidate_refs",
            )
            missing = sorted(set(eval_candidate_ids) - set(source_candidate_ids))
            if missing:
                raise ValueError(
                    f"Evaluation benchmark '{benchmark_id}' selects candidates not present in "
                    f"source inference benchmark '{source_id}': {missing}"
                )
    else:
        outputs = require_list(
            benchmark.get("outputs"),
            f"evaluation.benchmarks[{benchmark_id}].outputs",
        )
        if not outputs:
            raise ValueError(
                f"Evaluation benchmark '{benchmark_id}' must define at least one output entry."
            )
        for index, item in enumerate(outputs):
            if not isinstance(item, dict):
                raise ValueError(
                    f"evaluation.benchmarks[{benchmark_id}].outputs[{index}] must be a mapping."
                )
            candidate_ref = str(item.get("candidate_ref", "")).strip()
            if not candidate_ref:
                raise ValueError(
                    f"evaluation.benchmarks[{benchmark_id}].outputs[{index}].candidate_ref must not be empty."
                )
            if candidate_ref not in candidate_index:
                raise ValueError(
                    f"evaluation.benchmarks[{benchmark_id}].outputs[{index}] references unknown candidate "
                    f"'{candidate_ref}'."
                )
            output_path = resolve_path(config.path.parent, item.get("path"))
            if not output_path.exists():
                raise ValueError(f"Evaluation model outputs file does not exist: {output_path}")
