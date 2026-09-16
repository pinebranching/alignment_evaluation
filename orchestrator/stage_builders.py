from __future__ import annotations

from pathlib import Path
from typing import Any

from alignment.method_registry import get_method_spec

from .config import (
    get_candidate_index,
    get_catalog_index,
    get_evaluation_benchmark_index,
    get_inference_benchmark_index,
    get_resolved_eval_metrics,
    get_resolved_metric_set,
    merge_mappings,
    optional_mapping,
    require_mapping,
    resolve_judge_payload,
    resolve_path,
)
from .schemas import MasterConfig, StagePlan


def build_alignment_config(
    config: MasterConfig,
    plan: StagePlan,
    branch_root: Path,
    alignment_results_by_candidate: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not plan.candidate_id:
        raise ValueError("Alignment stage plan requires candidate_id.")

    defaults_raw = require_mapping(config.raw.get("defaults"), "defaults")
    alignment_defaults = require_mapping(defaults_raw.get("alignment"), "defaults.alignment")
    catalog_raw = require_mapping(config.raw.get("catalog"), "catalog")
    model_index = get_catalog_index(catalog_raw, "models", "catalog.models")
    dataset_index = get_catalog_index(catalog_raw, "datasets", "catalog.datasets")
    training_profiles = optional_mapping(
        catalog_raw.get("training_profiles"),
        "catalog.training_profiles",
    )
    candidate = get_candidate_index(config.raw)[plan.candidate_id]

    if str(candidate.get("kind", "")).strip().lower() != "aligned":
        raise ValueError(f"Candidate '{plan.candidate_id}' is not alignable.")

    base_model_ref = str(candidate.get("base_model_ref", "")).strip()
    dataset_ref = str(candidate.get("dataset_ref", "")).strip()
    base_model = require_mapping(model_index.get(base_model_ref), f"catalog.models.{base_model_ref}")
    dataset_entry = require_mapping(dataset_index.get(dataset_ref), f"catalog.datasets.{dataset_ref}")
    dataset_fields = require_mapping(dataset_entry.get("fields"), f"catalog.datasets.{dataset_ref}.fields")

    profile_payload: dict[str, Any] = {}
    profile_ref = candidate.get("profile_ref")
    if profile_ref is not None:
        profile_key = str(profile_ref).strip()
        profile_payload = require_mapping(
            training_profiles.get(profile_key),
            f"catalog.training_profiles.{profile_key}",
        )

    method = str(candidate.get("method", "")).strip().lower()
    method_spec = get_method_spec(method)
    output_payload = merge_mappings(
        require_mapping(alignment_defaults.get("output"), "defaults.alignment.output"),
        optional_mapping(candidate.get("output"), f"alignment.candidates[{plan.candidate_id}].output"),
    )

    method_config_payload = build_alignment_method_config(
        candidate=candidate,
        candidate_id=plan.candidate_id,
        method=method,
        alignment_results_by_candidate=alignment_results_by_candidate or {},
    )

    return {
        "alignment": {
            "backend": alignment_defaults.get("backend", "llama_factory"),
            "method": method,
            "run_name": plan.run_name,
            "model": {
                "base_model_name_or_path": base_model.get("model_name_or_path"),
                "tokenizer_name_or_path": base_model.get("tokenizer_name_or_path"),
                "template": base_model.get("template"),
                "trust_remote_code": base_model.get("trust_remote_code", False),
                "disable_gradient_checkpointing": base_model.get(
                    "disable_gradient_checkpointing"
                ),
            },
            "tuning": merge_mappings(
                require_mapping(alignment_defaults.get("tuning"), "defaults.alignment.tuning"),
                optional_mapping(candidate.get("tuning"), f"alignment.candidates[{plan.candidate_id}].tuning"),
            ),
            "dataset": {
                "path": str(resolve_path(config.path.parent, dataset_entry.get("path"))),
                "format": dataset_entry.get("format", "jsonl"),
                "schema": str(method_spec["dataset_schema"]),
                "prompt_field": dataset_fields.get("prompt", "prompt"),
                "response_field": dataset_fields.get("response"),
                "chosen_field": dataset_fields.get("chosen"),
                "rejected_field": dataset_fields.get("rejected"),
                "label_field": dataset_fields.get("label"),
                "system_field": dataset_fields.get("system"),
                "max_samples": dataset_entry.get("max_samples"),
            },
            "runtime": build_alignment_runtime(config.raw, plan.candidate_id),
            "training": merge_mappings(
                require_mapping(alignment_defaults.get("training"), "defaults.alignment.training"),
                optional_mapping(
                    profile_payload.get("training"),
                    f"catalog.training_profiles.{profile_ref}.training",
                )
                if profile_ref is not None
                else {},
                optional_mapping(
                    candidate.get("training"),
                    f"alignment.candidates[{plan.candidate_id}].training",
                ),
            ),
            "method_config": method_config_payload,
            "output": {
                **output_payload,
                "root_dir": str(branch_root / "alignment"),
            },
        }
    }


def build_alignment_method_config(
    *,
    candidate: dict[str, Any],
    candidate_id: str,
    method: str,
    alignment_results_by_candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    method_config_payload = dict(
        optional_mapping(
            candidate.get("method_config"),
            f"alignment.candidates[{candidate_id}].method_config",
        )
    )
    if method != "ppo":
        return method_config_payload

    ppo_payload = dict(
        optional_mapping(
            method_config_payload.get("ppo"),
            f"alignment.candidates[{candidate_id}].method_config.ppo",
        )
    )
    reward_model_candidate_ref = str(ppo_payload.get("reward_model_candidate_ref", "")).strip()
    if not reward_model_candidate_ref:
        return method_config_payload

    reward_result = alignment_results_by_candidate.get(reward_model_candidate_ref)
    if reward_result is None:
        raise ValueError(
            f"PPO candidate '{candidate_id}' requires reward model candidate "
            f"'{reward_model_candidate_ref}' to complete before PPO starts."
        )
    ppo_payload["reward_model_path"] = reward_result["artifacts"]["adapter_path"]
    ppo_payload["reward_model_type"] = "lora"
    method_config_payload["ppo"] = ppo_payload
    return method_config_payload


def build_inference_config(
    config: MasterConfig,
    plan: StagePlan,
    branch_root: Path,
    alignment_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if not plan.candidate_id or not plan.inference_benchmark_id:
        raise ValueError("Inference stage plan requires candidate_id and inference_benchmark_id.")

    defaults_raw = require_mapping(config.raw.get("defaults"), "defaults")
    defaults_inference = require_mapping(defaults_raw.get("inference"), "defaults.inference")
    benchmark = get_inference_benchmark_index(config.raw)[plan.inference_benchmark_id]
    dataset_payload = resolve_inference_dataset_payload(config, benchmark)
    model_payload = resolve_inference_model_payload(config, plan, benchmark, alignment_result)

    return {
        "inference": {
            "mode": defaults_inference.get("mode", "offline"),
            "dataset": dataset_payload,
            "model": model_payload,
            "generation": merge_mappings(
                require_mapping(defaults_inference.get("generation"), "defaults.inference.generation"),
                optional_mapping(
                    benchmark.get("generation"),
                    f"inference.benchmarks[{plan.inference_benchmark_id}].generation",
                ),
            ),
            "execution": merge_mappings(
                require_mapping(defaults_inference.get("execution"), "defaults.inference.execution"),
                optional_mapping(
                    benchmark.get("execution"),
                    f"inference.benchmarks[{plan.inference_benchmark_id}].execution",
                ),
            ),
            "runtime": build_stage_runtime(config.raw),
            "output": {
                "path": str(branch_root / "inference" / "model_outputs.generated.jsonl"),
                "overwrite": True,
                "save_manifest": True,
            },
        }
    }


def build_eval_config(
    config: MasterConfig,
    plan: StagePlan,
    branch_root: Path,
    inference_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if not plan.evaluation_benchmark_id:
        raise ValueError("Evaluation stage plan requires evaluation_benchmark_id.")

    defaults_raw = require_mapping(config.raw.get("defaults"), "defaults")
    defaults_eval = require_mapping(defaults_raw.get("evaluation"), "defaults.evaluation")
    evaluation_benchmark = get_evaluation_benchmark_index(config.raw)[plan.evaluation_benchmark_id]
    dataset_payload = resolve_eval_dataset_payload(config, evaluation_benchmark)
    catalog_raw = require_mapping(config.raw.get("catalog"), "catalog")
    metric_sets = get_catalog_index(catalog_raw, "metric_sets", "catalog.metric_sets")
    judge_models = get_catalog_index(catalog_raw, "judge_models", "catalog.judge_models")
    metric_set = get_resolved_metric_set(
        evaluation_benchmark,
        metric_sets,
        plan.evaluation_benchmark_id,
    )
    metrics_payload = get_resolved_eval_metrics(
        evaluation_benchmark,
        metric_sets,
        plan.evaluation_benchmark_id,
    )

    model_outputs_path = plan.external_model_outputs_path
    if inference_result is not None:
        model_outputs_path = inference_result["output_path"]

    resolved_inference_model = None
    if plan.inference_benchmark_id is not None:
        inference_benchmark = get_inference_benchmark_index(config.raw)[plan.inference_benchmark_id]
        resolved_inference_model = resolve_inference_model_payload(
            config,
            plan,
            inference_benchmark,
            alignment_result=None,
        )

    judge_payload = resolve_judge_payload(
        metric_set=metric_set,
        benchmark=evaluation_benchmark,
        benchmark_id=plan.evaluation_benchmark_id,
        judge_models=judge_models,
    )
    resolved_judge_payload: dict[str, Any] | None = None
    if judge_payload:
        resolved_judge_payload = {
            **judge_payload,
            "backend": judge_payload.get("backend", "transformers_local"),
            "model": judge_payload.get("model", judge_payload.get("model_name_or_path")),
            "revision": judge_payload.get(
                "revision",
                resolved_inference_model.get("revision") if resolved_inference_model else None,
            ),
            "trust_remote_code": judge_payload.get(
                "trust_remote_code",
                resolved_inference_model.get("trust_remote_code", False)
                if resolved_inference_model
                else False,
            ),
            "torch_dtype": judge_payload.get(
                "torch_dtype",
                resolved_inference_model.get("torch_dtype") if resolved_inference_model else None,
            ),
            "device": judge_payload.get(
                "device",
                resolved_inference_model.get("device", "auto")
                if resolved_inference_model
                else "auto",
            ),
            "temperature": judge_payload.get("temperature", 0.0),
            "generation_kwargs": judge_payload.get("generation_kwargs", {}),
        }

    return {
        "eval": {
            "mode": defaults_eval.get("mode", "offline"),
            "dataset": dataset_payload,
            "model_outputs": {
                "source": "file",
                "path": model_outputs_path,
                "id_field": "id",
                "actual_output_field": "actual_output",
            },
            "judge_model": resolved_judge_payload,
            "metrics": metrics_payload,
            "execution": merge_mappings(
                require_mapping(defaults_eval.get("execution"), "defaults.evaluation.execution"),
                optional_mapping(
                    evaluation_benchmark.get("execution"),
                    f"evaluation.benchmarks[{plan.evaluation_benchmark_id}].execution",
                ),
            ),
            "runtime": build_stage_runtime(config.raw),
            "output": {
                "dir": str(branch_root / "evaluation"),
                "save_json": True,
                "save_csv": True,
                "save_visualizations": True,
                "save_config_snapshot": True,
            },
        }
    }


def build_alignment_runtime(raw_config: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    shared_runtime = build_stage_runtime(raw_config)
    defaults_raw = require_mapping(raw_config.get("defaults"), "defaults")
    alignment_defaults = require_mapping(defaults_raw.get("alignment"), "defaults.alignment")
    candidate = get_candidate_index(raw_config)[candidate_id]
    alignment_runtime = merge_mappings(
        shared_runtime,
        optional_mapping(alignment_defaults.get("runtime"), "defaults.alignment.runtime"),
        optional_mapping(candidate.get("runtime"), f"alignment.candidates[{candidate_id}].runtime"),
    )
    return {
        "target": alignment_runtime.get("target", "cpu"),
        "device": alignment_runtime.get("device", "cpu"),
        "mixed_precision": alignment_runtime.get("mixed_precision", "none"),
        "cpu_threads": alignment_runtime.get("cpu_threads", 0),
        "cuda_visible_devices": alignment_runtime.get("cuda_visible_devices", []),
        "command_mode": alignment_runtime.get("command_mode", "module"),
        "python_executable": alignment_runtime.get("python_executable"),
        "force_torchrun": alignment_runtime.get("force_torchrun", False),
        "nproc_per_node": alignment_runtime.get("nproc_per_node"),
        "extra_env": alignment_runtime.get("extra_env", {}),
        "heartbeat_interval_seconds": alignment_runtime.get("heartbeat_interval_seconds", 30),
    }


def build_stage_runtime(raw_config: dict[str, Any]) -> dict[str, Any]:
    defaults_raw = require_mapping(raw_config.get("defaults"), "defaults")
    shared_runtime = require_mapping(defaults_raw.get("runtime"), "defaults.runtime")
    return {
        "target": shared_runtime.get("target", "cpu"),
        "device": shared_runtime.get("device", "cpu"),
        "mixed_precision": shared_runtime.get("mixed_precision", "none"),
        "cpu_threads": shared_runtime.get("cpu_threads", 0),
        "cuda_visible_devices": shared_runtime.get("cuda_visible_devices", []),
    }


def resolve_inference_dataset_payload(
    config: MasterConfig,
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    catalog_raw = require_mapping(config.raw.get("catalog"), "catalog")
    dataset_index = get_catalog_index(catalog_raw, "datasets", "catalog.datasets")
    dataset_ref = str(benchmark.get("dataset_ref", "")).strip()
    dataset_entry = require_mapping(dataset_index.get(dataset_ref), f"catalog.datasets.{dataset_ref}")
    fields = require_mapping(dataset_entry.get("fields"), f"catalog.datasets.{dataset_ref}.fields")
    return {
        "path": str(resolve_path(config.path.parent, dataset_entry.get("path"))),
        "format": dataset_entry.get("format", "jsonl"),
        "id_field": fields.get("id", "id"),
        "input_field": fields.get("input", "input"),
        "max_samples": dataset_entry.get("max_samples"),
        "expected_output_field": fields.get("expected_output"),
        "context_field": fields.get("context"),
        "retrieval_context_field": fields.get("retrieval_context"),
        "metadata_field": fields.get("metadata"),
    }


def resolve_inference_model_payload(
    config: MasterConfig,
    plan: StagePlan,
    benchmark: dict[str, Any],
    alignment_result: dict[str, Any] | None,
    output_model_path: str | None = None,
) -> dict[str, Any]:
    if not plan.candidate_id:
        raise ValueError("Inference model resolution requires candidate_id.")

    defaults_raw = require_mapping(config.raw.get("defaults"), "defaults")
    defaults_inference = require_mapping(defaults_raw.get("inference"), "defaults.inference")
    model_index = get_catalog_index(
        require_mapping(config.raw.get("catalog"), "catalog"),
        "models",
        "catalog.models",
    )
    candidate = get_candidate_index(config.raw)[plan.candidate_id]
    benchmark_model_override = optional_mapping(
        benchmark.get("model"),
        f"inference.benchmarks[{plan.inference_benchmark_id}].model",
    )

    if plan.candidate_kind == "base":
        model_ref = str(candidate.get("model_ref", "")).strip()
        model_entry = require_mapping(model_index.get(model_ref), f"catalog.models.{model_ref}")
        base_payload = {
            "model_name_or_path": model_entry.get("model_name_or_path"),
            "tokenizer_name_or_path": model_entry.get("tokenizer_name_or_path"),
            "revision": model_entry.get("revision"),
            "trust_remote_code": model_entry.get("trust_remote_code", False),
        }
        return merge_mappings(
            base_payload,
            require_mapping(defaults_inference.get("model"), "defaults.inference.model"),
            benchmark_model_override,
        )

    base_model_ref = str(candidate.get("base_model_ref", "")).strip()
    model_entry = require_mapping(model_index.get(base_model_ref), f"catalog.models.{base_model_ref}")
    exported_model_path = (
        alignment_result.get("artifacts", {}).get("exported_model_path")
        if alignment_result is not None else None
    )
    adapter_path = (
        alignment_result.get("artifacts", {}).get("adapter_path")
        if alignment_result is not None and not exported_model_path and not output_model_path else None
    )
    resolved_model_name = output_model_path or exported_model_path or model_entry.get("model_name_or_path")
    resolved_tokenizer_name = resolved_model_name
    if adapter_path:
        resolved_tokenizer_name = (
            model_entry.get("tokenizer_name_or_path") or model_entry.get("model_name_or_path")
        )
    base_payload = {
        "model_name_or_path": resolved_model_name,
        "adapter_name_or_path": adapter_path,
        "tokenizer_name_or_path": resolved_tokenizer_name,
        "revision": model_entry.get("revision"),
        "trust_remote_code": model_entry.get("trust_remote_code", False),
    }
    return merge_mappings(
        base_payload,
        require_mapping(defaults_inference.get("model"), "defaults.inference.model"),
        benchmark_model_override,
    )


def resolve_eval_dataset_payload(
    config: MasterConfig,
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    catalog_raw = require_mapping(config.raw.get("catalog"), "catalog")
    dataset_index = get_catalog_index(catalog_raw, "datasets", "catalog.datasets")
    dataset_ref = str(benchmark.get("dataset_ref", "")).strip()
    dataset_entry = require_mapping(dataset_index.get(dataset_ref), f"catalog.datasets.{dataset_ref}")
    fields = require_mapping(dataset_entry.get("fields"), f"catalog.datasets.{dataset_ref}.fields")
    return {
        "path": str(resolve_path(config.path.parent, dataset_entry.get("path"))),
        "format": dataset_entry.get("format", "jsonl"),
        "input_field": fields.get("input", "input"),
        "expected_output_field": fields.get("expected_output"),
        "id_field": fields.get("id", "id"),
        "max_samples": dataset_entry.get("max_samples"),
        "context_field": fields.get("context"),
        "retrieval_context_field": fields.get("retrieval_context"),
        "metadata_field": fields.get("metadata"),
    }
