from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    get_candidate_index,
    get_catalog_index,
    get_evaluation_benchmark_index,
    get_inference_benchmark_index,
    get_resolved_metric_set,
    load_master_config,
    optional_mapping,
    require_list,
    require_mapping,
    resolve_judge_payload,
    resolve_path,
    select_candidate_ids,
    validate_master_config,
)
from .executor import run_alignment_stage, run_evaluation_stage, run_inference_stage
from .run_context import (
    initialize_pipeline_state,
    persist_state,
    prepare_run_dir,
    utc_now,
    write_jsonl,
    write_yaml,
)
from .schemas import MasterConfig, PipelineState, StagePlan
from .stage_builders import (
    build_alignment_config,
    build_eval_config,
    build_inference_config,
    resolve_inference_model_payload,
)
from .csv_aggregator import export_orchestrator_csv_aggregates
from .visualization_exporter import export_orchestrator_visualizations


MAX_STAGE_ATTEMPTS = 2
BRANCH_SUMMARY_FILENAME = "branch_stage_summary.jsonl"
PREFETCH_ALLOW_PATTERNS = [
    "*.json",
    "*.model",
    "*.txt",
    "*.py",
    "*.safetensors",
    "*.bin",
    "*.tiktoken",
    "*.yaml",
    "*.yml",
]


def expand_stage_plans(config: MasterConfig) -> list[StagePlan]:
    candidate_index = get_candidate_index(config.raw)

    if config.execute_inference:
        inference_benchmarks = get_inference_benchmark_index(config.raw)
        if config.execute_evaluation:
            evaluation_benchmarks = get_evaluation_benchmark_index(config.raw)
            plans: list[StagePlan] = []
            for evaluation_benchmark_id, evaluation_benchmark in evaluation_benchmarks.items():
                source_inference_benchmark_id = str(
                    evaluation_benchmark.get("source_inference_benchmark", "")
                ).strip()
                inference_benchmark = inference_benchmarks[source_inference_benchmark_id]
                candidate_ids = select_candidate_ids(
                    candidate_index,
                    inference_benchmark.get("candidate_refs"),
                    f"inference.benchmarks[{source_inference_benchmark_id}].candidate_refs",
                )
                if evaluation_benchmark.get("candidate_refs") is not None:
                    candidate_ids = select_candidate_ids(
                        candidate_index,
                        evaluation_benchmark.get("candidate_refs"),
                        f"evaluation.benchmarks[{evaluation_benchmark_id}].candidate_refs",
                    )

                for candidate_id in candidate_ids:
                    candidate = candidate_index[candidate_id]
                    plans.append(
                        StagePlan(
                            branch_id=build_branch_id(
                                config,
                                candidate_id=candidate_id,
                                inference_benchmark_id=source_inference_benchmark_id,
                                evaluation_benchmark_id=evaluation_benchmark_id,
                            ),
                            candidate_id=candidate_id,
                            candidate_kind=str(candidate.get("kind", "")).strip().lower(),
                            method=_candidate_method(candidate),
                            run_name=build_run_name(
                                config.run_name,
                                candidate_id,
                                source_inference_benchmark_id,
                                evaluation_benchmark_id,
                            ),
                            inference_benchmark_id=source_inference_benchmark_id,
                            evaluation_benchmark_id=evaluation_benchmark_id,
                        )
                    )
            return deduplicate_stage_plan_branch_ids(add_reward_model_dependency_plans(config, plans))

        plans: list[StagePlan] = []
        for inference_benchmark_id, inference_benchmark in inference_benchmarks.items():
            candidate_ids = select_candidate_ids(
                candidate_index,
                inference_benchmark.get("candidate_refs"),
                f"inference.benchmarks[{inference_benchmark_id}].candidate_refs",
            )
            for candidate_id in candidate_ids:
                candidate = candidate_index[candidate_id]
                plans.append(
                    StagePlan(
                        branch_id=build_branch_id(
                            config,
                            candidate_id=candidate_id,
                            inference_benchmark_id=inference_benchmark_id,
                            evaluation_benchmark_id=None,
                        ),
                        candidate_id=candidate_id,
                        candidate_kind=str(candidate.get("kind", "")).strip().lower(),
                        method=_candidate_method(candidate),
                        run_name=build_run_name(
                            config.run_name,
                            candidate_id,
                            inference_benchmark_id,
                            None,
                        ),
                        inference_benchmark_id=inference_benchmark_id,
                        evaluation_benchmark_id=None,
                    )
                )
        return deduplicate_stage_plan_branch_ids(add_reward_model_dependency_plans(config, plans))

    if config.execute_alignment:
        plans: list[StagePlan] = []
        for candidate_id, candidate in candidate_index.items():
            if str(candidate.get("kind", "")).strip().lower() != "aligned":
                continue
            plans.append(
                StagePlan(
                    branch_id=build_branch_id(
                        config,
                        candidate_id=candidate_id,
                        inference_benchmark_id=None,
                        evaluation_benchmark_id=None,
                    ),
                    candidate_id=candidate_id,
                    candidate_kind="aligned",
                    method=_candidate_method(candidate),
                    run_name=build_run_name(config.run_name, candidate_id, None, None),
                )
            )
        return deduplicate_stage_plan_branch_ids(add_reward_model_dependency_plans(config, plans))

    evaluation_benchmarks = get_evaluation_benchmark_index(config.raw)
    plans: list[StagePlan] = []
    for evaluation_benchmark_id, evaluation_benchmark in evaluation_benchmarks.items():
        outputs = require_list(
            evaluation_benchmark.get("outputs"),
            f"evaluation.benchmarks[{evaluation_benchmark_id}].outputs",
        )
        for index, item in enumerate(outputs):
            candidate_id = str(item.get("candidate_ref", "")).strip()
            candidate = candidate_index[candidate_id]
            output_path = str(resolve_path(config.path.parent, item.get("path")))
            plans.append(
                StagePlan(
                    branch_id=build_branch_id(
                        config,
                        candidate_id=candidate_id,
                        inference_benchmark_id=None,
                        evaluation_benchmark_id=evaluation_benchmark_id,
                        suffix=str(index),
                    ),
                    candidate_id=candidate_id,
                    candidate_kind=str(candidate.get("kind", "")).strip().lower(),
                    method=_candidate_method(candidate),
                    run_name=build_run_name(
                        config.run_name,
                        candidate_id,
                        None,
                        evaluation_benchmark_id,
                    ),
                    evaluation_benchmark_id=evaluation_benchmark_id,
                    external_model_outputs_path=output_path,
                )
            )
    return deduplicate_stage_plan_branch_ids(plans)


def add_reward_model_dependency_plans(config: MasterConfig, plans: list[StagePlan]) -> list[StagePlan]:
    candidate_index = get_candidate_index(config.raw)
    expanded: list[StagePlan] = []
    emitted_alignment_candidates: set[str] = set()
    skipped_original_dependencies: set[str] = set()

    for plan in plans:
        if plan.candidate_id in skipped_original_dependencies:
            continue

        if plan.method == "ppo" and plan.candidate_id:
            reward_candidate_id = get_ppo_reward_model_candidate_ref(
                candidate_index[plan.candidate_id]
            )
            if reward_candidate_id and reward_candidate_id not in emitted_alignment_candidates:
                reward_candidate = candidate_index[reward_candidate_id]
                expanded.append(
                    StagePlan(
                        branch_id=build_branch_id(
                            config,
                            candidate_id=reward_candidate_id,
                            inference_benchmark_id=None,
                            evaluation_benchmark_id=None,
                        ),
                        candidate_id=reward_candidate_id,
                        candidate_kind=str(reward_candidate.get("kind", "")).strip().lower(),
                        method=_candidate_method(reward_candidate),
                        run_name=build_run_name(config.run_name, reward_candidate_id, None, None),
                    )
                )
                emitted_alignment_candidates.add(reward_candidate_id)
                skipped_original_dependencies.add(reward_candidate_id)

        expanded.append(plan)
        if plan.candidate_kind == "aligned" and plan.candidate_id:
            emitted_alignment_candidates.add(plan.candidate_id)

    return expanded


def deduplicate_stage_plan_branch_ids(plans: list[StagePlan]) -> list[StagePlan]:
    """Append numeric suffixes when a naming template produces duplicate branch ids."""

    seen: set[str] = set()
    next_suffix: dict[str, int] = {}
    for plan in plans:
        original_branch_id = plan.branch_id
        if original_branch_id not in seen:
            seen.add(original_branch_id)
            next_suffix.setdefault(original_branch_id, 2)
            continue

        suffix = next_suffix.get(original_branch_id, 2)
        while True:
            candidate_branch_id = f"{original_branch_id}--{suffix}"
            suffix += 1
            if candidate_branch_id not in seen:
                plan.branch_id = candidate_branch_id
                seen.add(candidate_branch_id)
                next_suffix[original_branch_id] = suffix
                break

    return plans


def get_ppo_reward_model_candidate_ref(candidate: dict[str, Any]) -> str | None:
    method_config = optional_mapping(
        candidate.get("method_config"),
        "alignment.candidates[].method_config",
    )
    ppo_config = optional_mapping(
        method_config.get("ppo"),
        "alignment.candidates[].method_config.ppo",
    )
    reward_candidate_id = str(ppo_config.get("reward_model_candidate_ref", "")).strip()
    return reward_candidate_id or None


def build_branch_id(
    config: MasterConfig,
    *,
    candidate_id: str | None,
    inference_benchmark_id: str | None,
    evaluation_benchmark_id: str | None,
    suffix: str | None = None,
) -> str:
    orchestrator_raw = config.raw["orchestrator"]
    naming_raw = optional_mapping(orchestrator_raw.get("naming"), "orchestrator.naming")
    template = str(naming_raw.get("branch_id_template", "{candidate_id}/{benchmark_id}")).strip()
    if not template:
        template = "{candidate_id}/{benchmark_id}"
    benchmark_id = evaluation_benchmark_id or inference_benchmark_id or "direct"
    rendered = template.format(
        candidate_id=candidate_id or "direct",
        benchmark_id=benchmark_id,
        inference_benchmark_id=inference_benchmark_id or "",
        evaluation_benchmark_id=evaluation_benchmark_id or "",
    )
    sanitized = sanitize_branch_component(rendered)
    if suffix:
        sanitized = f"{sanitized}--{sanitize_branch_component(suffix)}"
    return sanitized


def build_run_name(
    base_run_name: str,
    candidate_id: str | None,
    inference_benchmark_id: str | None,
    evaluation_benchmark_id: str | None,
) -> str:
    parts = [base_run_name]
    if candidate_id:
        parts.append(candidate_id)
    if inference_benchmark_id:
        parts.append(inference_benchmark_id)
    if evaluation_benchmark_id:
        parts.append(evaluation_benchmark_id)
    return "-".join(parts)


def sanitize_branch_component(value: str) -> str:
    normalized = []
    for character in value:
        if character.isalnum() or character in {"-", "_"}:
            normalized.append(character)
        elif character in {"/", "\\", ":"}:
            normalized.append("--")
        else:
            normalized.append("-")
    rendered = "".join(normalized)
    while "---" in rendered:
        rendered = rendered.replace("---", "--")
    return rendered.strip("-") or "branch"


def _candidate_method(candidate: dict[str, Any]) -> str | None:
    if str(candidate.get("kind", "")).strip().lower() != "aligned":
        return None
    method = str(candidate.get("method", "")).strip().lower()
    return method or None


class StageExecutionError(RuntimeError):
    def __init__(self, stage_name: str, message: str):
        super().__init__(message)
        self.stage_name = stage_name


def _normalized_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    if not rendered or rendered.lower() == "none":
        return None
    return rendered


def _append_prefetch_bundle(
    bundles: list[dict[str, Any]],
    seen: dict[tuple[str, str, str | None, bool], dict[str, Any]],
    *,
    role: str,
    label: str,
    model_name_or_path: str | None,
    tokenizer_name_or_path: str | None,
    revision: str | None,
    trust_remote_code: bool,
) -> None:
    model_reference = str(model_name_or_path or "").strip()
    if not model_reference:
        return

    tokenizer_reference = str(tokenizer_name_or_path or "").strip() or model_reference
    key = (
        model_reference,
        tokenizer_reference,
        revision,
        trust_remote_code,
    )
    bundle = seen.get(key)
    if bundle is None:
        bundle = {
            "model_name_or_path": model_reference,
            "tokenizer_name_or_path": tokenizer_reference,
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "roles": [],
        }
        seen[key] = bundle
        bundles.append(bundle)
    bundle["roles"].append({"role": role, "label": label})


def collect_huggingface_prefetch_bundles(
    config: MasterConfig,
    plans: list[StagePlan],
) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str | None, bool], dict[str, Any]] = {}
    candidate_index = get_candidate_index(config.raw)
    catalog_raw = require_mapping(config.raw.get("catalog"), "catalog")
    model_index = get_catalog_index(catalog_raw, "models", "catalog.models")

    for plan in plans:
        if plan.candidate_id:
            candidate = candidate_index[plan.candidate_id]
            if plan.candidate_kind == "base":
                model_ref = str(candidate.get("model_ref", "")).strip()
            else:
                model_ref = str(candidate.get("base_model_ref", "")).strip()
            if model_ref:
                model_entry = require_mapping(model_index.get(model_ref), f"catalog.models.{model_ref}")
                _append_prefetch_bundle(
                    bundles,
                    seen,
                    role="candidate_model",
                    label=plan.candidate_id,
                    model_name_or_path=_normalized_optional_string(
                        model_entry.get("model_name_or_path")
                    ),
                    tokenizer_name_or_path=_normalized_optional_string(
                        model_entry.get("tokenizer_name_or_path")
                    ),
                    revision=_normalized_optional_string(model_entry.get("revision")),
                    trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
                )

        if plan.inference_benchmark_id is not None:
            inference_benchmark = get_inference_benchmark_index(config.raw)[plan.inference_benchmark_id]
            inference_model_payload = resolve_inference_model_payload(
                config,
                plan,
                inference_benchmark,
                alignment_result=None,
            )
            _append_prefetch_bundle(
                bundles,
                seen,
                role="inference_model",
                label=plan.branch_id,
                model_name_or_path=_normalized_optional_string(
                    inference_model_payload.get("model_name_or_path")
                ),
                tokenizer_name_or_path=_normalized_optional_string(
                    inference_model_payload.get("tokenizer_name_or_path")
                ),
                revision=_normalized_optional_string(inference_model_payload.get("revision")),
                trust_remote_code=bool(inference_model_payload.get("trust_remote_code", False)),
            )

    if config.execute_evaluation:
        evaluation_benchmarks = get_evaluation_benchmark_index(config.raw)
        metric_sets = get_catalog_index(catalog_raw, "metric_sets", "catalog.metric_sets")
        judge_models = get_catalog_index(catalog_raw, "judge_models", "catalog.judge_models")
        evaluation_benchmark_ids = {
            plan.evaluation_benchmark_id
            for plan in plans
            if plan.evaluation_benchmark_id is not None
        }
        for benchmark_id in evaluation_benchmark_ids:
            benchmark = evaluation_benchmarks[benchmark_id]
            metric_set = get_resolved_metric_set(benchmark, metric_sets, benchmark_id)
            judge_payload = resolve_judge_payload(
                metric_set=metric_set,
                benchmark=benchmark,
                benchmark_id=benchmark_id,
                judge_models=judge_models,
            )
            if not judge_payload:
                continue
            _append_prefetch_bundle(
                bundles,
                seen,
                role="judge_model",
                label=benchmark_id,
                model_name_or_path=_normalized_optional_string(
                    judge_payload.get("model", judge_payload.get("model_name_or_path"))
                ),
                tokenizer_name_or_path=_normalized_optional_string(
                    judge_payload.get("tokenizer_name_or_path")
                ),
                revision=_normalized_optional_string(judge_payload.get("revision")),
                trust_remote_code=bool(judge_payload.get("trust_remote_code", False)),
            )

    return bundles


def _resolve_local_model_reference(base_dir: Path, reference: str) -> Path | None:
    if not reference:
        return None

    candidate = Path(reference).expanduser()
    resolved_relative = (base_dir / candidate).resolve()
    if resolved_relative.exists():
        return resolved_relative
    if candidate.is_absolute() or reference.startswith((".", "~")) or (
        len(reference) > 1 and reference[1] == ":"
    ):
        resolved = candidate if candidate.is_absolute() else (base_dir / candidate).resolve()
        if resolved.exists():
            return resolved
    return None


def _prefetch_huggingface_reference(
    *,
    base_dir: Path,
    reference: str,
    revision: str | None,
) -> dict[str, Any]:
    local_path = _resolve_local_model_reference(base_dir, reference)
    if local_path is not None:
        return {
            "status": "skipped",
            "reference": reference,
            "reason": "local_path",
            "local_path": str(local_path),
        }

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Hugging Face prefetch requires 'huggingface_hub' to be installed."
        ) from exc

    cache_path = snapshot_download(
        repo_id=reference,
        revision=revision,
        allow_patterns=PREFETCH_ALLOW_PATTERNS,
    )
    return {
        "status": "completed",
        "reference": reference,
        "revision": revision,
        "cache_path": cache_path,
    }


def prefetch_huggingface_assets(
    config: MasterConfig,
    plans: list[StagePlan],
    state: PipelineState,
) -> dict[str, Any]:
    bundles = collect_huggingface_prefetch_bundles(config, plans)
    payload = {
        "status": "running",
        "started_at_utc": utc_now(),
        "bundle_count": len(bundles),
        "bundles": [],
    }
    state.payload["prefetch"] = payload
    persist_state(state, config.save_pipeline_state)

    try:
        for bundle in bundles:
            print(
                "[orchestrator] [prefetch] Resolving "
                f"model={bundle['model_name_or_path']} tokenizer={bundle['tokenizer_name_or_path']}"
            )
            bundle_result = {
                **bundle,
                "status": "completed",
                "model_prefetch": _prefetch_huggingface_reference(
                    base_dir=config.path.parent,
                    reference=bundle["model_name_or_path"],
                    revision=bundle["revision"],
                ),
                "tokenizer_prefetch": None,
            }
            tokenizer_reference = bundle["tokenizer_name_or_path"]
            if tokenizer_reference != bundle["model_name_or_path"]:
                bundle_result["tokenizer_prefetch"] = _prefetch_huggingface_reference(
                    base_dir=config.path.parent,
                    reference=tokenizer_reference,
                    revision=bundle["revision"],
                )
            payload["bundles"].append(bundle_result)
            persist_state(state, config.save_pipeline_state)
    except Exception as exc:
        payload["status"] = "failed"
        payload["finished_at_utc"] = utc_now()
        payload["error"] = str(exc)
        persist_state(state, config.save_pipeline_state)
        raise

    payload["status"] = "completed"
    payload["finished_at_utc"] = utc_now()
    return payload


def _stage_summary_rows(state_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch in state_payload.get("branches", []):
        rows.append(
            {
                "branch_id": branch.get("branch_id"),
                "branch_status": branch.get("status"),
                "alignment": (branch.get("alignment") or {}).get("status", "pending"),
                "inference": (branch.get("inference") or {}).get("status", "pending"),
                "eval": (branch.get("evaluation") or {}).get("status", "pending"),
            }
        )
    return rows


def write_branch_stage_summary(run_dir: Path, state_payload: dict[str, Any]) -> Path:
    path = run_dir / BRANCH_SUMMARY_FILENAME
    write_jsonl(path, _stage_summary_rows(state_payload))
    return path


def _prepare_stage_payload(
    branch_payload: dict[str, Any],
    stage_name: str,
    *,
    config_path: Path | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    stage_payload: dict[str, Any] = {
        "status": "pending",
        "max_attempts": MAX_STAGE_ATTEMPTS,
        "attempts": [],
    }
    if config_path is not None:
        stage_payload["config_path"] = str(config_path)
    if reason is not None:
        stage_payload["reason"] = reason
    branch_payload[stage_name] = stage_payload
    return stage_payload


def _validate_stage_result(stage_name: str, result: dict[str, Any]) -> None:
    if stage_name == "alignment":
        status = str(result.get("status", "completed")).strip().lower()
        if status == "failed":
            raise RuntimeError("Alignment stage returned status='failed'.")
        return

    if stage_name == "inference":
        failed = int(result.get("failed", 0) or 0)
        if failed > 0:
            total_records = result.get("total_records")
            raise RuntimeError(
                f"Inference produced {failed} failed record(s)"
                f"{f' out of {total_records}' if total_records is not None else ''}."
            )


def _finalize_stage_payload(
    stage_name: str,
    stage_payload: dict[str, Any],
    result: dict[str, Any],
) -> None:
    if stage_name == "evaluation":
        stage_payload["result_summary"] = result.get("summary", result)
        return
    stage_payload["result"] = result
    if stage_name == "alignment":
        status = str(result.get("status", "completed")).strip().lower()
        if status == "skipped" and result.get("reason"):
            stage_payload["reason"] = result["reason"]


def _execute_stage_with_retry(
    *,
    branch_id: str,
    stage_name: str,
    stage_payload: dict[str, Any],
    config_path: Path,
    runner,
    state: PipelineState,
    save_pipeline_state: bool,
) -> dict[str, Any]:
    last_error: str | None = None
    last_exception: Exception | None = None

    for attempt in range(1, MAX_STAGE_ATTEMPTS + 1):
        attempt_payload = {
            "attempt": attempt,
            "status": "running",
            "started_at_utc": utc_now(),
        }
        stage_payload["status"] = "running"
        stage_payload["attempt_count"] = attempt
        stage_payload["attempts"].append(attempt_payload)
        persist_state(state, save_pipeline_state)

        try:
            result = runner(config_path)
            _validate_stage_result(stage_name, result)
            attempt_payload["status"] = str(result.get("status", "completed"))
            attempt_payload["finished_at_utc"] = utc_now()
            stage_payload["status"] = attempt_payload["status"]
            stage_payload["finished_at_utc"] = attempt_payload["finished_at_utc"]
            stage_payload.pop("last_error", None)
            _finalize_stage_payload(stage_name, stage_payload, result)
            persist_state(state, save_pipeline_state)
            return result
        except Exception as exc:
            last_exception = exc
            last_error = str(exc)
            attempt_payload["status"] = "failed"
            attempt_payload["finished_at_utc"] = utc_now()
            attempt_payload["error"] = last_error
            stage_payload["last_error"] = last_error
            stage_payload["finished_at_utc"] = attempt_payload["finished_at_utc"]
            stage_payload["status"] = "retrying" if attempt < MAX_STAGE_ATTEMPTS else "failed"
            persist_state(state, save_pipeline_state)
            if attempt < MAX_STAGE_ATTEMPTS:
                print(
                    f"[orchestrator] [{stage_name}] Attempt {attempt} failed for branch={branch_id}. "
                    "Retrying once."
                )

    raise StageExecutionError(
        stage_name,
        f"{stage_name} failed for branch={branch_id} after {MAX_STAGE_ATTEMPTS} attempts: {last_error}",
    ) from last_exception


def _mark_stage_skipped(
    branch_payload: dict[str, Any],
    stage_name: str,
    *,
    reason: str | None = None,
) -> None:
    stage_payload = _prepare_stage_payload(branch_payload, stage_name, reason=reason)
    stage_payload["status"] = "skipped"
    stage_payload["attempt_count"] = 0
    stage_payload["finished_at_utc"] = utc_now()


def _mark_downstream_stages_skipped(
    branch_payload: dict[str, Any],
    *,
    failed_stage_name: str,
    reason: str,
) -> None:
    downstream_by_stage = {
        "alignment": ["inference", "evaluation"],
        "inference": ["evaluation"],
        "evaluation": [],
    }
    for stage_name in downstream_by_stage[failed_stage_name]:
        stage_payload = branch_payload.get(stage_name) or {}
        if stage_payload.get("status") in {"completed", "failed", "skipped"}:
            continue
        _mark_stage_skipped(branch_payload, stage_name, reason=reason)


def run_orchestrator(config_path: str | Path) -> dict[str, Any]:
    print(f"[orchestrator] Loading master config: {config_path}")
    config = load_master_config(config_path)
    validate_master_config(config)

    run_id, run_dir = prepare_run_dir(config)
    print(f"[orchestrator] Run directory: {run_dir}")
    write_yaml(run_dir / "resolved_master_config.yaml", config.raw)
    state = initialize_pipeline_state(config, run_id, run_dir)
    persist_state(state, config.save_pipeline_state)

    try:
        plans = expand_stage_plans(config)
        print(f"[orchestrator] Expanded {len(plans)} branch(es): {[plan.branch_id for plan in plans]}")
        state.payload["plan_count"] = len(plans)
        state.payload["status"] = "running"
        state.payload["branch_summary_path"] = str(state.run_dir / BRANCH_SUMMARY_FILENAME)
        persist_state(state, config.save_pipeline_state)
        prefetch_huggingface_assets(config, plans, state)
        persist_state(state, config.save_pipeline_state)

        alignment_results_by_candidate: dict[str, dict[str, Any]] = {}
        for plan in plans:
            run_stage_plan(config, plan, state, alignment_results_by_candidate)
            write_branch_stage_summary(state.run_dir, state.payload)

        failed_branches = [
            branch["branch_id"]
            for branch in state.payload["branches"]
            if branch["status"] == "failed"
        ]
        state.payload["status"] = "failed" if failed_branches else "completed"
        state.payload["failed_branches"] = failed_branches
        state.payload["finished_at_utc"] = utc_now()
        aggregate_csv_paths = export_orchestrator_csv_aggregates(
            state.run_dir,
            state.payload,
            config.raw,
            config_dir=config.path.parent,
        )
        if aggregate_csv_paths:
            state.payload["aggregate_csvs"] = [str(path) for path in aggregate_csv_paths]
        visualization_paths = export_orchestrator_visualizations(
            state.run_dir,
            state.payload,
            config.raw,
            metric_summary_csv_path=state.run_dir / "all_metric_summary.csv",
        )
        if visualization_paths:
            state.payload["visualizations"] = [str(path) for path in visualization_paths]
        write_branch_stage_summary(state.run_dir, state.payload)
        persist_state(state, config.save_pipeline_state)
        print(f"[orchestrator] Finished with status={state.payload['status']}")
        return state.payload
    except Exception as exc:
        state.payload["status"] = "failed"
        state.payload["finished_at_utc"] = utc_now()
        state.payload["error"] = str(exc)
        persist_state(state, config.save_pipeline_state)
        print(f"[orchestrator] Failed: {exc}")
        raise


def run_stage_plan(
    config: MasterConfig,
    plan: StagePlan,
    state: PipelineState,
    alignment_results_by_candidate: dict[str, dict[str, Any]] | None = None,
) -> None:
    branch_root = state.run_dir / "branches" / plan.branch_id
    configs_dir = branch_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    branch_payload = {
        "branch_id": plan.branch_id,
        "candidate_id": plan.candidate_id,
        "candidate_kind": plan.candidate_kind,
        "method": plan.method,
        "inference_benchmark_id": plan.inference_benchmark_id,
        "evaluation_benchmark_id": plan.evaluation_benchmark_id,
        "run_name": plan.run_name,
        "status": "running",
        "started_at_utc": utc_now(),
        "paths": {
            "branch_root": str(branch_root),
            "configs_dir": str(configs_dir),
        },
        "alignment": {"status": "pending", "max_attempts": MAX_STAGE_ATTEMPTS, "attempts": []},
        "inference": {"status": "pending", "max_attempts": MAX_STAGE_ATTEMPTS, "attempts": []},
        "evaluation": {"status": "pending", "max_attempts": MAX_STAGE_ATTEMPTS, "attempts": []},
    }
    state.payload["branches"].append(branch_payload)
    persist_state(state, config.save_pipeline_state)
    print(f"[orchestrator] Starting branch={plan.branch_id}")

    try:
        alignment_result: dict[str, Any] | None = None
        inference_result: dict[str, Any] | None = None

        should_run_alignment = config.execute_alignment and plan.candidate_kind != "base"
        cached_alignment_result = (
            alignment_results_by_candidate.get(plan.candidate_id)
            if should_run_alignment
            and plan.candidate_id
            and alignment_results_by_candidate is not None
            else None
        )

        if cached_alignment_result is not None:
            alignment_result = cached_alignment_result
            _mark_stage_skipped(branch_payload, "alignment", reason="reused_existing_alignment")
            branch_payload["alignment"]["reused_candidate_id"] = plan.candidate_id
        elif should_run_alignment:
            alignment_config_path = configs_dir / "alignment.yaml"
            write_yaml(
                alignment_config_path,
                build_alignment_config(
                    config,
                    plan,
                    branch_root,
                    alignment_results_by_candidate=alignment_results_by_candidate,
                ),
            )
            print(f"[orchestrator] [alignment] Config: {alignment_config_path}")
            branch_payload["alignment"] = _prepare_stage_payload(
                branch_payload,
                "alignment",
                config_path=alignment_config_path,
            )
            alignment_result = _execute_stage_with_retry(
                branch_id=plan.branch_id,
                stage_name="alignment",
                stage_payload=branch_payload["alignment"],
                config_path=alignment_config_path,
                runner=run_alignment_stage,
                state=state,
                save_pipeline_state=config.save_pipeline_state,
            )
            if alignment_results_by_candidate is not None and plan.candidate_id:
                alignment_results_by_candidate[plan.candidate_id] = alignment_result
        else:
            reason = "base candidate" if plan.candidate_kind == "base" else None
            _mark_stage_skipped(branch_payload, "alignment", reason=reason)
        persist_state(state, config.save_pipeline_state)

        alignment_skip_reason = (
            alignment_result.get("reason")
            if alignment_result is not None and alignment_result.get("status") == "skipped"
            else None
        )

        if alignment_skip_reason is not None:
            _mark_stage_skipped(branch_payload, "inference", reason=alignment_skip_reason)
        elif config.execute_inference and plan.inference_benchmark_id is not None:
            inference_config_path = configs_dir / "inference.yaml"
            write_yaml(
                inference_config_path,
                build_inference_config(config, plan, branch_root, alignment_result),
            )
            print(f"[orchestrator] [inference] Config: {inference_config_path}")
            branch_payload["inference"] = _prepare_stage_payload(
                branch_payload,
                "inference",
                config_path=inference_config_path,
            )
            inference_result = _execute_stage_with_retry(
                branch_id=plan.branch_id,
                stage_name="inference",
                stage_payload=branch_payload["inference"],
                config_path=inference_config_path,
                runner=run_inference_stage,
                state=state,
                save_pipeline_state=config.save_pipeline_state,
            )
        else:
            _mark_stage_skipped(branch_payload, "inference")
        persist_state(state, config.save_pipeline_state)

        if alignment_skip_reason is not None:
            _mark_stage_skipped(branch_payload, "evaluation", reason=alignment_skip_reason)
        elif config.execute_evaluation and plan.evaluation_benchmark_id is not None:
            eval_config_path = configs_dir / "evaluation.yaml"
            write_yaml(
                eval_config_path,
                build_eval_config(config, plan, branch_root, inference_result),
            )
            print(f"[orchestrator] [eval] Config: {eval_config_path}")
            branch_payload["evaluation"] = _prepare_stage_payload(
                branch_payload,
                "evaluation",
                config_path=eval_config_path,
            )
            _execute_stage_with_retry(
                branch_id=plan.branch_id,
                stage_name="evaluation",
                stage_payload=branch_payload["evaluation"],
                config_path=eval_config_path,
                runner=run_evaluation_stage,
                state=state,
                save_pipeline_state=config.save_pipeline_state,
            )
        else:
            _mark_stage_skipped(branch_payload, "evaluation")

        branch_payload["status"] = "completed"
        branch_payload["finished_at_utc"] = utc_now()
        persist_state(state, config.save_pipeline_state)
        print(f"[orchestrator] Finished branch={plan.branch_id}")
    except StageExecutionError as exc:
        _mark_downstream_stages_skipped(
            branch_payload,
            failed_stage_name=exc.stage_name,
            reason=f"upstream_{exc.stage_name}_failed",
        )
        branch_payload["status"] = "failed"
        branch_payload["finished_at_utc"] = utc_now()
        branch_payload["error"] = str(exc)
        persist_state(state, config.save_pipeline_state)
        print(f"[orchestrator] Branch failed branch={plan.branch_id}: {exc}")
    except Exception as exc:
        branch_payload["status"] = "failed"
        branch_payload["finished_at_utc"] = utc_now()
        branch_payload["error"] = str(exc)
        persist_state(state, config.save_pipeline_state)
        print(f"[orchestrator] Branch failed branch={plan.branch_id}: {exc}")
        raise
