from __future__ import annotations

from collections import defaultdict
import json
from statistics import mean
from typing import Any

from deepeval.evaluate.types import EvaluationResult, MetricData

from ..schemas import EvalConfig


def normalize_results(
    evaluation_result: EvaluationResult,
    eval_records: list[dict[str, Any]],
    config: EvalConfig,
) -> dict[str, Any]:
    record_lookup = {record["id"]: record for record in eval_records}
    metric_case_rows: list[dict[str, Any]] = []
    analysis_case_rows: list[dict[str, Any]] = []
    per_case_records: list[dict[str, Any]] = []
    metric_scores: dict[str, list[float]] = defaultdict(list)
    metric_successes: dict[str, list[bool]] = defaultdict(list)

    for test_result in evaluation_result.test_results:
        case_id = test_result.name
        source_record = record_lookup.get(case_id, {})
        metric_payload: dict[str, Any] = {}
        for metric_data in test_result.metrics_data or []:
            metric_dict = _metric_to_dict(metric_data)
            metric_payload[metric_data.name] = metric_dict
            metric_case_rows.append(
                {
                    "case_id": case_id,
                    "input": test_result.input,
                    "expected_output": test_result.expected_output,
                    "actual_output": test_result.actual_output,
                    "context": _jsonable_cell(getattr(test_result, "context", None)),
                    "retrieval_context": _jsonable_cell(getattr(test_result, "retrieval_context", None)),
                    "additional_metadata": _jsonable_cell(
                        getattr(test_result, "additional_metadata", None)
                        or source_record.get("additional_metadata")
                    ),
                    "model_name": source_record.get("model_name"),
                    "backend": source_record.get("backend"),
                    "latency_ms": source_record.get("latency_ms"),
                    "prompt_tokens": source_record.get("prompt_tokens"),
                    "completion_tokens": source_record.get("completion_tokens"),
                    "finish_reason": source_record.get("finish_reason"),
                    "inference_error": source_record.get("inference_error"),
                    "metric": metric_data.name,
                    "score": metric_data.score,
                    "success": metric_data.success,
                    "threshold": metric_data.threshold,
                    "reason": metric_data.reason,
                    "evaluation_model": metric_data.evaluation_model,
                    "error": metric_data.error,
                }
            )
            if metric_data.score is not None:
                metric_scores[metric_data.name].append(float(metric_data.score))
            metric_successes[metric_data.name].append(bool(metric_data.success))

        per_case_records.append(
            {
                "case_id": case_id,
                "input": test_result.input,
                "expected_output": test_result.expected_output,
                "actual_output": test_result.actual_output,
                "context": getattr(test_result, "context", None),
                "retrieval_context": getattr(test_result, "retrieval_context", None),
                "additional_metadata": getattr(test_result, "additional_metadata", None) or source_record.get("additional_metadata"),
                "model_name": source_record.get("model_name"),
                "backend": source_record.get("backend"),
                "latency_ms": source_record.get("latency_ms"),
                "prompt_tokens": source_record.get("prompt_tokens"),
                "completion_tokens": source_record.get("completion_tokens"),
                "finish_reason": source_record.get("finish_reason"),
                "inference_error": source_record.get("inference_error"),
                "success": test_result.success,
                "metrics": metric_payload,
            }
        )
        analysis_case_rows.append(
            _build_analysis_case_row(
                case_id=case_id,
                test_result=test_result,
                source_record=source_record,
                metric_payload=metric_payload,
            )
        )

    metric_summary_rows: list[dict[str, Any]] = []
    for metric_name in sorted(metric_successes):
        scores = metric_scores.get(metric_name, [])
        successes = metric_successes[metric_name]
        metric_summary_rows.append(
            {
                "metric": metric_name,
                "num_cases": len(successes),
                "mean_score": mean(scores) if scores else None,
                "pass_rate": sum(1 for success in successes if success) / len(successes),
            }
        )

    summary = {
        "config_path": str(config.config_path),
        "mode": config.mode,
        "num_cases": len(per_case_records),
        "num_metrics": len(config.metrics),
        "judge_model": _judge_summary(config),
        "metrics": metric_summary_rows,
        "confident_link": evaluation_result.confident_link,
        "test_run_id": evaluation_result.test_run_id,
    }
    return {
        "summary": summary,
        "per_case_records": per_case_records,
        "metric_summary_rows": metric_summary_rows,
        "metric_case_rows": metric_case_rows,
        "analysis_case_rows": analysis_case_rows,
    }


def _metric_to_dict(metric_data: MetricData) -> dict[str, Any]:
    if hasattr(metric_data, "model_dump"):
        return metric_data.model_dump(by_alias=True)
    return metric_data.dict(by_alias=True)


def _judge_summary(config: EvalConfig) -> dict[str, Any] | None:
    if config.judge_model is None:
        return None
    return {
        "backend": config.judge_model.backend,
        "model": config.judge_model.model,
    }


def _build_analysis_case_row(
    *,
    case_id: str,
    test_result: Any,
    source_record: dict[str, Any],
    metric_payload: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "input": getattr(test_result, "input", None),
        "expected_output": getattr(test_result, "expected_output", None),
        "actual_output": getattr(test_result, "actual_output", None),
        "context": _jsonable_cell(getattr(test_result, "context", None)),
        "retrieval_context": _jsonable_cell(getattr(test_result, "retrieval_context", None)),
        "additional_metadata": _jsonable_cell(
            getattr(test_result, "additional_metadata", None)
            or source_record.get("additional_metadata")
        ),
        "model_name": source_record.get("model_name"),
        "backend": source_record.get("backend"),
        "latency_ms": source_record.get("latency_ms"),
        "prompt_tokens": source_record.get("prompt_tokens"),
        "completion_tokens": source_record.get("completion_tokens"),
        "finish_reason": source_record.get("finish_reason"),
        "inference_error": source_record.get("inference_error"),
        "overall_success": getattr(test_result, "success", None),
    }
    for metric_name, metric_details in metric_payload.items():
        prefix = _sanitize_metric_name(metric_name)
        row[f"{prefix}__score"] = metric_details.get("score")
        row[f"{prefix}__success"] = metric_details.get("success")
        row[f"{prefix}__threshold"] = metric_details.get("threshold")
        row[f"{prefix}__reason"] = metric_details.get("reason")
        row[f"{prefix}__evaluation_model"] = metric_details.get("evaluation_model")
        row[f"{prefix}__error"] = metric_details.get("error")
    return row


def _sanitize_metric_name(metric_name: str) -> str:
    rendered = "".join(
        character.lower() if character.isalnum() else "_"
        for character in metric_name
    )
    while "__" in rendered:
        rendered = rendered.replace("__", "_")
    return rendered.strip("_") or "metric"


def _jsonable_cell(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)
