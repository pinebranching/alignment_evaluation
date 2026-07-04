from __future__ import annotations

import time
from typing import Any

from .backends.base import InferenceBackend
from .prompt_builder import build_prompt
from .schemas import InferenceConfig, InferenceInputRecord


def run_inference_records(
    config: InferenceConfig,
    records: list[InferenceInputRecord],
    backend: InferenceBackend,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output_rows: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0

    for record in records:
        prompt = build_prompt(record)
        started = time.perf_counter()
        try:
            result = backend.generate_one(prompt)
            latency_ms = int((time.perf_counter() - started) * 1000)
            if not result.text.strip():
                output_rows.append(
                    _build_output_row(
                        record=record,
                        config=config,
                        actual_output="empty_output",
                        latency_ms=latency_ms,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        finish_reason="empty_output",
                        error=None,
                    )
                )
                succeeded += 1
                continue
            output_rows.append(
                _build_output_row(
                    record=record,
                    config=config,
                    actual_output=result.text,
                    latency_ms=latency_ms,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    finish_reason=result.finish_reason,
                    error=None,
                )
            )
            succeeded += 1
        except Exception as exc:  # pragma: no cover - best effort for runtime resiliency
            latency_ms = int((time.perf_counter() - started) * 1000)
            output_rows.append(
                _build_output_row(
                    record=record,
                    config=config,
                    actual_output="",
                    latency_ms=latency_ms,
                    prompt_tokens=None,
                    completion_tokens=None,
                    finish_reason="error",
                    error=str(exc),
                )
            )
            failed += 1

    return output_rows, {"succeeded": succeeded, "failed": failed}


def _build_output_row(
    *,
    record: InferenceInputRecord,
    config: InferenceConfig,
    actual_output: str,
    latency_ms: int,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    finish_reason: str | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "id": record.id,
        "input": record.input,
        "expected_output": record.expected_output,
        "context": record.context,
        "retrieval_context": record.retrieval_context,
        "additional_metadata": record.additional_metadata,
        "actual_output": actual_output,
        "model_name": config.model.model_name_or_path,
        "backend": config.model.backend,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "error": error,
    }
