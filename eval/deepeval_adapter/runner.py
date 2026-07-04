from __future__ import annotations

import asyncio
import contextlib
import io
import os

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.evaluate.types import EvaluationResult, TestResult
from deepeval.tracing.api import MetricData
from deepeval.utils import get_per_task_timeout_seconds

from ..schemas import ExecutionConfig


def run_deepeval(
    test_cases: list[object],
    metrics: list[object],
    execution_config: ExecutionConfig,
    output_dir,
) -> EvaluationResult:
    if execution_config.batch_size and execution_config.batch_size > 0:
        all_results = []
        confident_link = None
        test_run_id = None
        for batch in _chunked(test_cases, execution_config.batch_size):
            batch_result = _evaluate_quietly_with_timeout_retry(
                test_cases=batch,
                metrics=metrics,
                async_config=AsyncConfig(
                    run_async=True,
                    max_concurrent=execution_config.max_concurrency,
                ),
                display_config=DisplayConfig(
                    show_indicator=False,
                    print_results=False,
                    file_output_dir=str(output_dir),
                ),
                cache_config=CacheConfig(write_cache=False, use_cache=False),
                error_config=ErrorConfig(ignore_errors=False, skip_on_missing_params=False),
            )
            all_results.extend(batch_result.test_results)
            confident_link = confident_link or batch_result.confident_link
            test_run_id = test_run_id or batch_result.test_run_id
        return EvaluationResult(
            test_results=all_results,
            confident_link=confident_link,
            test_run_id=test_run_id,
        )

    return _evaluate_quietly_with_timeout_retry(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            run_async=True,
            max_concurrent=execution_config.max_concurrency,
        ),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            file_output_dir=str(output_dir),
        ),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        error_config=ErrorConfig(ignore_errors=False, skip_on_missing_params=False),
    )


def _chunked(items: list[object], size: int) -> list[list[object]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _evaluate_quietly_with_timeout_retry(**kwargs) -> EvaluationResult:
    """Evaluate with automatic retry on timeout, giving more time for slower cases."""
    max_timeout_retries = 1
    timeout_retry_count = 0
    timeout_increase_seconds = 120.0
    timeout_override_key = "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"

    while timeout_retry_count <= max_timeout_retries:
        try:
            if timeout_retry_count == 0:
                return _evaluate_quietly(**kwargs)

            current_timeout = get_per_task_timeout_seconds()
            retry_timeout = current_timeout + timeout_increase_seconds
            print(
                f"[eval] Timeout detected during evaluation. "
                f"Increasing DeepEval per-task timeout from {current_timeout:.0f}s to {retry_timeout:.0f}s "
                f"for retry (attempt {timeout_retry_count + 1}/{max_timeout_retries + 1})..."
            )
            with _temporary_env_override(timeout_override_key, str(retry_timeout)):
                return _evaluate_quietly(**kwargs)
        except (asyncio.TimeoutError, TimeoutError):
            if timeout_retry_count < max_timeout_retries:
                timeout_retry_count += 1
                print(
                    f"[eval] Timeout detected during evaluation. Increasing DeepEval per-task timeout by {timeout_increase_seconds:.0f}s and retrying..."
                )
            else:
                print(
                    "[eval] Retry also timed out. Returning fallback evaluation results marked as could not be evaluated."
                )
                return _build_timeout_fallback_result(**kwargs)


@contextlib.contextmanager
def _temporary_env_override(key: str, value: str):
    original_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


def _build_timeout_fallback_result(**kwargs) -> EvaluationResult:
    test_cases = kwargs.get("test_cases", [])
    metrics = kwargs.get("metrics", [])
    fallback_test_results: list[TestResult] = []

    for index, test_case in enumerate(test_cases):
        metric_data_list = [
            MetricData.model_construct(
                name=getattr(metric, "__name__", metric.__class__.__name__),
                threshold=float(getattr(metric, "threshold", 0.0) or 0.0),
                success=False,
                score=0.0,
                reason="Could not be evaluated.",
                strict_mode=bool(getattr(metric, "strict_mode", False)),
                evaluation_model=getattr(metric, "evaluation_model", None),
                error="Could not be evaluated.",
                evaluation_cost=None,
                verbose_logs=None,
            )
            for metric in metrics
        ]

        fallback_test_results.append(
            TestResult(
                name=getattr(test_case, "name", f"fallback_case_{index}"),
                success=False,
                metrics_data=metric_data_list,
                conversational=False,
                multimodal=getattr(test_case, "multimodal", None),
                input=getattr(test_case, "input", None),
                actual_output=getattr(test_case, "actual_output", None),
                expected_output=getattr(test_case, "expected_output", None),
                context=getattr(test_case, "context", None),
                retrieval_context=getattr(test_case, "retrieval_context", None),
                metadata=getattr(test_case, "metadata", None),
            )
        )

    return EvaluationResult(
        test_results=fallback_test_results,
        confident_link=None,
        test_run_id=None,
    )


def _evaluate_quietly(**kwargs) -> EvaluationResult:
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return evaluate(**kwargs)
