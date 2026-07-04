from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .config import load_eval_config, validate_eval_config
from .deepeval_adapter.judge_factory import build_judge
from .deepeval_adapter.metric_factory import build_metrics
from .deepeval_adapter.runner import run_deepeval
from .deepeval_adapter.test_case_builder import build_test_cases
from .exporters.csv_exporter import export_csv_artifacts
from .exporters.json_exporter import export_json_artifacts
from .exporters.result_normalizer import normalize_results
from .exporters.visualization_exporter import export_visualization_artifacts
from .preparation.eval_record_preparer import prepare_eval_records
from inference.huggingface_runtime import temporary_runtime_env


def run_evaluation(config_path: str | Path) -> dict[str, Any]:
    print(f"[eval] Loading config: {config_path}")
    eval_config = load_eval_config(config_path)
    validate_eval_config(eval_config)

    eval_config.output.dir.mkdir(parents=True, exist_ok=True)

    eval_records = prepare_eval_records(eval_config)
    test_cases = build_test_cases(eval_records)
    with temporary_runtime_env(
        target=eval_config.runtime.target,
        cuda_visible_devices=eval_config.runtime.cuda_visible_devices,
        cpu_threads=eval_config.runtime.cpu_threads,
    ):
        judge_model = build_judge(eval_config.judge_model)
        metrics = build_metrics(eval_config.metrics, judge_model)
        print(f"[eval] Running evaluation with {len(test_cases)} test cases and {len(metrics)} metrics.")
        print(f"[eval] Results will be saved to: {eval_config.output.dir}")
        evaluation_result = run_deepeval(
            test_cases=test_cases,
            metrics=metrics,
            execution_config=eval_config.execution,
            output_dir=eval_config.output.dir,
        )
    normalized = normalize_results(evaluation_result, eval_records, eval_config)
    export_results(eval_config, normalized)
    print(f"[eval] Finished. output_dir={eval_config.output.dir}")
    return normalized


def export_results(eval_config, normalized_results: dict[str, Any]) -> None:
    if eval_config.output.save_json:
        export_json_artifacts(
            eval_config.output.dir,
            normalized_results["summary"],
            normalized_results["per_case_records"],
        )

    if eval_config.output.save_csv:
        export_csv_artifacts(
            eval_config.output.dir,
            normalized_results["metric_summary_rows"],
            normalized_results["metric_case_rows"],
            normalized_results["analysis_case_rows"],
        )

    if eval_config.output.save_visualizations:
        export_visualization_artifacts(
            eval_config.output.dir,
            normalized_results["metric_summary_rows"],
            normalized_results["metric_case_rows"],
        )

    if eval_config.output.save_config_snapshot:
        snapshot_path = eval_config.output.dir / "config_snapshot.json"
        snapshot_path.write_text(
            json.dumps(_to_jsonable(eval_config), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value
