from __future__ import annotations

from dataclasses import dataclass

from ..schemas import MetricConfig


@dataclass(frozen=True, slots=True)
class MetricPolicy:
    required_fields: tuple[str, ...]
    uses_judge: bool


METRIC_POLICIES: dict[str, MetricPolicy] = {
    "answer_relevancy": MetricPolicy(
        required_fields=("input", "actual_output"),
        uses_judge=True,
    ),
    "faithfulness": MetricPolicy(
        required_fields=("input", "actual_output", "retrieval_context"),
        uses_judge=True,
    ),
    "exact_match": MetricPolicy(
        required_fields=("actual_output", "expected_output"),
        uses_judge=False,
    ),
}

# Getter functions
def get_metric_policy(metric_name: str) -> MetricPolicy:
    normalized = metric_name.strip().lower()
    try:
        return METRIC_POLICIES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(METRIC_POLICIES))
        raise ValueError(f"Unsupported metric '{metric_name}'. Supported metrics: {supported}") from exc


def collect_required_fields(metrics: list[MetricConfig]) -> set[str]:
    required: set[str] = {"id", "input", "actual_output"}
    for metric in metrics:
        required.update(get_metric_policy(metric.name).required_fields)
    return required


def metrics_require_judge(metrics: list[MetricConfig]) -> bool:
    return any(get_metric_policy(metric.name).uses_judge for metric in metrics)
