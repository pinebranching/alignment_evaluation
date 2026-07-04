from __future__ import annotations

from deepeval.metrics import AnswerRelevancyMetric, ExactMatchMetric, FaithfulnessMetric

from ..policies.metric_requirements import get_metric_policy
from ..schemas import MetricConfig


def build_metrics(metric_configs: list[MetricConfig], judge_model) -> list[object]:
    metrics: list[object] = []
    for metric_config in metric_configs:
        name = metric_config.name.strip().lower()
        policy = get_metric_policy(name)
        threshold = metric_config.threshold

        if name == "answer_relevancy":
            metrics.append(
                AnswerRelevancyMetric(
                    threshold=0.5 if threshold is None else threshold,
                    model=judge_model if policy.uses_judge else None,
                    include_reason=metric_config.include_reason,
                    strict_mode=metric_config.strict_mode,
                    verbose_mode=metric_config.verbose_mode,
                )
            )
            continue

        if name == "faithfulness":
            metrics.append(
                FaithfulnessMetric(
                    threshold=0.5 if threshold is None else threshold,
                    model=judge_model if policy.uses_judge else None,
                    include_reason=metric_config.include_reason,
                    strict_mode=metric_config.strict_mode,
                    verbose_mode=metric_config.verbose_mode,
                )
            )
            continue

        if name == "exact_match":
            metrics.append(
                ExactMatchMetric(
                    threshold=1.0 if threshold is None else threshold,
                    verbose_mode=metric_config.verbose_mode,
                )
            )
            continue

        raise ValueError(f"Unsupported metric '{metric_config.name}'.")
    return metrics
