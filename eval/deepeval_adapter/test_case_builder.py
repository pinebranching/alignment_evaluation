from __future__ import annotations

from deepeval.test_case import LLMTestCase


def build_test_cases(eval_records: list[dict[str, object]]) -> list[LLMTestCase]:
    test_cases: list[LLMTestCase] = []
    for record in eval_records:
        test_cases.append(
            LLMTestCase(
                name=str(record["id"]),
                input=str(record["input"]),
                actual_output=str(record["actual_output"]),
                expected_output=_optional_text(record.get("expected_output")),
                context=_optional_list(record.get("context")),
                retrieval_context=_optional_list(record.get("retrieval_context")),
                additional_metadata=_optional_metadata(record.get("additional_metadata")),
            )
        )
    return test_cases


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_list(value: object) -> list[str] | None:
    if value is None:
        return None
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def _optional_metadata(value: object) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("additional_metadata must be a dictionary when provided.")
    return value
