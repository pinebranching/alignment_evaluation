from __future__ import annotations

from .schemas import InferenceInputRecord


def build_prompt(record: InferenceInputRecord) -> str:
    return record.input.strip()

