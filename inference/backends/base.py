from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class GenerationResult:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class InferenceBackend(Protocol):
    def generate_one(self, prompt: str) -> GenerationResult:
        ...

