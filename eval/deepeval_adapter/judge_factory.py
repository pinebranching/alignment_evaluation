from __future__ import annotations

from ..schemas import JudgeModelConfig
from .transformers_judge import TransformersJudgeModel


def build_judge(judge_config: JudgeModelConfig | None):
    if judge_config is None:
        return None

    backend = judge_config.backend.strip().lower()
    if backend in {"transformers_local", "huggingface", "hf"}:
        return TransformersJudgeModel(judge_config)

    raise ValueError(
        f"Unsupported judge backend '{judge_config.backend}'. "
        "Currently only the local Hugging Face backend is supported."
    )
