from __future__ import annotations

from .backends.base import InferenceBackend
from .backends.transformers_local import TransformersLocalBackend
from .schemas import GenerationConfig, InferenceModelConfig


def build_backend(
    model_config: InferenceModelConfig, generation_config: GenerationConfig
) -> InferenceBackend:
    backend = model_config.backend.strip().lower()
    if backend == "transformers_local":
        return TransformersLocalBackend(model_config, generation_config)
    raise ValueError(
        f"Unsupported inference backend '{model_config.backend}'. "
        "Currently only 'transformers_local' is supported."
    )

