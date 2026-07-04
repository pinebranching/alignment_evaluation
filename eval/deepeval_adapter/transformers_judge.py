from __future__ import annotations

import asyncio
import threading
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.answer_relevancy.schema import (
    AnswerRelevancyScoreReason,
    AnswerRelevancyVerdict,
    Statements,
    Verdicts,
)
from outlines import Generator, from_transformers
from outlines.inputs import Chat
from pydantic import BaseModel

from inference.huggingface_runtime import (
    apply_stop_strings,
    build_chat_messages,
    load_causal_lm,
    render_prompt,
)

from ..schemas import JudgeModelConfig


class TransformersJudgeModel(DeepEvalBaseLLM):
    def __init__(self, config: JudgeModelConfig):
        self._config = config
        self._lock = threading.Lock()
        self._structured_generators: dict[type[BaseModel], Any] = {}
        self._runtime = None
        self._outlines_model = None
        super().__init__(config.model)

    def load_model(self):
        self._runtime = load_causal_lm(
            model_name_or_path=self._config.model,
            tokenizer_name_or_path=self._config.tokenizer_name_or_path,
            revision=self._config.revision,
            trust_remote_code=self._config.trust_remote_code,
            torch_dtype=self._config.torch_dtype,
            device=self._config.device,
        )
        self._outlines_model = from_transformers(
            self._runtime.model,
            self._runtime.tokenizer,
            device_dtype=self._runtime.device_dtype,
        )
        return self._runtime.model

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> tuple[str | BaseModel, float]:
        with self._lock:
            if schema is not None:
                return self._generate_structured(prompt, schema), 0
            return self._generate_text(prompt), 0

    async def a_generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> tuple[str | BaseModel, float]:
        return await asyncio.to_thread(self.generate, prompt, schema)

    def get_model_name(self):
        return f"{self.name} (Transformers + Outlines)"

    def supports_structured_outputs(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True

    def supports_temperature(self) -> bool:
        return True

    def _generate_text(self, prompt: str) -> str:
        rendered_prompt = render_prompt(
            self._runtime.tokenizer,
            prompt,
            add_generation_prompt=True,
        )
        inputs = self._runtime.tokenizer(rendered_prompt, return_tensors="pt")
        inputs = {
            key: value.to(self._runtime.model.device)
            for key, value in inputs.items()
        }

        generation_kwargs = self._build_generation_kwargs()
        with self._runtime.torch.no_grad():
            output_ids = self._runtime.model.generate(
                **inputs,
                **generation_kwargs,
            )

        prompt_token_count = int(inputs["input_ids"].shape[-1])
        completion_ids = output_ids[0][prompt_token_count:]
        text = self._runtime.tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
        ).strip()
        if not text:
            decoded = self._runtime.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
            ).strip()
            text = (
                decoded[len(rendered_prompt) :].strip()
                if decoded.startswith(rendered_prompt)
                else decoded
            )
        return apply_stop_strings(text, self._stop_strings)

    def _generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        generator = self._structured_generators.get(schema)
        if generator is None:
            generator = Generator(self._outlines_model, schema)
            self._structured_generators[schema] = generator

        structured_prompt: str | Chat
        if self._runtime.uses_chat_template:
            structured_prompt = Chat(build_chat_messages(prompt))
        else:
            structured_prompt = prompt

        original_max_tokens = self._config.generation_kwargs.get("max_new_tokens", 256)
        max_retries = 1
        retry_count = 0

        try:
            while True:
                try:
                    result = generator(
                        structured_prompt,
                        **self._build_generation_kwargs(include_stop_strings=False),
                    )
                    if isinstance(result, schema):
                        return result
                    if isinstance(result, str):
                        return schema.model_validate_json(result)
                    return schema.model_validate(result)
                except Exception as exc:
                    if self._is_retryable_json_error(exc) and retry_count < max_retries:
                        retry_count += 1
                        increased_tokens = original_max_tokens + 1024
                        print(
                            f"[judge] JSON truncation detected in structured generation. "
                            f"Retrying with increased max_new_tokens: {original_max_tokens} → {increased_tokens}"
                        )
                        self._config.generation_kwargs["max_new_tokens"] = increased_tokens
                        self._structured_generators.pop(schema, None)
                        generator = Generator(self._outlines_model, schema)
                        self._structured_generators[schema] = generator
                        continue

                    if self._is_retryable_timeout_error(exc) and retry_count < max_retries:
                        retry_count += 1
                        print(
                            f"[judge] Timeout detected in structured generation. "
                            f"Retrying once before using fallback JSON."
                        )
                        continue

                    print(
                        "[judge] Structured generation failed after retry. "
                        "Returning fallback JSON saying the result could not be evaluated."
                    )
                    return self._build_fallback_structured_result(schema)
        finally:
            self._config.generation_kwargs["max_new_tokens"] = original_max_tokens

    def _is_retryable_json_error(self, exc: Exception) -> bool:
        error_str = str(exc)
        return "EOF while parsing" in error_str or "json_invalid" in error_str

    def _is_retryable_timeout_error(self, exc: Exception) -> bool:
        return isinstance(exc, (asyncio.TimeoutError, TimeoutError))

    def _build_fallback_structured_result(self, schema: type[BaseModel]) -> BaseModel:
        if schema is Statements:
            return Statements(statements=["Could not be evaluated."])
        if schema is Verdicts:
            return Verdicts(
                verdicts=[
                    AnswerRelevancyVerdict(
                        verdict="idk",
                        reason="Could not be evaluated.",
                    )
                ]
            )
        if schema is AnswerRelevancyScoreReason:
            return AnswerRelevancyScoreReason(reason="Could not be evaluated.")
        return schema.model_construct()

    @property
    def _stop_strings(self) -> list[str]:
        stop_strings = self._config.generation_kwargs.get("stop_strings", [])
        if not isinstance(stop_strings, list):
            raise ValueError("judge_model.generation_kwargs.stop_strings must be a list.")
        return [str(item) for item in stop_strings]

    def _build_generation_kwargs(
        self,
        *,
        include_stop_strings: bool = True,
    ) -> dict[str, Any]:
        generation_kwargs = dict(self._config.generation_kwargs)
        if not include_stop_strings:
            generation_kwargs.pop("stop_strings", None)

        generation_kwargs.setdefault("max_new_tokens", 256)
        generation_kwargs.setdefault(
            "pad_token_id",
            self._runtime.tokenizer.pad_token_id,
        )

        temperature = self._config.temperature
        do_sample = generation_kwargs.get("do_sample")
        if do_sample is None and temperature is not None:
            do_sample = temperature > 0
            generation_kwargs["do_sample"] = do_sample
        if temperature is not None and do_sample:
            generation_kwargs.setdefault("temperature", temperature)
        if not generation_kwargs.get("do_sample", False):
            generation_kwargs.pop("temperature", None)
            generation_kwargs.pop("top_p", None)

        return generation_kwargs
