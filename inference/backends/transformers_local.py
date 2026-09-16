from __future__ import annotations

from .base import GenerationResult
from ..huggingface_runtime import (
    apply_stop_strings,
    build_transformers_generation_kwargs,
    load_causal_lm,
    render_prompt,
)
from ..schemas import GenerationConfig, InferenceModelConfig


class TransformersLocalBackend:
    def __init__(self, model_config: InferenceModelConfig, generation_config: GenerationConfig):
        self._runtime = load_causal_lm(
            model_name_or_path=model_config.model_name_or_path,
            adapter_name_or_path=model_config.adapter_name_or_path,
            tokenizer_name_or_path=model_config.tokenizer_name_or_path,
            revision=model_config.revision,
            trust_remote_code=model_config.trust_remote_code,
            torch_dtype=model_config.torch_dtype,
            device=model_config.device,
        )
        self._generation = generation_config
        self.tokenizer = self._runtime.tokenizer
        self.model = self._runtime.model

    def generate_one(self, prompt: str) -> GenerationResult:
        rendered_prompt = render_prompt(
            self.tokenizer,
            prompt,
            add_generation_prompt=self._generation.add_generation_prompt,
            enable_thinking=self._generation.enable_thinking,
        )
        inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        generation_kwargs = build_transformers_generation_kwargs(
            self._generation,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        with self._runtime.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        prompt_token_count = int(inputs["input_ids"].shape[-1])
        completion_ids = output_ids[0][prompt_token_count:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        if not text:
            decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            text = (
                decoded[len(rendered_prompt) :].strip()
                if decoded.startswith(rendered_prompt)
                else decoded
            )
        text = apply_stop_strings(text, self._generation.stop_strings)
        completion_token_count = int(completion_ids.shape[-1])
        return GenerationResult(
            text=text,
            finish_reason="length" if completion_token_count >= self._generation.max_new_tokens else "stop",
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_token_count,
        )
