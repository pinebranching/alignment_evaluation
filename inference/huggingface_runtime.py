from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .schemas import GenerationConfig


@dataclass(slots=True)
class LoadedCausalLM:
    model: Any
    tokenizer: Any
    torch: Any
    device_dtype: Any | None
    uses_chat_template: bool


def load_causal_lm(
    *,
    model_name_or_path: str,
    tokenizer_name_or_path: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    torch_dtype: str | None = None,
    device: str = "auto",
) -> LoadedCausalLM:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as transformers_logging
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "Hugging Face local generation requires 'transformers' and 'torch'."
        ) from exc

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    try:
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except Exception:  # pragma: no cover - best effort only
        pass

    tokenizer_name = tokenizer_name_or_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )

    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
    }
    device_dtype = None
    if torch_dtype:
        device_dtype = parse_torch_dtype(torch, torch_dtype)
        model_kwargs["torch_dtype"] = device_dtype
    if device.strip().lower() == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    if device.strip().lower() not in {"", "auto"}:
        model.to(device)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return LoadedCausalLM(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        device_dtype=device_dtype,
        uses_chat_template=tokenizer_has_chat_template(tokenizer),
    )


def parse_torch_dtype(torch_module: Any, raw_value: str):
    normalized = raw_value.strip().lower()
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(mapping))
        raise ValueError(
            f"Unsupported torch_dtype '{raw_value}'. Supported: {supported}"
        ) from exc


def tokenizer_has_chat_template(tokenizer: Any) -> bool:
    try:
        tokenizer.get_chat_template()
        return True
    except ValueError:
        return False
    except AttributeError:
        return False


def build_chat_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def render_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    add_generation_prompt: bool = True,
) -> str:
    if tokenizer_has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            build_chat_messages(prompt),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return prompt


def build_transformers_generation_kwargs(
    generation_config: GenerationConfig,
    *,
    pad_token_id: int | None,
) -> dict[str, Any]:
    generation_kwargs = {
        "max_new_tokens": generation_config.max_new_tokens,
        "repetition_penalty": generation_config.repetition_penalty,
        "pad_token_id": pad_token_id,
    }
    do_sample = generation_config.do_sample
    if do_sample is None:
        do_sample = generation_config.temperature > 0
    generation_kwargs["do_sample"] = do_sample
    if do_sample:
        generation_kwargs["temperature"] = generation_config.temperature
        generation_kwargs["top_p"] = generation_config.top_p
    return generation_kwargs


def apply_stop_strings(text: str, stop_strings: list[str]) -> str:
    if not stop_strings:
        return text
    cutoff = len(text)
    for stop in stop_strings:
        if not stop:
            continue
        index = text.find(stop)
        if index >= 0:
            cutoff = min(cutoff, index)
    return text[:cutoff].strip()


@contextmanager
def temporary_runtime_env(
    *,
    target: str,
    cuda_visible_devices: list[str] | None = None,
    cpu_threads: int = 0,
):
    updates: dict[str, str] = {}
    if cpu_threads > 0:
        updates["OMP_NUM_THREADS"] = str(cpu_threads)
    if target.strip().lower() != "cpu" and cuda_visible_devices:
        updates["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)

    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
