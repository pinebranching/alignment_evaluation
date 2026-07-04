"""Translate the pipeline-owned alignment config into LLaMA-Factory YAML payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import AlignmentConfig, PreparedDataset


def build_train_config(
    config: AlignmentConfig,
    prepared_dataset: PreparedDataset,
    adapter_output_dir: Path,
) -> dict[str, Any]:
    """Build the LLaMA-Factory train config for the selected method."""

    payload = _build_common_train_config(config, prepared_dataset, adapter_output_dir)

    if config.method == "sft":
        payload["stage"] = "sft"
    elif config.method == "dpo":
        payload["stage"] = "dpo"
        payload["pref_beta"] = config.method_config.dpo.pref_beta
        payload["pref_loss"] = config.method_config.dpo.pref_loss
    elif config.method == "rm":
        payload["stage"] = "rm"
    elif config.method == "kto":
        kto_config = config.method_config.kto
        payload["stage"] = "kto"
        payload["pref_beta"] = kto_config.pref_beta
        payload["kto_chosen_weight"] = kto_config.kto_chosen_weight
        payload["kto_rejected_weight"] = kto_config.kto_rejected_weight
        _set_if_not_none(payload, "ref_model", kto_config.ref_model)
        _set_if_not_none(payload, "ref_model_adapters", kto_config.ref_model_adapters)
        _set_if_not_none(
            payload,
            "ref_model_quantization_bit",
            kto_config.ref_model_quantization_bit,
        )
    elif config.method == "ppo":
        ppo_config = config.method_config.ppo
        payload["stage"] = "ppo"
        payload["reward_model"] = ppo_config.reward_model_path
        payload["reward_model_type"] = ppo_config.reward_model_type
        payload["ppo_buffer_size"] = ppo_config.ppo_buffer_size
        payload["ppo_epochs"] = ppo_config.ppo_epochs
        payload["ppo_score_norm"] = ppo_config.ppo_score_norm
        payload["ppo_target"] = ppo_config.ppo_target
        payload["ppo_whiten_rewards"] = ppo_config.ppo_whiten_rewards
        payload["max_new_tokens"] = ppo_config.max_new_tokens
        payload["temperature"] = ppo_config.temperature
        payload["top_p"] = ppo_config.top_p
        payload["top_k"] = ppo_config.top_k
        _set_if_not_none(payload, "reward_model_adapters", ppo_config.reward_model_adapters)
        _set_if_not_none(
            payload,
            "reward_model_quantization_bit",
            ppo_config.reward_model_quantization_bit,
        )
        _set_if_not_none(payload, "ref_model", ppo_config.ref_model)
        _set_if_not_none(payload, "ref_model_adapters", ppo_config.ref_model_adapters)
        _set_if_not_none(
            payload,
            "ref_model_quantization_bit",
            ppo_config.ref_model_quantization_bit,
        )
    else:
        raise ValueError(f"Unsupported alignment method: {config.method}")

    return payload


def _set_if_not_none(payload: dict[str, Any], key: str, value: Any) -> None:
    """Add optional LLaMA-Factory fields only when explicitly configured."""

    if value is not None:
        payload[key] = value


def build_export_config(
    config: AlignmentConfig,
    adapter_output_dir: Path,
    export_output_dir: Path,
) -> dict[str, Any]:
    """Build the LLaMA-Factory export config for merging LoRA adapters."""

    payload: dict[str, Any] = {
        "model_name_or_path": config.model.base_model_name_or_path,
        "adapter_name_or_path": str(adapter_output_dir),
        "template": config.model.template,
        "trust_remote_code": config.model.trust_remote_code,
        "export_dir": str(export_output_dir),
        "export_device": config.output.export_device,
        "export_legacy_format": False,
    }

    if config.tuning.quantization_bit is not None:
        raise ValueError(
            "Merged export should not be used with quantized models in this implementation."
        )

    return payload


def _build_common_train_config(
    config: AlignmentConfig,
    prepared_dataset: PreparedDataset,
    adapter_output_dir: Path,
) -> dict[str, Any]:
    """Build the shared train config fields used by all current methods."""

    payload: dict[str, Any] = {
        "model_name_or_path": config.model.base_model_name_or_path,
        "trust_remote_code": config.model.trust_remote_code,
        "do_train": True,
        "finetuning_type": config.tuning.finetuning_type,
        "dataset": prepared_dataset.dataset_name,
        "dataset_dir": str(prepared_dataset.dataset_dir),
        "template": config.model.template,
        "cutoff_len": config.training.cutoff_len,
        "overwrite_cache": True,
        "output_dir": str(adapter_output_dir),
        "logging_steps": config.training.logging_steps,
        "save_steps": config.training.save_steps,
        "plot_loss": True,
        "overwrite_output_dir": config.output.overwrite_output_dir,
        "save_only_model": config.output.save_only_model,
        "report_to": config.output.report_to,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "num_train_epochs": config.training.num_train_epochs,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "warmup_ratio": config.training.warmup_ratio,
        "resume_from_checkpoint": None,
    }

    disable_gradient_checkpointing = config.model.disable_gradient_checkpointing
    if disable_gradient_checkpointing is None and config.runtime.target == "cpu":
        disable_gradient_checkpointing = True
    if disable_gradient_checkpointing is not None:
        payload["disable_gradient_checkpointing"] = disable_gradient_checkpointing

    if config.training.preprocessing_num_workers > 0:
        payload["preprocessing_num_workers"] = config.training.preprocessing_num_workers
    if config.training.dataloader_num_workers > 0:
        payload["dataloader_num_workers"] = config.training.dataloader_num_workers

    if config.dataset.max_samples is not None:
        payload["max_samples"] = config.dataset.max_samples
    if config.training.save_total_limit is not None:
        payload["save_total_limit"] = config.training.save_total_limit
    if config.model.tokenizer_name_or_path:
        payload["tokenizer_name_or_path"] = config.model.tokenizer_name_or_path
    if config.tuning.quantization_bit is not None:
        payload["quantization_bit"] = config.tuning.quantization_bit

    if config.tuning.finetuning_type == "lora":
        payload["lora_rank"] = config.tuning.lora_rank
        payload["lora_alpha"] = config.tuning.lora_alpha
        payload["lora_dropout"] = config.tuning.lora_dropout
        payload["lora_target"] = config.tuning.lora_target

    payload["bf16"] = config.runtime.mixed_precision == "bf16"
    payload["fp16"] = config.runtime.mixed_precision == "fp16"
    
    return payload
