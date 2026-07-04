"""Load and validate the high-level alignment configuration for a single run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .method_registry import get_method_spec
from .schemas import (
    AlignmentConfig,
    AlignmentDatasetConfig,
    AlignmentDPOConfig,
    AlignmentKTOConfig,
    AlignmentMethodConfig,
    AlignmentModelConfig,
    AlignmentOutputConfig,
    AlignmentPPOConfig,
    AlignmentRuntimeConfig,
    AlignmentTrainingConfig,
    AlignmentTuningConfig,
)


def load_alignment_config(config_path: str | Path) -> AlignmentConfig:
    """Read an alignment YAML file and convert it into typed config objects."""

    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"YAML root in {path} must be a dict.")

    alignment_raw = raw.get("alignment", raw)
    if not isinstance(alignment_raw, dict):
        raise ValueError("The alignment config must be a dict.")

    base_dir = path.parent
    model_raw = _require_mapping(alignment_raw.get("model"), "model")
    tuning_raw = _require_mapping(alignment_raw.get("tuning"), "tuning")
    dataset_raw = _require_mapping(alignment_raw.get("dataset"), "dataset")
    runtime_raw = _require_mapping(alignment_raw.get("runtime"), "runtime")
    training_raw = _require_mapping(alignment_raw.get("training"), "training")
    method_raw = _optional_mapping(alignment_raw.get("method_config"), "method_config")
    output_raw = _require_mapping(alignment_raw.get("output"), "output")

    raw_lora_target = tuning_raw.get("lora_target", "all")
    if isinstance(raw_lora_target, str):
        lora_target = raw_lora_target
    elif isinstance(raw_lora_target, list):
        lora_target = [str(item) for item in raw_lora_target]
    else:
        raise ValueError("Expected a string or a list of strings.")

    raw_cuda_visible_devices = runtime_raw.get("cuda_visible_devices", [])
    if raw_cuda_visible_devices is None:
        cuda_visible_devices = []
    elif not isinstance(raw_cuda_visible_devices, list):
        raise ValueError("Expected a list of strings.")
    else:
        cuda_visible_devices = [str(item) for item in raw_cuda_visible_devices]

    raw_extra_env = runtime_raw.get("extra_env", {})
    if raw_extra_env is None:
        extra_env = {}
    elif not isinstance(raw_extra_env, dict):
        raise ValueError("runtime.extra_env must be a mapping when provided.")
    else:
        extra_env = {str(key): str(value) for key, value in raw_extra_env.items()}

    return AlignmentConfig(
        backend=str(alignment_raw.get("backend", "llama_factory")),
        method=str(alignment_raw.get("method", "")),
        run_name=str(alignment_raw.get("run_name", "")),
        model=AlignmentModelConfig(
            base_model_name_or_path=str(model_raw.get("base_model_name_or_path", "")),
            template=str(model_raw.get("template", "")),
            tokenizer_name_or_path=str(v)
            if (v := model_raw.get("tokenizer_name_or_path")) is not None
            else None,
            trust_remote_code=bool(model_raw.get("trust_remote_code", False)),
            disable_gradient_checkpointing=bool(v)
            if (v := model_raw.get("disable_gradient_checkpointing")) is not None
            else None,
        ),
        tuning=AlignmentTuningConfig(
            finetuning_type=str(tuning_raw.get("finetuning_type", "lora")),
            lora_rank=int(tuning_raw.get("lora_rank", 8)),
            lora_alpha=int(tuning_raw.get("lora_alpha", 16)),
            lora_dropout=float(tuning_raw.get("lora_dropout", 0.05)),
            lora_target=lora_target,
            quantization_bit=int(v) if (v := tuning_raw.get("quantization_bit")) is not None else None,
        ),
        dataset=AlignmentDatasetConfig(
            path=_resolve_path(base_dir, dataset_raw.get("path")),
            format=str(dataset_raw.get("format", "jsonl")),
            schema=str(dataset_raw.get("schema", "")),
            prompt_field=str(dataset_raw.get("prompt_field", "prompt")),
            response_field=str(v) if (v := dataset_raw.get("response_field")) is not None else None,
            chosen_field=str(v) if (v := dataset_raw.get("chosen_field")) is not None else None,
            rejected_field=str(v) if (v := dataset_raw.get("rejected_field")) is not None else None,
            label_field=str(v) if (v := dataset_raw.get("label_field")) is not None else None,
            system_field=str(v) if (v := dataset_raw.get("system_field")) is not None else None,
            max_samples=int(v) if (v := dataset_raw.get("max_samples")) is not None else None,
        ),
        runtime=AlignmentRuntimeConfig(
            target=str(runtime_raw.get("target", "cpu")),
            device=str(runtime_raw.get("device", "cpu")),
            mixed_precision=str(runtime_raw.get("mixed_precision", "none")),
            cpu_threads=int(runtime_raw.get("cpu_threads", 0)),
            cuda_visible_devices=cuda_visible_devices,
            command_mode=str(runtime_raw.get("command_mode", "module")),
            python_executable=str(v)
            if (v := runtime_raw.get("python_executable")) is not None
            else None,
            force_torchrun=bool(runtime_raw.get("force_torchrun", False)),
            nproc_per_node=int(v)
            if (v := runtime_raw.get("nproc_per_node")) is not None
            else None,
            extra_env=extra_env,
            heartbeat_interval_seconds=int(runtime_raw.get("heartbeat_interval_seconds", 30)),
        ),
        training=AlignmentTrainingConfig(
            cutoff_len=int(training_raw.get("cutoff_len", 1024)),
            per_device_train_batch_size=int(training_raw.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(training_raw.get("gradient_accumulation_steps", 1)),
            learning_rate=float(training_raw.get("learning_rate", 1e-4)),
            num_train_epochs=float(training_raw.get("num_train_epochs", 1.0)),
            lr_scheduler_type=str(training_raw.get("lr_scheduler_type", "cosine")),
            warmup_ratio=float(training_raw.get("warmup_ratio", 0.0)),
            logging_steps=int(training_raw.get("logging_steps", 10)),
            save_steps=int(training_raw.get("save_steps", 100)),
            save_total_limit=int(v) if (v := training_raw.get("save_total_limit")) is not None else None,
            dataloader_num_workers=int(training_raw.get("dataloader_num_workers", 0)),
            preprocessing_num_workers=int(training_raw.get("preprocessing_num_workers", 1)),
        ),
        method_config=AlignmentMethodConfig(
            sft=_optional_mapping(method_raw.get("sft"), "method_config.sft"),
            dpo=_build_dpo_config(method_raw.get("dpo")),
            kto=_build_kto_config(method_raw.get("kto")),
            ppo=_build_ppo_config(method_raw.get("ppo")),
        ),
        output=AlignmentOutputConfig(
            root_dir=_resolve_path(base_dir, output_raw.get("root_dir")),
            save_adapter_checkpoint=bool(output_raw.get("save_adapter_checkpoint", True)),
            export_merged_model=bool(output_raw.get("export_merged_model", True)),
            export_device=str(output_raw.get("export_device", "cpu")),
            overwrite_output_dir=bool(output_raw.get("overwrite_output_dir", False)),
            save_only_model=bool(output_raw.get("save_only_model", False)),
            report_to=str(output_raw.get("report_to", "none")),
        ),
        config_path=path,
        raw_alignment_config=alignment_raw,
    )


def validate_alignment_config(config: AlignmentConfig) -> None:
    """Enforce the alignment contract and surface future-method placeholders clearly."""

    if config.backend != "llama_factory":
        raise ValueError("Only alignment.backend='llama_factory' is supported.")
    if not config.method:
        raise ValueError("alignment.method must not be empty.")
    if not config.run_name:
        raise ValueError("alignment.run_name must not be empty.")
    if not config.model.base_model_name_or_path:
        raise ValueError("model.base_model_name_or_path must not be empty.")
    if not config.model.template:
        raise ValueError("model.template must not be empty.")

    method_spec = get_method_spec(config.method)
    if not bool(method_spec["implemented"]):
        raise ValueError(
            f"alignment.method='{config.method}' is not implemented yet in this module."
        )

    if config.dataset.format.lower() not in {"json", "jsonl"}:
        raise ValueError("Only dataset.format='json' or 'jsonl' is supported.")
    if not config.dataset.path.exists():
        raise ValueError(f"Dataset file does not exist: {config.dataset.path}")
    if config.dataset.max_samples is not None and config.dataset.max_samples <= 0:
        raise ValueError("dataset.max_samples must be > 0 when provided.")

    expected_schema = str(method_spec["dataset_schema"])
    if config.dataset.schema != expected_schema:
        raise ValueError(
            f"alignment.method='{config.method}' requires dataset.schema='{expected_schema}'."
        )

    if config.dataset.schema == "prompt_response":
        if not config.dataset.response_field:
            raise ValueError("dataset.response_field is required for prompt_response datasets.")
    if config.dataset.schema == "prompt_response_label":
        if not config.dataset.response_field or not config.dataset.label_field:
            raise ValueError(
                "dataset.response_field and dataset.label_field are required for prompt_response_label datasets."
            )
    if config.dataset.schema == "prompt_chosen_rejected":
        if not config.dataset.chosen_field or not config.dataset.rejected_field:
            raise ValueError(
                "dataset.chosen_field and dataset.rejected_field are required for preference datasets."
            )

    if config.runtime.target not in {"cpu", "single_gpu", "multi_gpu"}:
        raise ValueError("runtime.target must be one of: cpu, single_gpu, multi_gpu.")
    if config.runtime.mixed_precision not in {"none", "bf16", "fp16"}:
        raise ValueError("runtime.mixed_precision must be one of: none, bf16, fp16.")
    if config.runtime.cpu_threads < 0:
        raise ValueError("runtime.cpu_threads must be >= 0.")
    if config.runtime.command_mode not in {"module", "executable"}:
        raise ValueError("runtime.command_mode must be one of: module, executable.")
    if config.runtime.nproc_per_node is not None and config.runtime.nproc_per_node <= 0:
        raise ValueError("runtime.nproc_per_node must be > 0 when provided.")
    if config.runtime.heartbeat_interval_seconds <= 0:
        raise ValueError("runtime.heartbeat_interval_seconds must be > 0.")
    if config.runtime.python_executable is not None:
        python_path = Path(config.runtime.python_executable).expanduser()
        if not python_path.exists():
            raise ValueError(
                f"runtime.python_executable does not exist: {config.runtime.python_executable}"
            )
    if config.runtime.target == "cpu" and config.runtime.cuda_visible_devices:
        raise ValueError("runtime.cuda_visible_devices must be empty when runtime.target='cpu'.")

    if not config.tuning.finetuning_type:
        raise ValueError("tuning.finetuning_type must not be empty.")
    if config.tuning.lora_rank <= 0:
        raise ValueError("tuning.lora_rank must be > 0.")
    if config.tuning.lora_alpha <= 0:
        raise ValueError("tuning.lora_alpha must be > 0.")
    if config.tuning.lora_dropout < 0:
        raise ValueError("tuning.lora_dropout must be >= 0.")
    if config.tuning.quantization_bit is not None and config.tuning.quantization_bit <= 0:
        raise ValueError("tuning.quantization_bit must be > 0 when provided.")

    if config.training.cutoff_len <= 0:
        raise ValueError("training.cutoff_len must be > 0.")
    if config.training.per_device_train_batch_size <= 0:
        raise ValueError("training.per_device_train_batch_size must be > 0.")
    if config.training.gradient_accumulation_steps <= 0:
        raise ValueError("training.gradient_accumulation_steps must be > 0.")
    if config.training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be > 0.")
    if config.training.num_train_epochs <= 0:
        raise ValueError("training.num_train_epochs must be > 0.")
    if not 0 <= config.training.warmup_ratio <= 1:
        raise ValueError("training.warmup_ratio must be between 0 and 1.")
    if config.training.logging_steps <= 0:
        raise ValueError("training.logging_steps must be > 0.")
    if config.training.save_steps <= 0:
        raise ValueError("training.save_steps must be > 0.")
    if (
        config.training.save_total_limit is not None
        and config.training.save_total_limit <= 0
    ):
        raise ValueError("training.save_total_limit must be > 0 when provided.")
    if config.training.dataloader_num_workers < 0:
        raise ValueError("training.dataloader_num_workers must be >= 0.")
    if config.training.preprocessing_num_workers <= 0:
        raise ValueError("training.preprocessing_num_workers must be > 0.")

    if config.method == "dpo":
        if config.method_config.dpo.pref_beta <= 0:
            raise ValueError("method_config.dpo.pref_beta must be > 0.")
        if config.method_config.dpo.pref_loss not in {"sigmoid", "orpo", "simpo"}:
            raise ValueError(
                "method_config.dpo.pref_loss must be one of: sigmoid, orpo, simpo."
            )

    if config.method == "kto":
        if config.method_config.kto.pref_beta <= 0:
            raise ValueError("method_config.kto.pref_beta must be > 0.")
        if config.method_config.kto.kto_chosen_weight <= 0:
            raise ValueError("method_config.kto.kto_chosen_weight must be > 0.")
        if config.method_config.kto.kto_rejected_weight <= 0:
            raise ValueError("method_config.kto.kto_rejected_weight must be > 0.")
        if config.method_config.kto.ref_model_quantization_bit not in {None, 4, 8}:
            raise ValueError("method_config.kto.ref_model_quantization_bit must be 4, 8, or null.")

    if config.method == "ppo":
        if config.method_config.ppo.reward_service:
            raise ValueError(
                "method_config.ppo.reward_service is not supported in Phase 1. "
                "Use method_config.ppo.reward_model_path with a LLaMA-Factory-compatible reward model."
            )
        if not config.method_config.ppo.reward_model_path:
            raise ValueError(
                "PPO requires method_config.ppo.reward_model_path."
            )
        if config.method_config.ppo.reward_model_type not in {"lora", "full", "api"}:
            raise ValueError("method_config.ppo.reward_model_type must be one of: lora, full, api.")
        if config.method_config.ppo.reward_model_type == "lora" and config.tuning.finetuning_type != "lora":
            raise ValueError(
                "method_config.ppo.reward_model_type='lora' requires tuning.finetuning_type='lora'."
            )
        if config.method_config.ppo.reward_model_quantization_bit not in {None, 4, 8}:
            raise ValueError(
                "method_config.ppo.reward_model_quantization_bit must be 4, 8, or null."
            )
        if config.method_config.ppo.ref_model_quantization_bit not in {None, 4, 8}:
            raise ValueError("method_config.ppo.ref_model_quantization_bit must be 4, 8, or null.")
        if config.method_config.ppo.ppo_buffer_size <= 0:
            raise ValueError("method_config.ppo.ppo_buffer_size must be > 0.")
        if config.method_config.ppo.ppo_epochs <= 0:
            raise ValueError("method_config.ppo.ppo_epochs must be > 0.")
        if config.method_config.ppo.ppo_target <= 0:
            raise ValueError("method_config.ppo.ppo_target must be > 0.")
        if config.method_config.ppo.max_new_tokens <= 0:
            raise ValueError("method_config.ppo.max_new_tokens must be > 0.")
        if config.method_config.ppo.temperature <= 0:
            raise ValueError("method_config.ppo.temperature must be > 0.")
        if config.method_config.ppo.top_p <= 0 or config.method_config.ppo.top_p > 1:
            raise ValueError("method_config.ppo.top_p must be in (0, 1].")
        if config.method_config.ppo.top_k < 0:
            raise ValueError("method_config.ppo.top_k must be >= 0.")

    if config.output.export_merged_model and config.tuning.finetuning_type != "lora":
        raise ValueError("Merged export is only implemented for tuning.finetuning_type='lora'.")
    if config.method == "rm" and config.output.export_merged_model:
        raise ValueError(
            "Reward-model training produces a reward adapter/value head. "
            "Set output.export_merged_model=false for alignment.method='rm'."
        )
    if config.tuning.finetuning_type == "lora" and not config.output.save_adapter_checkpoint:
        raise ValueError("LoRA runs must keep adapter checkpoints in this implementation.")


def _build_dpo_config(raw_value: Any) -> AlignmentDPOConfig:
    """Build the DPO-specific nested config with defaults."""

    raw = _optional_mapping(raw_value, "method_config.dpo")
    return AlignmentDPOConfig(
        pref_beta=float(raw.get("pref_beta", 0.1)),
        pref_loss=str(raw.get("pref_loss", "sigmoid")),
    )


def _build_kto_config(raw_value: Any) -> AlignmentKTOConfig:
    """Build the KTO-specific nested config with LLaMA-Factory defaults."""

    raw = _optional_mapping(raw_value, "method_config.kto")
    return AlignmentKTOConfig(
        pref_beta=float(raw.get("pref_beta", 0.1)),
        kto_chosen_weight=float(raw.get("kto_chosen_weight", 1.0)),
        kto_rejected_weight=float(raw.get("kto_rejected_weight", 1.0)),
        ref_model=str(v) if (v := raw.get("ref_model")) is not None else None,
        ref_model_adapters=str(v) if (v := raw.get("ref_model_adapters")) is not None else None,
        ref_model_quantization_bit=int(v)
        if (v := raw.get("ref_model_quantization_bit")) is not None
        else None,
    )


def _build_ppo_config(raw_value: Any) -> AlignmentPPOConfig:
    """Build the PPO-specific nested config with LLaMA-Factory defaults."""

    raw = _optional_mapping(raw_value, "method_config.ppo")
    reward_service = raw.get("reward_service")
    if reward_service is not None and not isinstance(reward_service, dict):
        raise ValueError("method_config.ppo.reward_service must be a mapping when provided.")
    return AlignmentPPOConfig(
        reward_model_path=str(v)
        if (v := raw.get("reward_model_path", raw.get("reward_model"))) is not None
        else None,
        reward_service=reward_service,
        reward_model_type=str(raw.get("reward_model_type", "lora")),
        reward_model_adapters=str(v)
        if (v := raw.get("reward_model_adapters")) is not None
        else None,
        reward_model_quantization_bit=int(v)
        if (v := raw.get("reward_model_quantization_bit")) is not None
        else None,
        ref_model=str(v) if (v := raw.get("ref_model")) is not None else None,
        ref_model_adapters=str(v) if (v := raw.get("ref_model_adapters")) is not None else None,
        ref_model_quantization_bit=int(v)
        if (v := raw.get("ref_model_quantization_bit")) is not None
        else None,
        ppo_buffer_size=int(raw.get("ppo_buffer_size", 1)),
        ppo_epochs=int(raw.get("ppo_epochs", 4)),
        ppo_score_norm=bool(raw.get("ppo_score_norm", False)),
        ppo_target=float(raw.get("ppo_target", 6.0)),
        ppo_whiten_rewards=bool(raw.get("ppo_whiten_rewards", False)),
        max_new_tokens=int(raw.get("max_new_tokens", 256)),
        temperature=float(raw.get("temperature", 0.95)),
        top_p=float(raw.get("top_p", 0.9)),
        top_k=int(raw.get("top_k", 50)),
    )


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    """Require that a nested config section is a mapping."""

    if value is None or not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _optional_mapping(value: Any, label: str) -> dict[str, Any]:
    """Return a nested mapping or an empty dict when omitted."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping when provided.")
    return value


def _resolve_path(base_dir: Path, raw_path: Any) -> Path:
    """Resolve a required path field relative to the config file."""

    if raw_path in (None, ""):
        raise ValueError("A required path field is missing or empty.")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path
