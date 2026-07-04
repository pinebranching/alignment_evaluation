"""Define the typed configuration and result objects used by the alignment module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AlignmentModelConfig:
    """Store model-loading fields that are shared across alignment methods."""

    base_model_name_or_path: str
    template: str
    tokenizer_name_or_path: str | None = None
    trust_remote_code: bool = False
    disable_gradient_checkpointing: bool | None = None


@dataclass(slots=True)
class AlignmentTuningConfig:
    """Store tuning-strategy settings that should stay stable across backends."""

    finetuning_type: str
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target: str | list[str] = "all"
    quantization_bit: int | None = None


@dataclass(slots=True)
class AlignmentDatasetConfig:
    """Store dataset paths and field mappings for a single alignment run."""

    path: Path
    format: str
    schema: str
    prompt_field: str
    response_field: str | None = None
    chosen_field: str | None = None
    rejected_field: str | None = None
    label_field: str | None = None
    system_field: str | None = None
    max_samples: int | None = None


@dataclass(slots=True)
class AlignmentRuntimeConfig:
    """Store runtime-placement settings for CPU and future GPU execution."""

    target: str
    device: str
    mixed_precision: str
    cpu_threads: int = 0
    cuda_visible_devices: list[str] = field(default_factory=list)
    command_mode: str = "module"
    python_executable: str | None = None
    llamafactory_src_dir: str | None = None
    force_torchrun: bool = False
    nproc_per_node: int | None = None
    extra_env: dict[str, str] = field(default_factory=dict)
    heartbeat_interval_seconds: int = 30


@dataclass(slots=True)
class AlignmentTrainingConfig:
    """Store backend-agnostic training hyperparameters for alignment."""

    cutoff_len: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    lr_scheduler_type: str
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    save_total_limit: int | None = None
    dataloader_num_workers: int = 0
    preprocessing_num_workers: int = 0


@dataclass(slots=True)
class AlignmentDPOConfig:
    """Store DPO-specific settings."""

    pref_beta: float = 0.1
    pref_loss: str = "sigmoid"


@dataclass(slots=True)
class AlignmentKTOConfig:
    """Store KTO-specific settings."""

    pref_beta: float = 0.1
    kto_chosen_weight: float = 1.0
    kto_rejected_weight: float = 1.0
    ref_model: str | None = None
    ref_model_adapters: str | None = None
    ref_model_quantization_bit: int | None = None


@dataclass(slots=True)
class AlignmentPPOConfig:
    """Store PPO fields passed through to LLaMA-Factory."""

    reward_model_path: str | None = None
    reward_service: dict[str, Any] | None = None
    reward_model_type: str = "lora"
    reward_model_adapters: str | None = None
    reward_model_quantization_bit: int | None = None
    ref_model: str | None = None
    ref_model_adapters: str | None = None
    ref_model_quantization_bit: int | None = None
    ppo_buffer_size: int = 1
    ppo_epochs: int = 4
    ppo_score_norm: bool = False
    ppo_target: float = 6.0
    ppo_whiten_rewards: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.95
    top_p: float = 0.9
    top_k: int = 50


@dataclass(slots=True)
class AlignmentMethodConfig:
    """Store per-method nested configuration sections."""

    sft: dict[str, Any] = field(default_factory=dict)
    dpo: AlignmentDPOConfig = field(default_factory=AlignmentDPOConfig)
    kto: AlignmentKTOConfig = field(default_factory=AlignmentKTOConfig)
    ppo: AlignmentPPOConfig = field(default_factory=AlignmentPPOConfig)


@dataclass(slots=True)
class AlignmentOutputConfig:
    """Store run-output policies and artifact preferences."""

    root_dir: Path
    save_adapter_checkpoint: bool
    export_merged_model: bool
    export_device: str
    overwrite_output_dir: bool
    save_only_model: bool
    report_to: str


@dataclass(slots=True)
class AlignmentConfig:
    """Store the full high-level configuration for one alignment run."""

    backend: str
    method: str
    run_name: str
    model: AlignmentModelConfig
    tuning: AlignmentTuningConfig
    dataset: AlignmentDatasetConfig
    runtime: AlignmentRuntimeConfig
    training: AlignmentTrainingConfig
    method_config: AlignmentMethodConfig
    output: AlignmentOutputConfig
    config_path: Path
    raw_alignment_config: dict[str, Any]


@dataclass(slots=True)
class PreparedDataset:
    """Describe the local dataset bundle that LLaMA-Factory will consume."""

    dataset_name: str
    dataset_dir: Path
    dataset_info_path: Path
    materialized_data_path: Path


@dataclass(slots=True)
class CommandResult:
    """Capture the outcome of a backend command."""

    command: list[str]
    log_path: Path
    return_code: int
    duration_seconds: float
    launcher_python: str


@dataclass(slots=True)
class AlignmentResult:
    """Return the important artifact and status fields for downstream callers."""

    run_dir: Path
    adapter_path: Path
    exported_model_path: Path | None
    train_config_path: Path
    export_config_path: Path | None
    manifest_path: Path
    method: str
    backend: str
