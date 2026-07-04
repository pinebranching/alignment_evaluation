"""Centralize supported alignment methods and their validation requirements. Can be extended."""

from __future__ import annotations


METHOD_REGISTRY: dict[str, dict[str, object]] = {
    "sft": {
        "implemented": True,
        "dataset_schema": "prompt_response",
        "requires_reward_model": False,
    },
    "dpo": {
        "implemented": True,
        "dataset_schema": "prompt_chosen_rejected",
        "requires_reward_model": False,
    },
    "rm": {
        "implemented": True,
        "dataset_schema": "prompt_chosen_rejected",
        "requires_reward_model": False,
    },
    "kto": {
        "implemented": True,
        "dataset_schema": "prompt_response_label",
        "requires_reward_model": False,
    },
    "ppo": {
        "implemented": True,
        "dataset_schema": "prompt_response",
        "requires_reward_model": True,
    },
}


def get_method_spec(method: str) -> dict[str, object]:
    """Return the registry entry for a method or raise a clear error."""

    normalized = method.strip().lower()
    if normalized not in METHOD_REGISTRY:
        raise ValueError(f"Unsupported alignment method: {method}")
    return METHOD_REGISTRY[normalized]
