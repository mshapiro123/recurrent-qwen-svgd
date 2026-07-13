from __future__ import annotations

import pytest

from training.train_dense_full import checkpoint_name, validate_config


def test_validate_config_locks_full_fp32_adamw_contract() -> None:
    cfg = validate_config(
        {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "abc",
            "optimizer": "adamw",
            "parameter_dtype": "float32",
            "compute_dtype": "bfloat16",
            "max_steps": 4000,
            "gradient_accumulation_steps": 8,
        }
    )

    assert cfg["optimizer"] == "adamw"
    assert cfg["parameter_dtype"] == "float32"
    assert cfg["gradient_accumulation_steps"] == 8


def test_validate_config_rejects_non_fp32_parameters() -> None:
    with pytest.raises(ValueError, match="parameter_dtype=float32"):
        validate_config(
            {
                "model_name": "model",
                "revision": "abc",
                "optimizer": "adamw",
                "parameter_dtype": "bfloat16",
                "compute_dtype": "bfloat16",
                "max_steps": 1,
                "gradient_accumulation_steps": 1,
            }
        )


def test_checkpoint_name_is_step_stable() -> None:
    assert checkpoint_name(4000) == "dense_full_step_4000"
