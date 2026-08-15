"""Hugging Face configuration for the forced-depth recurrent Qwen release."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedConfig


class RecurrentQwenConfig(PreTrainedConfig):
    """Configuration for a Qwen base plus a recurrent-depth delta."""

    model_type = "recurrent_qwen"

    def __init__(
        self,
        *,
        base_model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        base_model_revision: str = "main",
        prelude_end: int = 6,
        recurrent_end: int = 18,
        checkpoint_kind: str = "full_block_delta",
        delta_filename: str = "recurrent_delta.safetensors",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not 0 < int(prelude_end) < int(recurrent_end):
            raise ValueError("Expected 0 < prelude_end < recurrent_end")
        if checkpoint_kind not in {"full_block_delta", "lora_adapter"}:
            raise ValueError("checkpoint_kind must be full_block_delta or lora_adapter")
        self.base_model_name_or_path = str(base_model_name_or_path)
        self.base_model_revision = str(base_model_revision)
        self.prelude_end = int(prelude_end)
        self.recurrent_end = int(recurrent_end)
        self.checkpoint_kind = str(checkpoint_kind)
        self.delta_filename = str(delta_filename)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha)
        self.lora_target_modules = list(
            lora_target_modules
            or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.use_cache = False

