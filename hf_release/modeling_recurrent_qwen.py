"""Self-contained forced-depth recurrent Qwen loader for the Paper One release.

This module deliberately contains no halting head or adaptive-depth path. Every
forward call uses an externally supplied ``max_loops`` and, optionally, an
externally supplied per-row ``loop_selection`` bounded by that maximum.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.hub import cached_file

from .configuration_recurrent_qwen import RecurrentQwenConfig


@dataclass
class RecurrentCausalLMOutput(CausalLMOutputWithPast):
    """Causal-LM output with optional loop-indexed logits."""

    loop_logits: Optional[torch.FloatTensor] = None
    selected_loop_counts: Optional[torch.LongTensor] = None


class IdentityGatedBridge(nn.Module):
    """Split re-entry bridge used by the frozen Paper One checkpoints."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.prelude_norm = nn.LayerNorm(hidden_size)
        self.prelude_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.state_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.bridge_gate = nn.Parameter(torch.tensor(1.0))
        with torch.no_grad():
            self.prelude_proj.weight.zero_()
            self.state_proj.weight.copy_(torch.eye(hidden_size))
            self.state_proj.bias.zero_()

    def forward(self, state: torch.Tensor, prelude: torch.Tensor) -> torch.Tensor:
        input_dtype = state.dtype
        work_dtype = self.state_proj.weight.dtype
        work = state.to(dtype=work_dtype)
        normalized_prelude = self.prelude_norm(prelude.to(dtype=work_dtype))
        translated = self.prelude_proj(normalized_prelude) + self.state_proj(work)
        gate = self.bridge_gate.to(device=state.device, dtype=work_dtype)
        return (work + gate * (translated - work)).to(dtype=input_dtype)


class LoRALinear(nn.Module):
    """Inference-only rank-decomposition wrapper matching the training keys."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: int) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(rank)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
            self.lora_b.weight.zero_()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        adapter = self.lora_b(self.lora_a(inputs.float())) * self.scaling
        return base_output + adapter.to(dtype=base_output.dtype)


def _replace_lora_targets(
    module: nn.Module,
    target_names: set[str],
    *,
    rank: int,
    alpha: int,
) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if child_name in target_names and isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
            replaced += 1
        else:
            replaced += _replace_lora_targets(child, target_names, rank=rank, alpha=alpha)
    return replaced


class RecurrentQwenForCausalLM(PreTrainedModel, GenerationMixin):
    """Qwen causal LM with a forced, weight-tied middle-block recurrence."""

    config_class = RecurrentQwenConfig
    base_model_prefix = "backbone"
    main_input_name = "input_ids"

    def __init__(self, config: RecurrentQwenConfig, backbone: nn.Module) -> None:
        super().__init__(config)
        self.backbone = backbone
        if not hasattr(backbone, "model") or not hasattr(backbone.model, "layers"):
            raise TypeError("Expected a Qwen-style causal LM with .model.layers")
        if int(config.recurrent_end) >= len(self.qwen.layers):
            raise ValueError("recurrent_end must leave at least one coda layer")
        hidden_size = int(getattr(backbone.config, "hidden_size"))
        base_parameter = next(backbone.parameters())
        self.bridge = IdentityGatedBridge(hidden_size).to(
            device=base_parameter.device,
            dtype=base_parameter.dtype,
        )
        self.lora_module_count = 0
        if config.checkpoint_kind == "lora_adapter":
            targets = set(config.lora_target_modules)
            for layer_index in range(config.prelude_end, config.recurrent_end):
                self.lora_module_count += _replace_lora_targets(
                    self.qwen.layers[layer_index],
                    targets,
                    rank=config.lora_rank,
                    alpha=config.lora_alpha,
                )
            if self.lora_module_count != 84:
                raise RuntimeError(f"Expected 84 recurrent LoRA modules, got {self.lora_module_count}")

    @property
    def qwen(self) -> nn.Module:
        return self.backbone.model

    @property
    def lm_head(self) -> nn.Module:
        return self.backbone.lm_head

    def get_input_embeddings(self) -> nn.Module:
        return self.qwen.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.qwen.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.backbone.lm_head = value

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *model_args: Any,
        config: RecurrentQwenConfig | None = None,
        **kwargs: Any,
    ) -> "RecurrentQwenForCausalLM":
        """Load the pinned base model, construct the surgery, then apply the delta."""

        if model_args:
            raise TypeError("Positional model arguments are not supported by this release loader")
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", "main")
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = bool(kwargs.pop("local_files_only", False))
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("_from_auto", None)
        kwargs.pop("_fast_init", None)
        kwargs.pop("state_dict", None)
        kwargs.pop("weights_only", None)
        kwargs.pop("adapter_kwargs", None)

        if config is None:
            config = RecurrentQwenConfig.from_pretrained(
                pretrained_model_name_or_path,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        base_keys = {
            "attn_implementation",
            "device_map",
            "dtype",
            "torch_dtype",
            "low_cpu_mem_usage",
            "max_memory",
            "offload_folder",
            "offload_state_dict",
        }
        base_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in base_keys}
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported loader keyword(s): {unknown}")
        base_kwargs.update(
            {
                "revision": config.base_model_revision,
                "token": token,
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
                "trust_remote_code": False,
            }
        )
        backbone = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            **{key: value for key, value in base_kwargs.items() if value is not None},
        )
        model = cls(config, backbone)

        source = Path(pretrained_model_name_or_path)
        if source.is_dir():
            delta_path = source / config.delta_filename
        else:
            resolved = cached_file(
                str(pretrained_model_name_or_path),
                config.delta_filename,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            if resolved is None:
                raise FileNotFoundError(config.delta_filename)
            delta_path = Path(resolved)
        delta = load_file(str(delta_path), device="cpu")
        current = model.state_dict()
        absent = sorted(set(delta) - set(current))
        mismatched = {
            key: {"delta": tuple(value.shape), "model": tuple(current[key].shape)}
            for key, value in delta.items()
            if key in current and value.shape != current[key].shape
        }
        if absent or mismatched:
            raise RuntimeError(f"Release delta is incompatible: absent={absent}, mismatched={mismatched}")
        model.load_state_dict(delta, strict=False)
        model._release_load_receipt = {
            "delta_file": str(delta_path),
            "tensor_count": len(delta),
            "total_parameters": sum(int(tensor.numel()) for tensor in delta.values()),
        }
        model.eval()
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        max_loops: int = 1,
        loop_selection: int | torch.LongTensor | None = None,
        return_loop_logits: bool = False,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> RecurrentCausalLMOutput | tuple[Any, ...]:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported forward keyword(s): {unknown}")
        if int(max_loops) < 1:
            raise ValueError("max_loops must be at least 1")
        if use_cache:
            raise ValueError("KV caching is disabled for the forced-depth recurrent release")
        if output_attentions or output_hidden_states:
            raise ValueError("Attention/hidden-state collection is not exposed by the release loader")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify input_ids or inputs_embeds, not both")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.qwen.embed_tokens(input_ids)

        batch_size, sequence_length = inputs_embeds.shape[:2]
        device = inputs_embeds.device
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)
        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=device).unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = position_ids.to(device=device)
        cache_position = torch.arange(sequence_length, device=device)
        causal_mask = self._causal_mask(attention_mask, inputs_embeds, cache_position)
        position_embeddings = self._rotary_embeddings(inputs_embeds, position_ids)

        hidden = self._run_layers(
            0,
            self.config.prelude_end,
            inputs_embeds,
            causal_mask,
            position_ids,
            cache_position,
            position_embeddings,
        )
        prelude = hidden
        recurrent_state = hidden
        logits_by_loop: list[torch.Tensor] = []
        for loop_index in range(int(max_loops)):
            loop_input = recurrent_state if loop_index == 0 else self.bridge(recurrent_state, prelude)
            recurrent_state = self._run_layers(
                self.config.prelude_end,
                self.config.recurrent_end,
                loop_input,
                causal_mask,
                position_ids,
                cache_position,
                position_embeddings,
            )
            coda = self._run_layers(
                self.config.recurrent_end,
                len(self.qwen.layers),
                recurrent_state,
                causal_mask,
                position_ids,
                cache_position,
                position_embeddings,
            )
            normed = self.qwen.norm(coda)
            logits_by_loop.append(self.lm_head(self._slice_for_logits(normed, logits_to_keep)))

        loop_logits = torch.stack(logits_by_loop, dim=1)
        selected_counts = self._resolve_loop_selection(loop_selection, batch_size, int(max_loops), device)
        batch_indices = torch.arange(batch_size, device=device)
        logits = loop_logits[batch_indices, selected_counts - 1]
        loss = None
        if labels is not None:
            if not (isinstance(logits_to_keep, int) and logits_to_keep == 0):
                raise ValueError("labels require logits_to_keep=0")
            shifted_logits = logits[:, :-1, :].contiguous().float()
            shifted_labels = labels[:, 1:].contiguous().to(device=device)
            loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_labels.view(-1),
                ignore_index=-100,
            )

        output = RecurrentCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            loop_logits=loop_logits if return_loop_logits else None,
            selected_loop_counts=selected_counts,
        )
        return output if return_dict is not False else output.to_tuple()

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_loops": int(kwargs.get("max_loops", 1)),
            "loop_selection": kwargs.get("loop_selection"),
            "use_cache": False,
        }

    @staticmethod
    def _resolve_loop_selection(
        selection: int | torch.LongTensor | None,
        batch_size: int,
        max_loops: int,
        device: torch.device,
    ) -> torch.LongTensor:
        if selection is None:
            counts = torch.full((batch_size,), max_loops, dtype=torch.long, device=device)
        elif isinstance(selection, int):
            counts = torch.full((batch_size,), int(selection), dtype=torch.long, device=device)
        else:
            counts = selection.to(device=device, dtype=torch.long).reshape(-1)
            if counts.numel() != batch_size:
                raise ValueError("loop_selection tensor must contain one value per batch row")
        if bool(((counts < 1) | (counts > max_loops)).any()):
            raise ValueError("loop_selection values must lie in [1, max_loops]")
        return counts

    def _causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor | None:
        update = getattr(self.qwen, "_update_causal_mask", None)
        if update is not None:
            return self._call_supported(
                update,
                {
                    "attention_mask": attention_mask,
                    "input_tensor": inputs_embeds,
                    "inputs_embeds": inputs_embeds,
                    "cache_position": cache_position,
                    "past_key_values": None,
                    "output_attentions": False,
                },
            )
        batch_size, sequence_length = inputs_embeds.shape[:2]
        minimum = torch.finfo(inputs_embeds.dtype).min
        causal = torch.full(
            (sequence_length, sequence_length),
            minimum,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        ).triu(diagonal=1)
        causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal = causal.masked_fill(attention_mask[:, None, None, :].eq(0), minimum)
        return causal

    def _rotary_embeddings(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Any:
        rotary = getattr(self.qwen, "rotary_emb", None)
        if rotary is None:
            return None
        try:
            return rotary(hidden_states, position_ids)
        except TypeError:
            return None

    def _run_layers(
        self,
        start: int,
        end: int,
        hidden_states: torch.Tensor,
        causal_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        position_embeddings: Any,
    ) -> torch.Tensor:
        for layer in self.qwen.layers[start:end]:
            parameters = inspect.signature(layer.forward).parameters
            cache_key = "past_key_values" if "past_key_values" in parameters else "past_key_value"
            outputs = layer(
                hidden_states,
                **self._filter_supported(
                    layer.forward,
                    {
                        "attention_mask": causal_mask,
                        "position_ids": position_ids,
                        cache_key: None,
                        "output_attentions": False,
                        "use_cache": False,
                        "cache_position": cache_position,
                        "position_embeddings": position_embeddings,
                    },
                ),
            )
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        return hidden_states

    @staticmethod
    def _filter_supported(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        parameters = inspect.signature(function).parameters
        accepts_kwargs = any(value.kind == inspect.Parameter.VAR_KEYWORD for value in parameters.values())
        if accepts_kwargs:
            return {key: value for key, value in kwargs.items() if value is not None}
        return {
            key: value
            for key, value in kwargs.items()
            if key in parameters and (value is not None or key in {"use_cache", "output_attentions"})
        }

    @classmethod
    def _call_supported(cls, function: Any, kwargs: dict[str, Any]) -> Any:
        return function(**cls._filter_supported(function, kwargs))

    @staticmethod
    def _slice_for_logits(hidden_states: torch.Tensor, logits_to_keep: int | torch.Tensor) -> torch.Tensor:
        if isinstance(logits_to_keep, int):
            return hidden_states if logits_to_keep == 0 else hidden_states[:, -logits_to_keep:, :]
        return hidden_states[:, logits_to_keep, :]
