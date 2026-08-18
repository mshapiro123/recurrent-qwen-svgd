"""Minimal zero-init LoRA adapters for recurrent-block experiments."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


DEFAULT_QWEN_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

ATTENTION_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    """Wrap an existing Linear layer with an identity-preserving LoRA branch."""

    is_lora_adapter = True

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        adapter_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)

        for param in self.base.parameters():
            param.requires_grad_(False)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
            nn.init.zeros_(self.lora_b.weight)
        self.lora_a.to(device=base.weight.device, dtype=adapter_dtype)
        self.lora_b.to(device=base.weight.device, dtype=adapter_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        adapter_dtype = self.lora_a.weight.dtype
        adapter_in = self.dropout(x).to(dtype=adapter_dtype)
        adapter_out = self.lora_b(self.lora_a(adapter_in)) * self.scaling
        return base_out + adapter_out.to(dtype=base_out.dtype)

    def lora_parameters(self) -> list[nn.Parameter]:
        return list(self.lora_a.parameters()) + list(self.lora_b.parameters())

    def set_adapter_dtype(self, dtype: torch.dtype) -> None:
        device = self.base.weight.device
        self.lora_a.to(device=device, dtype=dtype)
        self.lora_b.to(device=device, dtype=dtype)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold the LoRA branch into ``base`` and return the plain Linear layer."""

        delta = (self.lora_b.weight.float() @ self.lora_a.weight.float()) * float(self.scaling)
        self.base.weight.add_(delta.to(device=self.base.weight.device, dtype=self.base.weight.dtype))
        for param in self.base.parameters():
            param.requires_grad_(False)
        return self.base


class LoopScopedLoRALinear(LoRALinear):
    """LoRA branch that is structurally disabled on the first loop pass.

    The adapter state may change during training, but ``loop_index == 0`` always
    returns the frozen base projection exactly. This is stronger than relying on
    zero initialization and keeps the T=1 path invariant at every checkpoint.
    """

    is_loop_scoped_adapter = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loop_index = 0

    def set_loop_index(self, loop_index: int) -> None:
        if int(loop_index) < 0:
            raise ValueError("loop index must be nonnegative")
        self.loop_index = int(loop_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self.loop_index == 0:
            return base_out
        adapter_dtype = self.lora_a.weight.dtype
        adapter_in = self.dropout(x).to(dtype=adapter_dtype)
        adapter_out = self.lora_b(self.lora_a(adapter_in)) * self.scaling
        return base_out + adapter_out.to(dtype=base_out.dtype)


def apply_lora_to_recurrent_block(
    wrapper: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    adapter_dtype: torch.dtype = torch.float32,
    target_module_names: Iterable[str] = DEFAULT_QWEN_LORA_TARGETS,
) -> int:
    """Replace target Linear modules inside the wrapper's recurrent block.

    Returns the number of modules wrapped. The LoRA branch is zero-initialized,
    so applying it does not change logits before training.
    """

    target_names = set(target_module_names)
    replaced = 0
    for layer_idx in range(wrapper.layer_split.prelude_end, wrapper.layer_split.recurrent_end):
        layer = wrapper.qwen.layers[layer_idx]
        replaced += _replace_lora_targets(layer, target_names, rank, alpha, dropout, adapter_dtype)
    return replaced


def apply_loop_scoped_lora_to_recurrent_block(
    wrapper: nn.Module,
    *,
    rank: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    adapter_dtype: torch.dtype = torch.float32,
    target_module_names: Iterable[str] = ATTENTION_LORA_TARGETS,
) -> int:
    """Install first-pass-inactive LoRA on recurrent attention projections."""

    target_names = set(target_module_names)
    replaced = 0
    for layer_idx in range(wrapper.layer_split.prelude_end, wrapper.layer_split.recurrent_end):
        layer = wrapper.qwen.layers[layer_idx]
        replaced += _replace_loop_scoped_lora_targets(
            layer, target_names, rank, alpha, dropout, adapter_dtype
        )
    return replaced


def set_loop_scoped_lora_index(module: nn.Module, loop_index: int) -> int:
    """Set the active recurrent pass on every loop-scoped adapter."""

    changed = 0
    for child in module.modules():
        if isinstance(child, LoopScopedLoRALinear):
            child.set_loop_index(loop_index)
            changed += 1
    return changed


def apply_lora_to_qwen_layers(
    model: nn.Module,
    *,
    start_layer: int = 0,
    end_layer: int | None = None,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    adapter_dtype: torch.dtype = torch.float32,
    target_module_names: Iterable[str] = DEFAULT_QWEN_LORA_TARGETS,
) -> int:
    """Replace target Linear modules in a Qwen-style dense model layer range.

    This is the dense-model control counterpart to
    :func:`apply_lora_to_recurrent_block`.  It keeps the same zero-init
    identity behavior while allowing a standard non-recurrent Qwen baseline to
    train on the same curriculum as the recurrent wrapper.
    """

    qwen = getattr(model, "model", model)
    if not hasattr(qwen, "layers"):
        raise TypeError("Expected a Qwen-style model with .model.layers or .layers")
    layers = qwen.layers
    end = len(layers) if end_layer is None else end_layer
    if not 0 <= start_layer < end <= len(layers):
        raise ValueError(f"Invalid LoRA layer range {start_layer}:{end} for {len(layers)} layers")
    target_names = set(target_module_names)
    replaced = 0
    for layer_idx in range(start_layer, end):
        replaced += _replace_lora_targets(layers[layer_idx], target_names, rank, alpha, dropout, adapter_dtype)
    return replaced


def mark_only_lora_trainable(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            for param in child.lora_parameters():
                param.requires_grad_(True)
            for param in child.base.parameters():
                param.requires_grad_(False)


def set_lora_adapter_dtype(module: nn.Module, dtype: torch.dtype) -> None:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            child.set_adapter_dtype(dtype)


def merge_lora_adapters(module: nn.Module) -> int:
    """Replace every ``LoRALinear`` child with its merged base Linear layer."""

    merged = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, child_name, child.merge())
            merged += 1
            continue
        merged += merge_lora_adapters(child)
    return merged


def _replace_lora_targets(
    module: nn.Module,
    target_names: set[str],
    rank: int,
    alpha: int,
    dropout: float,
    adapter_dtype: torch.dtype,
) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if child_name in target_names and isinstance(child, nn.Linear):
            setattr(
                module,
                child_name,
                LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout, adapter_dtype=adapter_dtype),
            )
            replaced += 1
            continue
        replaced += _replace_lora_targets(child, target_names, rank, alpha, dropout, adapter_dtype)
    return replaced


def _replace_loop_scoped_lora_targets(
    module: nn.Module,
    target_names: set[str],
    rank: int,
    alpha: int,
    dropout: float,
    adapter_dtype: torch.dtype,
) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if child_name in target_names and isinstance(child, nn.Linear):
            setattr(
                module,
                child_name,
                LoopScopedLoRALinear(
                    child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    adapter_dtype=adapter_dtype,
                ),
            )
            replaced += 1
            continue
        replaced += _replace_loop_scoped_lora_targets(
            child, target_names, rank, alpha, dropout, adapter_dtype
        )
    return replaced
