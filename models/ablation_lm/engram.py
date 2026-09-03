"""Causal token-ID Engram for the ablation-first language model.

The module retrieves fixed-width rows from independent hashed suffix n-gram
tables, filters them with the current (already causal) hidden state, and adds a
small residual.  Addressing never reads a token to the right of the position
being updated.  Supplying ``document_ids`` invalidates suffixes that cross a
packed-document boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .optim import ParameterRole, tag_optimizer_role


@dataclass(frozen=True)
class TokenEngramConfig:
    """Shape and initialization contract for :class:`CausalTokenEngram`."""

    hidden_dim: int
    num_slots: int = 131_071
    ngram_orders: tuple[int, ...] = (2, 3)
    num_hash_heads: int = 4
    head_dim: int = 8
    initial_scale: float = 1.0e-3
    max_scale: float = 0.1
    seed: int = 20_260_826
    enabled: bool = True


class CausalTokenEngram(nn.Module):
    """Regular token-ID Engram with causal, document-local suffix keys.

    Parameters are intentionally separated by hash head: every ``(order,
    head)`` pair owns an independent embedding table and hashing seed.  The
    forward method returns ``(hidden_states_with_residual, audit)``.  Passing
    ``enabled=False`` takes an exact structural bypass and returns the input
    tensor itself without evaluating the memory arm.
    """

    def __init__(self, config: TokenEngramConfig) -> None:
        super().__init__()
        self._validate_config(config)
        self.config = config
        self.hidden_dim = int(config.hidden_dim)
        self.num_slots = int(config.num_slots)
        self.ngram_orders = tuple(int(order) for order in config.ngram_orders)
        self.num_hash_heads = int(config.num_hash_heads)
        self.head_dim = int(config.head_dim)
        self.memory_dim = (
            len(self.ngram_orders) * self.num_hash_heads * self.head_dim
        )

        self.tables = nn.ModuleDict(
            {
                self._table_name(order, head): nn.Embedding(
                    self.num_slots, self.head_dim
                )
                for order in self.ngram_orders
                for head in range(self.num_hash_heads)
            }
        )
        self.memory_norm = nn.RMSNorm(
            self.memory_dim, eps=1.0e-6, elementwise_affine=False
        )
        # Catch #37: addressing is performed in the memory space itself.  The
        # retrieved row is the key; there is no learned key projection that
        # could silently move the dot product back to hidden width.
        self.query_proj = nn.Linear(self.hidden_dim, self.memory_dim, bias=False)
        self.value_proj = nn.Linear(self.memory_dim, self.hidden_dim, bias=False)
        self.query_norm = nn.RMSNorm(
            self.memory_dim, eps=1.0e-6, elementwise_affine=True
        )
        # EG-1 / Catch #39 keeps unit-initialized, trainable gains on both
        # gate operands so the row-as-key form can learn its temperature.  A
        # distinct gate-only norm leaves the value path exactly unchanged.
        self.key_norm = nn.RMSNorm(
            self.memory_dim, eps=1.0e-6, elementwise_affine=True
        )
        self.gate_bias = nn.Parameter(torch.zeros(()))

        scale_ratio = float(config.initial_scale) / float(config.max_scale)
        self.raw_residual_scale = nn.Parameter(
            torch.tensor(math.atanh(scale_ratio), dtype=torch.float32)
        )
        for table in self.tables.values():
            tag_optimizer_role(table, "weight", ParameterRole.ENGRAM)
        for projection in (self.query_proj, self.value_proj):
            tag_optimizer_role(projection, "weight", ParameterRole.ENGRAM)
        tag_optimizer_role(self, "gate_bias", ParameterRole.GATE)
        tag_optimizer_role(self, "raw_residual_scale", ParameterRole.GATE)
        self._reset_parameters()

    @staticmethod
    def _validate_config(config: TokenEngramConfig) -> None:
        orders = tuple(config.ngram_orders)
        if type(config.hidden_dim) is not int or config.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if type(config.num_slots) is not int or config.num_slots < 2:
            raise ValueError("num_slots must be at least two")
        if not orders or any(type(order) is not int or order < 1 for order in orders):
            raise ValueError("ngram_orders must contain positive integers")
        if len(set(orders)) != len(orders):
            raise ValueError("ngram_orders must be unique")
        if (
            type(config.num_hash_heads) is not int
            or config.num_hash_heads < 1
            or type(config.head_dim) is not int
            or config.head_dim < 1
        ):
            raise ValueError("num_hash_heads and head_dim must be positive")
        initial_scale = float(config.initial_scale)
        max_scale = float(config.max_scale)
        if (
            not math.isfinite(initial_scale)
            or not math.isfinite(max_scale)
            or not 0.0 < abs(initial_scale) < max_scale
        ):
            raise ValueError(
                "scales must be finite with 0 < abs(initial_scale) < max_scale"
            )
        if type(config.seed) is not int:
            raise ValueError("seed must be an exact integer")
        if type(config.enabled) is not bool:
            raise ValueError("enabled must be boolean")

    @staticmethod
    def _table_name(order: int, head: int) -> str:
        return f"n{order}_h{head}"

    def _reset_parameters(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(int(self.config.seed))
        for table in self.tables.values():
            nn.init.normal_(table.weight, mean=0.0, std=0.02, generator=generator)
        for projection in (self.query_proj, self.value_proj):
            nn.init.xavier_uniform_(projection.weight, generator=generator)

    def residual_scale(self) -> torch.Tensor:
        """Return the bounded scalar multiplying the active memory residual."""

        return float(self.config.max_scale) * torch.tanh(self.raw_residual_scale)

    def _suffix_indices(
        self,
        token_ids: torch.Tensor,
        document_ids: torch.Tensor | None,
        *,
        order: int,
        head: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full-length suffix indices and their document-local validity."""

        batch, length = token_ids.shape
        indices = torch.full(
            (batch, length), -1, dtype=torch.long, device=token_ids.device
        )
        valid = torch.zeros((batch, length), dtype=torch.bool, device=token_ids.device)
        if length < order:
            return indices, valid

        token_windows = token_ids.unfold(1, order, 1).long()
        window_count = token_windows.shape[1]
        window_valid = torch.ones(
            (batch, window_count), dtype=torch.bool, device=token_ids.device
        )
        if document_ids is not None:
            document_windows = document_ids.unfold(1, order, 1).long()
            final_document = document_windows[..., -1:]
            window_valid = document_windows.eq(final_document).all(dim=-1)
            window_valid &= final_document.squeeze(-1).ge(0)

        # Reduce at every step so the hash remains deterministic without
        # relying on signed-int overflow behavior.
        hash_seed = (
            int(self.config.seed)
            + 1_000_003 * order
            + 104_729 * (head + 1)
        ) % self.num_slots
        multiplier = 1_000_003 + 2 * (head + len(self.ngram_orders) * order)
        state = torch.full(
            (batch, window_count),
            hash_seed,
            dtype=torch.long,
            device=token_ids.device,
        )
        for offset in range(order):
            state = torch.remainder(
                state * multiplier + token_windows[..., offset] + 1 + 97 * offset,
                self.num_slots,
            )

        end_positions = slice(order - 1, None)
        indices[:, end_positions] = torch.where(
            window_valid, state, torch.full_like(state, -1)
        )
        valid[:, end_positions] = window_valid
        return indices, valid

    @staticmethod
    def _validate_inputs(
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        document_ids: torch.Tensor | None,
    ) -> None:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must use an integer dtype")
        if hidden_states.shape[:2] != token_ids.shape:
            raise ValueError("hidden_states and token_ids must share batch and sequence")
        if document_ids is not None:
            if document_ids.shape != token_ids.shape:
                raise ValueError("document_ids must have the same shape as token_ids")
            if document_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("document_ids must use an integer dtype")

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        document_ids: torch.Tensor | None = None,
        enabled: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Read causal memory and add its gated, bounded residual.

        ``document_ids`` may use any nonnegative integer labels.  Negative IDs
        are treated as invalid/padding and cannot produce Engram keys.
        """

        self._validate_inputs(hidden_states, token_ids, document_ids)
        if hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"expected hidden dimension {self.hidden_dim}, "
                f"got {hidden_states.shape[-1]}"
            )

        is_enabled = self.config.enabled if enabled is None else bool(enabled)
        if not is_enabled:
            return hidden_states, {
                "enabled": torch.tensor(False, device=hidden_states.device)
            }

        retrieved: list[torch.Tensor] = []
        audit: dict[str, torch.Tensor] = {
            "enabled": torch.tensor(True, device=hidden_states.device)
        }
        validity: list[torch.Tensor] = []
        for order in self.ngram_orders:
            order_valid: torch.Tensor | None = None
            for head in range(self.num_hash_heads):
                name = self._table_name(order, head)
                indices, valid = self._suffix_indices(
                    token_ids, document_ids, order=order, head=head
                )
                safe_indices = indices.masked_fill(~valid, 0)
                rows = self.tables[name](safe_indices)
                rows = rows * valid.unsqueeze(-1).to(dtype=rows.dtype)
                retrieved.append(rows)
                audit[f"indices_{name}"] = indices
                order_valid = valid if order_valid is None else order_valid | valid
            assert order_valid is not None
            audit[f"valid_n{order}"] = order_valid
            validity.append(order_valid)

        memory = torch.cat(retrieved, dim=-1)
        normalized_memory = self.memory_norm(memory)
        query = self.query_norm(self.query_proj(hidden_states))
        value = self.value_proj(normalized_memory)
        key = self.key_norm(memory)
        gate = torch.sigmoid(
            (query.float() * key.float()).sum(dim=-1, keepdim=True)
            / math.sqrt(self.memory_dim)
            + self.gate_bias.float()
        )
        has_memory = torch.stack(validity, dim=0).any(dim=0).unsqueeze(-1)
        gate = gate * has_memory.to(dtype=gate.dtype)
        delta = gate * value
        scale = self.residual_scale().to(dtype=hidden_states.dtype)
        output = hidden_states + scale * delta.to(dtype=hidden_states.dtype)

        audit["gate"] = gate
        audit["has_memory"] = has_memory
        audit["residual_scale"] = scale
        return output, audit
