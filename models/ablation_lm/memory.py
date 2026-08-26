"""Leakage-safe first long-term-memory substrate: a frozen read-only store."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .layers import RMSNorm
from .optim import ParameterRole, tag_optimizer_role


class ReadOnlyLatentMemory(nn.Module):
    """Frozen content-addressed records with trainable query/read adapters.

    Keys, values, and provenance IDs are buffers rather than parameters. A
    query-side ``record_ids`` tensor excludes every record with matching
    provenance, implementing the mechanical part of leave-one-record-out
    retrieval. Corpus construction and answer-exclusion still require a
    separately receipted data process.
    """

    def __init__(
        self,
        d_model: int,
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        provenance_ids: torch.Tensor,
        layer_scale: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        if keys.ndim != 2 or values.ndim != 2 or keys.shape != values.shape:
            raise ValueError("keys and values must share [records, memory_width]")
        if keys.shape[0] < 1 or keys.shape[1] < 1:
            raise ValueError("the read-only store cannot be empty")
        if provenance_ids.ndim != 1 or provenance_ids.shape[0] != keys.shape[0]:
            raise ValueError("provenance_ids must contain one ID per memory record")
        if provenance_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("provenance_ids must use an integer dtype")
        if not bool(torch.isfinite(keys).all()) or not bool(torch.isfinite(values).all()):
            raise ValueError("memory keys and values must be finite")
        self.d_model = int(d_model)
        self.slots = int(keys.shape[0])
        self.memory_width = int(keys.shape[1])
        self.hidden_norm = RMSNorm(d_model, norm_eps)
        self.query = nn.Linear(d_model, self.memory_width, bias=False)
        self.output = nn.Linear(self.memory_width, d_model, bias=False)
        self.layer_scale = nn.Parameter(torch.full((d_model,), float(layer_scale)))
        tag_optimizer_role(self, "layer_scale", ParameterRole.GATE)
        self.register_buffer("memory_keys", keys.detach().float().clone(), persistent=True)
        self.register_buffer("memory_values", values.detach().float().clone(), persistent=True)
        self.register_buffer(
            "provenance_ids", provenance_ids.detach().long().clone(), persistent=True
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        record_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        if record_ids is not None:
            if record_ids.shape != hidden.shape[:2]:
                raise ValueError("record_ids must match [batch, sequence]")
            if record_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("record_ids must use an integer dtype")

        query = F.normalize(self.query(self.hidden_norm(hidden)).float(), dim=-1)
        keys = F.normalize(self.memory_keys.float(), dim=-1)
        scores = query @ keys.transpose(0, 1) * math.sqrt(self.memory_width)
        allowed = torch.ones_like(scores, dtype=torch.bool)
        if record_ids is not None:
            allowed &= record_ids[..., None].ne(self.provenance_ids.view(1, 1, -1))
        any_allowed = allowed.any(dim=-1, keepdim=True)
        safe_scores = scores.masked_fill(~allowed, float("-inf"))
        safe_scores = torch.where(any_allowed, safe_scores, torch.zeros_like(safe_scores))
        weights = torch.softmax(safe_scores, dim=-1) * allowed.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        read = weights @ self.memory_values.float()
        update = self.output(read.to(dtype=hidden.dtype))
        scale = self.layer_scale.to(dtype=hidden.dtype)
        return hidden + scale * update, {
            "attention_entropy": -(weights * weights.clamp_min(1e-30).log()).sum(dim=-1),
            "max_weight": weights.max(dim=-1).values,
            "retrieval_available": any_allowed.squeeze(-1),
        }
