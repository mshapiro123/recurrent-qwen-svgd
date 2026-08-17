"""Loss-free Sidecar v2 primitives used by the Stage 2A build."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FingerprintMemoryReadout:
    """Auditable output of one fingerprint-keyed memory read."""

    value: torch.Tensor
    ungated_value: torch.Tensor
    compatibility_gate: torch.Tensor
    slot_indices: torch.Tensor
    slot_scores: torch.Tensor
    slot_weights: torch.Tensor


def deterministic_value_permutation(values: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Permute complete memory values without changing their marginal tensor."""

    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("memory values must have shape [slots, value_dim]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(values.shape[0], generator=generator)
    return values.index_select(0, order.to(values.device))


class FingerprintContentMemory(nn.Module):
    """Fixed fingerprint keys with trainable values and top-k MIPS retrieval.

    Keys and the query coordinate transform are immutable buffers. The
    compatibility gate is trainable, while ``trainable_values=False`` creates
    the frozen-value honesty control with the same forward graph. Injection
    remains a separate zero-gated operation so attaching this reader cannot
    change the substrate by itself.
    """

    def __init__(
        self,
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        top_k: int,
        temperature: float = 1.0,
        trainable_values: bool = True,
    ) -> None:
        super().__init__()
        if keys.ndim != 2 or values.ndim != 2:
            raise ValueError("keys and values must be rank-two tensors")
        if keys.shape[0] != values.shape[0] or keys.shape[0] < 1:
            raise ValueError("keys and values require the same nonzero slot count")
        if not keys.is_floating_point() or not values.is_floating_point():
            raise TypeError("keys and values must be floating point")
        if not bool(torch.isfinite(keys).all()) or not bool(torch.isfinite(values).all()):
            raise ValueError("keys and values must be finite")
        if not 1 <= int(top_k) <= int(keys.shape[0]):
            raise ValueError("top_k must be in [1, slots]")
        if not float(temperature) > 0.0:
            raise ValueError("temperature must be positive")
        normalized_keys = F.normalize(keys.detach().float(), dim=-1)
        self.register_buffer("keys", normalized_keys.to(dtype=keys.dtype))
        self.values = nn.Parameter(values.detach().clone(), requires_grad=trainable_values)
        self.register_buffer(
            "query_projection",
            torch.eye(keys.shape[1], dtype=keys.dtype, device=keys.device),
        )
        self.compatibility_projection = nn.Linear(3, 1, bias=True)
        nn.init.zeros_(self.compatibility_projection.weight)
        nn.init.zeros_(self.compatibility_projection.bias)
        self.top_k = int(top_k)
        self.temperature = float(temperature)

    @property
    def slot_count(self) -> int:
        return int(self.keys.shape[0])

    def forward(self, query: torch.Tensor) -> FingerprintMemoryReadout:
        if query.ndim != 2 or query.shape[1] != self.keys.shape[1]:
            raise ValueError("query must have shape [batch, key_dim]")
        projected = query.float() @ self.query_projection.float()
        projected = F.normalize(projected, dim=-1)
        scores = projected @ self.keys.float().T
        top_scores, top_indices = torch.topk(scores, k=self.top_k, dim=-1)
        weights = torch.softmax(top_scores / self.temperature, dim=-1)
        selected = self.values[top_indices]
        raw = torch.einsum("bk,bkd->bd", weights.to(selected.dtype), selected)
        maximum = top_scores[:, :1]
        if self.top_k == 1:
            gap = maximum
            normalized_entropy = torch.zeros_like(maximum)
        else:
            gap = maximum - top_scores[:, 1:2]
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1, keepdim=True)
            normalized_entropy = entropy / math.log(self.top_k)
        features = torch.cat((maximum, gap, 1.0 - normalized_entropy), dim=-1)
        gate = torch.sigmoid(self.compatibility_projection(features.float()))
        gated = gate.to(raw.dtype) * raw
        return FingerprintMemoryReadout(
            value=gated,
            ungated_value=raw,
            compatibility_gate=gate,
            slot_indices=top_indices,
            slot_scores=top_scores,
            slot_weights=weights,
        )


class ProbePool(nn.Module):
    """Attention-pool a detached cell set with deterministic learned probes."""

    def __init__(
        self,
        *,
        cell_dim: int,
        n_probes: int = 4,
        query_dim: int = 256,
        seed: int = 20_260_815,
    ) -> None:
        super().__init__()
        if min(int(cell_dim), int(n_probes), int(query_dim)) < 1:
            raise ValueError("cell_dim, n_probes, and query_dim must be positive")
        self.cell_dim = int(cell_dim)
        self.n_probes = int(n_probes)
        self.query_dim = int(query_dim)
        generator = torch.Generator().manual_seed(int(seed))
        self.probes = nn.Parameter(
            torch.randn((self.n_probes, self.cell_dim), generator=generator) * 0.02
        )
        self.output = nn.Parameter(
            torch.randn(
                (self.query_dim, self.n_probes * self.cell_dim), generator=generator
            )
            / math.sqrt(self.n_probes * self.cell_dim)
        )

    def pool_with_weights(
        self, cells: torch.Tensor, cell_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cells.ndim != 3 or cells.shape[-1] != self.cell_dim:
            raise ValueError("cells must have shape [batch, n_cells, cell_dim]")
        if cell_mask.shape != cells.shape[:2]:
            raise ValueError("cell_mask must have shape [batch, n_cells]")
        mask = cell_mask.bool()
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every example requires at least one readable cell")
        detached = cells.detach()
        scores = detached.float() @ self.probes.float().T / math.sqrt(self.cell_dim)
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bca,bcd->bad", weights, detached.float())
        query = pooled.flatten(1) @ self.output.float().T
        query = F.normalize(query, dim=-1)
        return query.to(dtype=cells.dtype), weights

    def forward(self, cells: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
        query, _ = self.pool_with_weights(cells, cell_mask)
        return query


class GatedSidecarInjection(nn.Module):
    """Add a sidecar memory value through an exactly inert bridge attachment."""

    def __init__(self, *, memory_dim: int, hidden_dim: int, seed: int = 20_260_815) -> None:
        super().__init__()
        if min(int(memory_dim), int(hidden_dim)) < 1:
            raise ValueError("memory_dim and hidden_dim must be positive")
        generator = torch.Generator().manual_seed(int(seed))
        self.projection = nn.Parameter(
            torch.randn((int(hidden_dim), int(memory_dim)), generator=generator) * 1e-3
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, base: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if base.shape[:-1] != memory.shape[:-1]:
            raise ValueError("base and memory leading dimensions differ")
        if base.shape[-1] != self.projection.shape[0]:
            raise ValueError("base width and injection output width differ")
        if memory.shape[-1] != self.projection.shape[1]:
            raise ValueError("memory width and injection input width differ")
        projected = memory @ self.projection.to(dtype=memory.dtype).T
        return base + torch.tanh(self.gate).to(dtype=base.dtype) * projected.to(
            dtype=base.dtype
        )


def fast_wht(x: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal Walsh-Hadamard transform on the final axis.

    The implementation is allocation-safe for autograd and accepts arbitrary
    leading dimensions. The transformed width must be a positive power of two.
    """

    width = int(x.shape[-1]) if x.ndim else 0
    if width < 1 or width & (width - 1):
        raise ValueError(f"WHT requires a positive power-of-two width, got {width}")
    result = x
    block = 1
    while block < width:
        shaped = result.reshape(*result.shape[:-1], width // (2 * block), 2, block)
        left = shaped[..., 0, :]
        right = shaped[..., 1, :]
        result = torch.stack((left + right, left - right), dim=-2).flatten(-3)
        block *= 2
    return result / math.sqrt(width)


class LiteralNGramMemory(nn.Module):
    """Causal hashed prefix n-gram memory for the Stage 2A T3b control.

    Addressing uses only existing token IDs ending at each position. The module
    returns a sidecar value and never mutates or embeds tokens in the substrate.
    Separate tables make each hash function independently auditable.
    """

    def __init__(
        self,
        *,
        value_dim: int,
        num_slots: int,
        ngram_sizes: Sequence[int] = (2, 3),
        hashes_per_ngram: int = 2,
        seed: int = 20_260_815,
    ) -> None:
        super().__init__()
        sizes = tuple(int(size) for size in ngram_sizes)
        if not sizes or any(size < 1 for size in sizes):
            raise ValueError("ngram_sizes must contain positive integers")
        if len(set(sizes)) != len(sizes):
            raise ValueError("ngram_sizes must be unique")
        if min(int(value_dim), int(num_slots), int(hashes_per_ngram)) < 1:
            raise ValueError("value_dim, num_slots, and hashes_per_ngram must be positive")
        self.value_dim = int(value_dim)
        self.num_slots = int(num_slots)
        self.ngram_sizes = sizes
        self.hashes_per_ngram = int(hashes_per_ngram)
        self.seed = int(seed)
        self.tables = nn.ModuleList(
            nn.Embedding(self.num_slots, self.value_dim)
            for _ in range(len(sizes) * self.hashes_per_ngram)
        )
        generator = torch.Generator().manual_seed(self.seed)
        for table in self.tables:
            nn.init.normal_(table.weight, std=0.02, generator=generator)

    def _indices(self, tokens: torch.Tensor, size: int, hash_index: int) -> torch.Tensor:
        batch, length = tokens.shape
        valid_length = length - size + 1
        if valid_length <= 0:
            return tokens.new_empty((batch, 0), dtype=torch.long)
        windows = tokens.unfold(dimension=1, size=size, step=1).long()
        state = torch.full(
            (batch, valid_length),
            self.seed + 104_729 * (size + 17 * hash_index),
            dtype=torch.long,
            device=tokens.device,
        )
        multiplier = 1_000_003 + 2 * hash_index
        for offset in range(size):
            state = state * multiplier + windows[..., offset] + 1 + 97 * offset
        return torch.remainder(state, self.num_slots)

    def forward(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must use an integer dtype")
        batch, length = token_ids.shape
        output = next(self.parameters()).new_zeros((batch, length, self.value_dim))
        count = output.new_zeros((batch, length, 1))
        audit: dict[str, torch.Tensor] = {}
        table_index = 0
        for size in self.ngram_sizes:
            start = size - 1
            for hash_index in range(self.hashes_per_ngram):
                indices = self._indices(token_ids, size, hash_index)
                audit[f"n{size}_h{hash_index}"] = indices
                if indices.numel():
                    output[:, start:, :] += self.tables[table_index](indices)
                    count[:, start:, :] += 1
                table_index += 1
        output = output / count.clamp_min(1)
        return output, audit
