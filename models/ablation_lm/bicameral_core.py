"""Standalone full-width bicameral Transformer core with fixed-context K/V."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .bicameral import SwapLinear
from .layers import RMSNorm, RotaryEmbedding
from .rng import derive_module_seed


_PROJECTION_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _positive_integer(value: int, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class BicameralProjectedKeyValue:
    """One block-owned fixed-context cache for both hemispheres.

    Keys remain in grouped-query form.  The position receipt is cloned when the
    cache is built and shared by all four payload tensors through this envelope.
    """

    key_a: torch.Tensor
    value_a: torch.Tensor
    key_b: torch.Tensor
    value_b: torch.Tensor
    position_ids: torch.Tensor
    owner_id: int


class BicameralTransformerBlock(nn.Module):
    """Full-width paired Transformer block in swap-eigenmode coordinates.

    All seven dense projections are :class:`SwapLinear` modules.  RMSNorms and
    RoPE are shared between hemispheres.  Queries are projected from the live
    ``h_A``/``h_B`` states, while both hemispheres consume one block-owned K/V
    cache projected from the same fixed ``h_0`` source.

    Attention dropout is structurally fixed at zero.  The block therefore
    consumes no dropout draws and cannot shift any other module's RNG stream.
    Each randomly initialized projection instead owns a stable, namespaced
    initialization seed derived from ``module_path``.
    """

    def __init__(
        self,
        d_model: int,
        *,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        max_sequence_length: int,
        rope_theta: float = 500_000.0,
        norm_eps: float = 1e-5,
        rank: int = 32,
        sigma_delta0: float = 0.02,
        initialization_seed: int = 20_260_826,
        rng_replica: int = 0,
        module_path: str = "model.bicameral_core.0",
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = _positive_integer(d_model, name="d_model")
        self.n_heads = _positive_integer(n_heads, name="n_heads")
        self.n_kv_heads = _positive_integer(n_kv_heads, name="n_kv_heads")
        self.d_ff = _positive_integer(d_ff, name="d_ff")
        self.max_sequence_length = _positive_integer(
            max_sequence_length,
            name="max_sequence_length",
        )
        self.rank = _positive_integer(rank, name="rank")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads != 2 * self.n_kv_heads:
            raise ValueError("bicameral GQA requires the ratified 2:1 Q/KV ratio")
        self.head_dim = self.d_model // self.n_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.kv_width = self.n_kv_heads * self.head_dim
        if self.rank > min(self.kv_width, self.d_model, self.d_ff):
            raise ValueError("rank exceeds at least one projection dimension")
        if isinstance(attention_dropout, bool):
            raise TypeError("attention_dropout must be an exact real scalar")
        try:
            dropout = float(attention_dropout)
        except (TypeError, ValueError) as error:
            raise TypeError("attention_dropout must be an exact real scalar") from error
        if dropout != 0.0:
            raise ValueError(
                "bicameral attention dropout is structurally fixed at zero until "
                "a generator-aware fused kernel is ratified"
            )
        self.attention_dropout = 0.0

        # The first derivation also validates the base seed, replica, and source
        # key before any Parameters are allocated.
        projection_seeds = {
            name: derive_module_seed(
                initialization_seed,
                f"{module_path}.{name}",
                rng_replica,
            )
            % (2**63)
            for name in _PROJECTION_NAMES
        }
        self.module_path = module_path
        self.initialization_seed = initialization_seed
        self.rng_replica = rng_replica
        self.projection_initialization_seeds = tuple(
            (name, projection_seeds[name]) for name in _PROJECTION_NAMES
        )

        self.attention_norm = RMSNorm(self.d_model, norm_eps)
        self.ffn_norm = RMSNorm(self.d_model, norm_eps)
        self.query_norm = RMSNorm(self.head_dim, norm_eps)
        self.key_norm = RMSNorm(self.head_dim, norm_eps)
        self.rope = RotaryEmbedding(
            self.head_dim,
            theta=rope_theta,
            max_sequence_length=self.max_sequence_length,
        )
        self.q_proj = SwapLinear(
            self.d_model,
            self.d_model,
            rank=self.rank,
            sigma_delta0=sigma_delta0,
            seed=projection_seeds["q_proj"],
        )
        self.k_proj = SwapLinear(
            self.d_model,
            self.kv_width,
            rank=self.rank,
            sigma_delta0=sigma_delta0,
            seed=projection_seeds["k_proj"],
        )
        self.v_proj = SwapLinear(
            self.d_model,
            self.kv_width,
            rank=self.rank,
            sigma_delta0=sigma_delta0,
            seed=projection_seeds["v_proj"],
        )
        self.o_proj = SwapLinear(
            self.d_model,
            self.d_model,
            rank=self.rank,
            sigma_delta0=sigma_delta0,
            seed=projection_seeds["o_proj"],
        )
        self.gate_proj = SwapLinear(
            self.d_model,
            self.d_ff,
            rank=self.rank,
            sigma_delta0=sigma_delta0,
            seed=projection_seeds["gate_proj"],
        )
        self.up_proj = SwapLinear(
            self.d_model,
            self.d_ff,
            rank=self.rank,
            sigma_delta0=sigma_delta0,
            seed=projection_seeds["up_proj"],
        )
        self.down_proj = SwapLinear(
            self.d_ff,
            self.d_model,
            rank=self.rank,
            sigma_delta0=sigma_delta0,
            seed=projection_seeds["down_proj"],
        )

    @property
    def swap_linears(self) -> tuple[SwapLinear, ...]:
        """Return the seven coupled projections in execution order."""

        return tuple(getattr(self, name) for name in _PROJECTION_NAMES)

    @property
    def disagreement_parameter_count(self) -> int:
        """Return the exact parameter delta over seven ordinary dense maps."""

        return sum(layer.dU.numel() + layer.dV.numel() for layer in self.swap_linears)

    def _split_heads(self, values: torch.Tensor, heads: int) -> torch.Tensor:
        batch, length, _ = values.shape
        return values.view(batch, length, heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _validate_position_ids(
        position_ids: torch.Tensor,
        *,
        batch: int,
        length: int,
        device: torch.device,
    ) -> None:
        if position_ids.shape != (batch, length):
            raise ValueError("position_ids must match [batch, sequence]")
        if position_ids.device != device:
            raise ValueError("position_ids must share the hidden-state device")
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("position_ids must use an integer dtype")
        if position_ids.numel() and int(position_ids.min()) < 0:
            raise ValueError("position_ids must be non-negative")

    def _positions_for(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor | None,
        *,
        clone: bool,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        if position_ids is None:
            positions = torch.arange(length, device=hidden.device).view(1, -1)
            positions = positions.expand(batch, -1)
        else:
            positions = position_ids
        self._validate_position_ids(
            positions,
            batch=batch,
            length=length,
            device=hidden.device,
        )
        if positions.numel() and int(positions.max()) >= self.max_sequence_length:
            raise ValueError("position_ids exceed max_sequence_length")
        return positions.detach().clone() if clone else positions

    def project_kv(
        self,
        h0: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
    ) -> BicameralProjectedKeyValue:
        """Project one reusable paired K/V cache from the fixed ``h0`` stream."""

        self._validate_hidden(h0, name="h0")
        positions = self._positions_for(h0, position_ids, clone=True)
        normalized = self.attention_norm(h0)
        key_a = self.key_norm(self._split_heads(self.k_proj(normalized, +1), self.n_kv_heads))
        key_b = self.key_norm(self._split_heads(self.k_proj(normalized, -1), self.n_kv_heads))
        key_a = self.rope.apply_rotary(key_a, positions)
        key_b = self.rope.apply_rotary(key_b, positions)
        value_a = self._split_heads(self.v_proj(normalized, +1), self.n_kv_heads)
        value_b = self._split_heads(self.v_proj(normalized, -1), self.n_kv_heads)
        return BicameralProjectedKeyValue(
            key_a=key_a,
            value_a=value_a,
            key_b=key_b,
            value_b=value_b,
            position_ids=positions,
            owner_id=id(self),
        )

    def _validate_hidden(self, hidden: torch.Tensor, *, name: str) -> None:
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError(f"{name} must have shape [batch, sequence, d_model]")
        if not hidden.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if hidden.device != self.attention_norm.weight.device:
            raise ValueError(f"{name} and block parameters must share a device")
        if hidden.dtype != self.attention_norm.weight.dtype:
            raise TypeError(f"{name} and block parameters must share a dtype")

    def _validate_cache(
        self,
        cache: BicameralProjectedKeyValue,
        *,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        if not isinstance(cache, BicameralProjectedKeyValue):
            raise TypeError("projected_kv must be a BicameralProjectedKeyValue")
        if cache.owner_id != id(self):
            raise ValueError("projected K/V belong to a different bicameral block")
        batch, length, _ = hidden.shape
        expected = (batch, self.n_kv_heads, length, self.head_dim)
        for name in ("key_a", "value_a", "key_b", "value_b"):
            value = getattr(cache, name)
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if value.device != hidden.device:
                raise ValueError("projected K/V must share the query device")
            if value.dtype != hidden.dtype:
                raise TypeError("projected K/V must share the query dtype")
        if cache.position_ids.shape != (batch, length):
            raise ValueError("projected K/V position IDs have the wrong shape")
        if cache.position_ids.device != hidden.device:
            raise ValueError("projected K/V position IDs must share the query device")
        if cache.position_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("projected K/V position IDs must use an integer dtype")
        if not torch.equal(cache.position_ids, position_ids):
            raise ValueError("projected K/V were encoded with different position IDs")

    @staticmethod
    def _causal_segment_mask(
        attention_mask: torch.Tensor,
        *,
        document_ids: torch.Tensor | None,
        batch: int,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask.ndim != 2 or attention_mask.shape != (batch, length):
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if attention_mask.device != device:
            raise ValueError("attention_mask must share the hidden-state device")
        integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        if attention_mask.dtype != torch.bool and attention_mask.dtype not in integer_dtypes:
            raise TypeError("attention_mask must be boolean or an exact 0/1 integer tensor")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("attention_mask values must be exactly zero or one")
        positions = torch.arange(length, device=device)
        causal = positions[:, None] >= positions[None, :]
        allowed = causal.view(1, 1, length, length) & attention_mask.bool()[:, None, None, :]
        if document_ids is not None:
            if document_ids.shape != attention_mask.shape:
                raise ValueError("document_ids must match [batch, sequence]")
            if document_ids.device != device:
                raise ValueError("document_ids must share the hidden-state device")
            if document_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("document_ids must use an integer dtype")
            valid_document = document_ids.ge(0)
            boundaries = torch.ones_like(document_ids, dtype=torch.bool)
            if length > 1:
                boundaries[:, 1:] = document_ids[:, 1:].ne(document_ids[:, :-1])
                boundaries[:, 1:] |= valid_document[:, 1:].ne(valid_document[:, :-1])
            segments = boundaries.long().cumsum(dim=1) - 1
            segments = segments.masked_fill(~valid_document, -1)
            same_segment = segments[:, :, None].eq(segments[:, None, :])
            same_segment &= valid_document[:, :, None]
            allowed &= same_segment[:, None, :, :]
        return allowed

    def _attention(
        self,
        hidden: torch.Tensor,
        *,
        hemi: int,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        document_ids: torch.Tensor | None,
        force_math_attention: bool,
    ) -> torch.Tensor:
        query = self.query_norm(
            self._split_heads(
                self.q_proj(self.attention_norm(hidden), hemi),
                self.n_heads,
            )
        )
        query = self.rope.apply_rotary(query, position_ids)
        repeat = self.n_heads // self.n_kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        length = hidden.shape[1]
        sdpa_mask = None
        is_causal = True
        if attention_mask is not None or document_ids is not None:
            if attention_mask is None:
                assert document_ids is not None
                attention_mask = document_ids.ge(0)
            sdpa_mask = self._causal_segment_mask(
                attention_mask,
                document_ids=document_ids,
                batch=hidden.shape[0],
                length=length,
                device=hidden.device,
            )
            is_causal = False
        if force_math_attention:
            if sdpa_mask is None:
                indices = torch.arange(length, device=hidden.device)
                sdpa_mask = (indices[:, None] >= indices[None, :]).view(
                    1,
                    1,
                    length,
                    length,
                )
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~sdpa_mask, float("-inf"))
            row_has_key = sdpa_mask.any(dim=-1, keepdim=True)
            scores = torch.where(row_has_key, scores, torch.zeros_like(scores))
            probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
            probabilities = torch.where(row_has_key, probabilities, torch.zeros_like(probabilities))
            attended = probabilities @ value
        else:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=sdpa_mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )
        batch = hidden.shape[0]
        merged = attended.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        return self.o_proj(merged, hemi)

    def _ffn(self, hidden: torch.Tensor, *, hemi: int) -> torch.Tensor:
        normalized = self.ffn_norm(hidden)
        gated = F.silu(self.gate_proj(normalized, hemi)) * self.up_proj(normalized, hemi)
        return self.down_proj(gated, hemi)

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        *,
        projected_kv: BicameralProjectedKeyValue,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        residual_scale: float = 1.0,
        force_math_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one full-width paired pass using a caller-owned static cache."""

        self._validate_hidden(h_a, name="h_a")
        self._validate_hidden(h_b, name="h_b")
        if h_a.shape != h_b.shape:
            raise ValueError("h_a and h_b must have identical shapes")
        positions = self._positions_for(h_a, position_ids, clone=False)
        self._validate_cache(projected_kv, hidden=h_a, position_ids=positions)
        scale = float(residual_scale)
        if not math.isfinite(scale):
            raise ValueError("residual_scale must be finite")
        attention_a = self._attention(
            h_a,
            hemi=+1,
            key=projected_kv.key_a,
            value=projected_kv.value_a,
            position_ids=positions,
            attention_mask=attention_mask,
            document_ids=document_ids,
            force_math_attention=force_math_attention,
        )
        attention_b = self._attention(
            h_b,
            hemi=-1,
            key=projected_kv.key_b,
            value=projected_kv.value_b,
            position_ids=positions,
            attention_mask=attention_mask,
            document_ids=document_ids,
            force_math_attention=force_math_attention,
        )
        h_a = h_a + scale * attention_a
        h_b = h_b + scale * attention_b
        h_a = h_a + scale * self._ffn(h_a, hemi=+1)
        h_b = h_b + scale * self._ffn(h_b, hemi=-1)
        return h_a, h_b
