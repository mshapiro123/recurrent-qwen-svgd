"""Conservative Transformer layers surrounding the experimental substrate."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from models.sidecar_v2 import fast_wht

from .config import AblationLMConfig
from .optim import ParameterRole, mark_muon_eligible, tag_optimizer_role


class RMSNorm(nn.Module):
    """RMSNorm with an FP32 reduction and output in the input dtype."""

    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(width))
        tag_optimizer_role(self, "weight", ParameterRole.NORMALIZATION)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype=values.dtype) * self.weight.to(dtype=values.dtype)


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    """Qwen-style half rotation paired with duplicated RoPE frequencies."""

    left, right = values.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class RotaryEmbedding(nn.Module):
    """Token-position-only rotary embedding.

    Recurrent loop indices are intentionally absent from this interface, which
    prevents reasoning depth from being mistaken for additional text position.
    """

    def __init__(self, head_dim: int, *, theta: float, max_sequence_length: int) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        inverse = 1.0 / (
            float(theta) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.head_dim = int(head_dim)
        self.max_sequence_length = int(max_sequence_length)
        self.register_buffer("inverse_frequency", inverse, persistent=False)

    def cos_sin(
        self,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have shape [batch, sequence]")
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("position_ids must use an integer dtype")
        if position_ids.numel() and int(position_ids.min()) < 0:
            raise ValueError("position_ids must be non-negative")
        if position_ids.numel() and int(position_ids.max()) >= self.max_sequence_length:
            raise ValueError("position_ids exceed the configured sequence limit")
        frequencies = torch.einsum(
            "bt,d->btd",
            position_ids.to(device=device, dtype=torch.float32),
            self.inverse_frequency.to(device=device),
        )
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(dtype=dtype).unsqueeze(1), angles.sin().to(dtype=dtype).unsqueeze(1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape[0] != key.shape[0] or query.shape[-2:] != key.shape[-2:]:
            raise ValueError("query and key must align on batch, sequence, and head width")
        if position_ids.shape != (query.shape[0], query.shape[-2]):
            raise ValueError("position_ids must align with query batch and sequence")
        cosine, sine = self.cos_sin(position_ids, dtype=query.dtype, device=query.device)
        return (
            query * cosine + rotate_half(query) * sine,
            key * cosine + rotate_half(key) * sine,
        )


class GroupedQueryAttention(nn.Module):
    """Bias-free causal GQA using PyTorch scaled-dot-product attention."""

    def __init__(self, config: AblationLMConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.dropout = float(config.attention_dropout)
        self.q_proj = mark_muon_eligible(nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False))
        self.k_proj = mark_muon_eligible(
            nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        )
        self.v_proj = mark_muon_eligible(
            nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        )
        self.output_proj = mark_muon_eligible(nn.Linear(self.d_model, self.d_model, bias=False))
        self.query_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.key_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.rope = RotaryEmbedding(
            self.head_dim,
            theta=config.rope_theta,
            max_sequence_length=config.max_sequence_length,
        )

    def _split_heads(self, values: torch.Tensor, heads: int) -> torch.Tensor:
        batch, length, _ = values.shape
        return values.view(batch, length, heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _causal_padding_mask(
        attention_mask: torch.Tensor,
        length: int,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask.ndim != 2 or attention_mask.shape[1] != length:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        if attention_mask.dtype != torch.bool and attention_mask.dtype not in integer_dtypes:
            raise TypeError("attention_mask must be boolean or an exact 0/1 integer tensor")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("attention_mask values must be exactly zero or one")
        positions = torch.arange(length, device=attention_mask.device)
        causal = positions[:, None] >= positions[None, :]
        allowed = causal.view(1, 1, length, length) & attention_mask.bool()[:, None, None, :]
        if document_ids is not None:
            if document_ids.shape != attention_mask.shape:
                raise ValueError("document_ids must match [batch, sequence]")
            valid_document = document_ids.ge(0)
            boundaries = torch.ones_like(document_ids, dtype=torch.bool)
            if length > 1:
                boundaries[:, 1:] = document_ids[:, 1:].ne(document_ids[:, :-1])
                boundaries[:, 1:] |= valid_document[:, 1:].ne(valid_document[:, :-1])
            segments = boundaries.long().cumsum(dim=1) - 1
            segments = segments.masked_fill(~valid_document, -1)
            same_document = segments[:, :, None].eq(segments[:, None, :])
            same_document &= valid_document[:, :, None]
            allowed &= same_document[:, None, :, :]
        return allowed

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        batch, length, _ = hidden.shape
        if position_ids is None:
            position_ids = torch.arange(length, device=hidden.device).view(1, -1).expand(batch, -1)
        elif position_ids.shape != (batch, length):
            raise ValueError("position_ids must match [batch, sequence]")

        query = self.query_norm(self._split_heads(self.q_proj(hidden), self.n_heads))
        key = self.key_norm(self._split_heads(self.k_proj(hidden), self.n_kv_heads))
        value = self._split_heads(self.v_proj(hidden), self.n_kv_heads)
        query, key = self.rope(query, key, position_ids)

        repeat = self.n_heads // self.n_kv_heads
        if repeat > 1:
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        sdpa_mask = None
        is_causal = True
        if attention_mask is not None or document_ids is not None:
            if attention_mask is None:
                assert document_ids is not None
                attention_mask = document_ids.ge(0)
            sdpa_mask = self._causal_padding_mask(attention_mask, length, document_ids)
            is_causal = False
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        merged = attended.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        return self.output_proj(merged)


class SwiGLU(nn.Module):
    """Bias-free SwiGLU with separately addressable optimizer tensors."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = mark_muon_eligible(nn.Linear(d_model, d_ff, bias=False))
        self.up_proj = mark_muon_eligible(nn.Linear(d_model, d_ff, bias=False))
        self.down_proj = mark_muon_eligible(nn.Linear(d_ff, d_model, bias=False))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class TransformerBlock(nn.Module):
    """Plain Pre-RMSNorm Transformer block with an explicit residual scale."""

    def __init__(self, config: AblationLMConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.feed_forward = SwiGLU(config.d_model, config.d_ff)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        residual_scale: float = 1.0,
    ) -> torch.Tensor:
        scale = float(residual_scale)
        hidden = hidden + scale * self.attention(
            self.attention_norm(hidden),
            attention_mask=attention_mask,
            position_ids=position_ids,
            document_ids=document_ids,
        )
        return hidden + scale * self.feed_forward(self.ffn_norm(hidden))


class ModifiedHadamardExpertBank(nn.Module):
    """Differentiable dense-routing WHT experts for the upfront challenger.

    Unlike the legacy detached hard-top-k bank, every expert and the router are
    gradient-live at initialization.  Sparse routing remains a later ablation.
    """

    def __init__(
        self,
        d_model: int,
        *,
        experts: int,
        layer_scale: float,
        norm_eps: float,
        seed: int,
    ) -> None:
        super().__init__()
        if d_model < 1 or d_model & (d_model - 1):
            raise ValueError("Hadamard width must be a positive power of two")
        self.d_model = int(d_model)
        self.experts = int(experts)
        self.norm = RMSNorm(d_model, norm_eps)
        self.router = nn.Linear(d_model, experts, bias=False)
        self.expert_gains = nn.Parameter(torch.empty(experts, d_model))
        self.layer_scale = nn.Parameter(torch.full((d_model,), float(layer_scale)))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        permutations = torch.stack(
            [torch.randperm(d_model, generator=generator) for _ in range(experts)]
        )
        signs = 2 * torch.randint(0, 2, (experts, d_model), generator=generator) - 1
        permutations[0] = torch.arange(d_model)
        signs[0] = 1
        self.register_buffer("permutations", permutations, persistent=True)
        self.register_buffer("signs", signs.float(), persistent=True)
        tag_optimizer_role(self, "expert_gains", ParameterRole.NOVEL_MODULE)
        tag_optimizer_role(self, "layer_scale", ParameterRole.GATE)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router.weight, std=0.02)
        with torch.no_grad():
            self.expert_gains.normal_(mean=1.0, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden)
        routing = torch.softmax(self.router(normalized).float(), dim=-1).to(dtype=hidden.dtype)
        coefficients = fast_wht(normalized)
        expanded = coefficients.unsqueeze(-2).expand(*coefficients.shape[:-1], self.experts, -1)
        indices = self.permutations.view(
            *((1,) * (coefficients.ndim - 1)), self.experts, self.d_model
        ).expand_as(expanded)
        permuted = torch.gather(expanded, dim=-1, index=indices)
        filtered = (
            permuted
            * self.signs.to(dtype=hidden.dtype)
            * self.expert_gains.to(dtype=hidden.dtype)
        )
        expert_updates = fast_wht(filtered)
        update = (routing.unsqueeze(-1) * expert_updates).sum(dim=-2)
        return hidden + self.layer_scale.to(dtype=hidden.dtype) * update
