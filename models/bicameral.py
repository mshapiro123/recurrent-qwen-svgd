"""Frozen-substrate Bicameral task graph derived from the byte-locked reference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


WHT_BLOCK = 128
SEQUENTIAL_EXECUTION_SCHEDULE = "sequential_shared_middle_v1"
OPERATING_GATE_VALUE = 1.0


def sequency_wht(width: int = WHT_BLOCK) -> torch.Tensor:
    if width < 1 or width & (width - 1):
        raise ValueError("WHT width must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < width:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    changes = (matrix[:, 1:] != matrix[:, :-1]).sum(dim=1)
    return matrix[torch.argsort(changes)] / math.sqrt(width)


class WHTFrame(nn.Module):
    def __init__(self, hidden_size: int, block_size: int = WHT_BLOCK) -> None:
        super().__init__()
        if hidden_size % block_size:
            raise ValueError("hidden size must be divisible by the WHT block size")
        self.hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.blocks = self.hidden_size // self.block_size
        self.register_buffer("matrix", sequency_wht(self.block_size), persistent=True)

    def forward_transform(self, values: torch.Tensor) -> torch.Tensor:
        blocks = values.reshape(*values.shape[:-1], self.blocks, self.block_size)
        matrix = self.matrix.to(device=values.device, dtype=values.dtype)
        return blocks @ matrix.T

    def inverse_transform(self, coefficients: torch.Tensor) -> torch.Tensor:
        matrix = self.matrix.to(device=coefficients.device, dtype=coefficients.dtype)
        values = coefficients @ matrix
        return values.reshape(*values.shape[:-2], self.hidden_size)


class CallosumSplit(nn.Module):
    def __init__(self, hidden_size: int, keep_fraction: float = 0.8) -> None:
        super().__init__()
        if not 0.5 < float(keep_fraction) <= 1.0:
            raise ValueError("keep_fraction must be in (0.5, 1]")
        self.frame = WHTFrame(hidden_size)
        kept = int(round(float(keep_fraction) * self.frame.block_size))
        remove_a = torch.zeros(self.frame.block_size, dtype=torch.float32)
        remove_b = torch.zeros_like(remove_a)
        remove_a[kept:] = 1.0
        remove_b[: self.frame.block_size - kept] = 1.0
        self.register_buffer("remove_a", remove_a, persistent=True)
        self.register_buffer("remove_b", remove_b, persistent=True)
        self.gate_a = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.gate_b = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.frame.forward_transform(hidden)
        remove_a = self.remove_a.to(device=hidden.device, dtype=hidden.dtype)
        remove_b = self.remove_b.to(device=hidden.device, dtype=hidden.dtype)
        gate_a = self.gate_a.to(dtype=hidden.dtype)
        gate_b = self.gate_b.to(dtype=hidden.dtype)
        branch_a = hidden - gate_a * self.frame.inverse_transform(remove_a * coefficients)
        branch_b = hidden - gate_b * self.frame.inverse_transform(remove_b * coefficients)
        return branch_a, branch_b


class HadamardExpertBank(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        experts: int = 8,
        top_k: int = 4,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if WHT_BLOCK % experts or not 1 <= top_k <= experts:
            raise ValueError("expert count and top-k do not tile WHT128")
        self.frame = WHTFrame(hidden_size)
        self.experts = int(experts)
        self.top_k = int(top_k)
        width = WHT_BLOCK // self.experts
        supports = torch.zeros((self.experts, WHT_BLOCK), dtype=torch.float32)
        for expert in range(self.experts):
            supports[expert, expert * width : (expert + 1) * width] = 1.0
        self.register_buffer("supports", supports, persistent=True)
        generator = torch.Generator().manual_seed(int(initialization_seed))
        self.gains = nn.Parameter(
            0.02
            * torch.randn(
                (self.experts, WHT_BLOCK), generator=generator, dtype=torch.float32
            )
        )
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def occupancy_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        coefficients = self.frame.forward_transform(hidden).detach().float()
        energy = coefficients.square().sum(dim=-2)
        supports = self.supports.to(device=hidden.device)
        scores = energy @ supports.T
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        scores = self.occupancy_scores(hidden)
        indices = scores.topk(self.top_k, dim=-1).indices
        selected = torch.zeros_like(scores).scatter_(-1, indices, 1.0)
        supports = self.supports.to(device=hidden.device)
        gains = self.gains.to(device=hidden.device)
        band_gain = (selected.unsqueeze(-1) * supports * gains).sum(dim=-2)
        coefficients = self.frame.forward_transform(hidden)
        update = self.frame.inverse_transform(
            band_gain.to(dtype=hidden.dtype).unsqueeze(-2) * coefficients
        )
        return hidden + self.gate.to(dtype=hidden.dtype) * update

    @torch.no_grad()
    def load_closed_form_gain(self, gain: torch.Tensor) -> None:
        if tuple(gain.shape) != (WHT_BLOCK,) or not torch.isfinite(gain).all():
            raise ValueError("closed-form bank gain must be finite WHT128")
        self.gains.copy_(gain.float().unsqueeze(0).expand_as(self.gains))


class MuDeltaCombiner(nn.Module):
    def __init__(self, hidden_size: int, rms_cap: float = 0.55) -> None:
        super().__init__()
        if not math.isfinite(float(rms_cap)) or rms_cap <= 0.0:
            raise ValueError("rms_cap must be finite and positive")
        self.frame = WHTFrame(hidden_size)
        self.mu = nn.Parameter(torch.ones(WHT_BLOCK, dtype=torch.float32))
        self.delta = nn.Parameter(torch.zeros(WHT_BLOCK, dtype=torch.float32))
        self.rms_cap = float(rms_cap)
        self.last_saturation_fraction = 0.0
        self.last_deviation_rms = 0.0

    def forward(self, branch_a: torch.Tensor, branch_b: torch.Tensor) -> torch.Tensor:
        if branch_a.shape != branch_b.shape:
            raise ValueError("Bicameral branches must have identical shapes")
        mean_state = (branch_a + branch_b) / 2
        a = self.frame.forward_transform(branch_a)
        b = self.frame.forward_transform(branch_b)
        mu_residual = (self.mu - 1.0).to(device=a.device, dtype=a.dtype)
        delta = self.delta.to(device=a.device, dtype=a.dtype)
        spectral_deviation = mu_residual * (a + b) / 2 + delta * (a - b) / 2
        deviation = self.frame.inverse_transform(spectral_deviation)
        rms = (deviation.float().square().mean(dim=-1, keepdim=True) + 1e-24).sqrt()
        scale = (self.rms_cap / rms).clamp(max=1.0).to(dtype=deviation.dtype)
        self.last_saturation_fraction = float((rms > self.rms_cap).float().mean().detach())
        self.last_deviation_rms = float(rms.mean().detach())
        return mean_state + deviation * scale

    @torch.no_grad()
    def fit_state_matching(
        self,
        branch_a: torch.Tensor,
        branch_b: torch.Tensor,
        target: torch.Tensor,
        *,
        ridge: float = 1e-3,
    ) -> None:
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        a = self.frame.forward_transform(branch_a).float()
        b = self.frame.forward_transform(branch_b).float()
        target_spectral = self.frame.forward_transform(target).float()
        consensus = ((a + b) / 2).reshape(-1, self.frame.blocks, WHT_BLOCK)
        disagreement = ((a - b) / 2).reshape(-1, self.frame.blocks, WHT_BLOCK)
        residual = target_spectral.reshape(-1, self.frame.blocks, WHT_BLOCK) - consensus
        cc = (consensus * consensus).sum(dim=(0, 1)) + ridge
        dd = (disagreement * disagreement).sum(dim=(0, 1)) + ridge
        cd = (consensus * disagreement).sum(dim=(0, 1))
        cr = (consensus * residual).sum(dim=(0, 1))
        dr = (disagreement * residual).sum(dim=(0, 1))
        determinant = (cc * dd - cd.square()).clamp_min(1e-12)
        self.mu.copy_(1.0 + (cr * dd - dr * cd) / determinant)
        self.delta.copy_((dr * cc - cr * cd) / determinant)


class BicameralCore(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        keep_fraction: float = 0.8,
        experts: int = 8,
        top_k: int = 4,
        rms_cap: float = 0.55,
        initialization_seed: int = 20260823,
    ) -> None:
        super().__init__()
        self.callosum = CallosumSplit(hidden_size, keep_fraction)
        self.bank_a = HadamardExpertBank(
            hidden_size,
            experts=experts,
            top_k=top_k,
            initialization_seed=initialization_seed,
        )
        self.bank_b = HadamardExpertBank(
            hidden_size,
            experts=experts,
            top_k=top_k,
            initialization_seed=initialization_seed + 1,
        )
        self.combiner = MuDeltaCombiner(hidden_size, rms_cap)
        self.conditioning_receipt_sha256: str | None = None
        self.execution_schedule = SEQUENTIAL_EXECUTION_SCHEDULE

    def forward(
        self,
        hidden: torch.Tensor,
        run_middle: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_a, branch_b = self.callosum(hidden)
        branch_a = self.bank_a(branch_a)
        branch_b = self.bank_b(branch_b)
        # BF16 execution schedule is part of evaluator identity. This must
        # match the byte-locked reference's two sequential middle calls.
        branch_a = run_middle(branch_a)
        branch_b = run_middle(branch_b)
        return self.combiner(branch_a, branch_b), branch_a, branch_b

    @torch.no_grad()
    def zero_gates(self) -> None:
        self.callosum.gate_a.zero_()
        self.callosum.gate_b.zero_()
        self.bank_a.gate.zero_()
        self.bank_b.gate.zero_()
        self.combiner.mu.fill_(1.0)
        self.combiner.delta.zero_()

    @torch.no_grad()
    def set_conditioning_gates(
        self,
        *,
        callosum_a: float,
        callosum_b: float,
        bank_a: float,
        bank_b: float,
        source_receipt_sha256: str,
    ) -> None:
        values = (callosum_a, callosum_b, bank_a, bank_b)
        if not source_receipt_sha256 or any(not math.isfinite(float(value)) for value in values):
            raise ValueError("operating gates require finite values and a measurement receipt")
        self.callosum.gate_a.fill_(float(callosum_a))
        self.callosum.gate_b.fill_(float(callosum_b))
        self.bank_a.gate.fill_(float(bank_a))
        self.bank_b.gate.fill_(float(bank_b))
        self.conditioning_receipt_sha256 = str(source_receipt_sha256)

    @torch.no_grad()
    def bind_strategy_operating_gates(self, *, source_receipt_sha256: str) -> None:
        self.set_conditioning_gates(
            callosum_a=OPERATING_GATE_VALUE,
            callosum_b=OPERATING_GATE_VALUE,
            bank_a=OPERATING_GATE_VALUE,
            bank_b=OPERATING_GATE_VALUE,
            source_receipt_sha256=source_receipt_sha256,
        )

    def configure_step1_trainable(self) -> list[str]:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.combiner.parameters():
            parameter.requires_grad_(True)
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    @torch.no_grad()
    def load_branch_initializers(
        self,
        path: str | Path,
        *,
        branch_a_cluster: int = 0,
        branch_b_cluster: int = 1,
    ) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.bank_a.load_closed_form_gain(payload[f"cluster_{branch_a_cluster}_bank_gain"])
        self.bank_b.load_closed_form_gain(payload[f"cluster_{branch_b_cluster}_bank_gain"])


@dataclass
class BicameralOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    branch_a: torch.Tensor
    branch_b: torch.Tensor
    metrics: dict[str, float]


@dataclass
class BicameralBranchStates:
    combined: torch.Tensor
    branch_a: torch.Tensor
    branch_b: torch.Tensor


class BicameralTaskInferenceGraph(nn.Module):
    """Qwen layer-6-to-17 substrate adapter for the Bicameral core."""

    def __init__(
        self,
        wrapper: nn.Module,
        *,
        prelude_end: int = 6,
        middle_end: int = 18,
        keep_fraction: float = 0.8,
        experts: int = 8,
        top_k: int = 4,
        rms_cap: float = 0.55,
        initialization_seed: int = 20260823,
    ) -> None:
        super().__init__()
        if not hasattr(wrapper, "qwen") or not hasattr(wrapper, "_run_layer_range"):
            raise TypeError("Bicameral graph requires RecurrentQwenForCausalLM")
        layers = len(wrapper.qwen.layers)
        if not 0 < prelude_end < middle_end < layers:
            raise ValueError("invalid Bicameral Qwen layer split")
        hidden_size = int(wrapper.config.hidden_size)
        if hidden_size != 896 or middle_end - prelude_end != 12:
            raise ValueError("locked substrate requires Qwen2.5-0.5B layers 6-17 at d=896")
        self.wrapper = wrapper
        self.prelude_end = int(prelude_end)
        self.middle_end = int(middle_end)
        for parameter in self.wrapper.parameters():
            parameter.requires_grad_(False)
        self.core = BicameralCore(
            hidden_size,
            keep_fraction=keep_fraction,
            experts=experts,
            top_k=top_k,
            rms_cap=rms_cap,
            initialization_seed=initialization_seed,
        )

    @property
    def device(self) -> torch.device:
        return next(self.wrapper.parameters()).device

    def _run_layers(
        self,
        start: int,
        end: int,
        hidden: torch.Tensor,
        *,
        causal_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        position_embeddings: Any,
    ) -> torch.Tensor:
        output, _attentions = self.wrapper._run_layer_range(
            start=start,
            end=end,
            hidden_states=hidden,
            causal_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            collect_hidden=False,
            hidden_history=None,
        )
        return output

    def _encode_middle(
        self,
        *,
        bicameral: bool,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[BicameralBranchStates, dict[str, Any]]:
        prepared = self.wrapper._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            cache_position=None,
        )
        hidden = prepared["inputs_embeds"]
        position_ids = prepared["position_ids"]
        cache_position = prepared["cache_position"]
        causal_mask = self.wrapper._update_causal_mask(
            prepared["attention_mask"], hidden, cache_position, None, False
        )
        position_embeddings = self.wrapper._rotary_embeddings(hidden, position_ids)
        hidden = self._run_layers(
            0,
            self.prelude_end,
            hidden,
            causal_mask=causal_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        if bicameral:
            def run_middle(branch: torch.Tensor) -> torch.Tensor:
                return self._run_layers(
                    self.prelude_end,
                    self.middle_end,
                    branch,
                    causal_mask=causal_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden, branch_a, branch_b = self.core(hidden, run_middle)
        else:
            hidden = self._run_layers(
                self.prelude_end,
                self.middle_end,
                hidden,
                causal_mask=causal_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            branch_a = hidden
            branch_b = hidden
        context = {
            "causal_mask": causal_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
        }
        return BicameralBranchStates(hidden, branch_a, branch_b), context

    def cache_branch_states(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> BicameralBranchStates:
        """Run only the prelude and two frozen middle branches for Step-1 caching."""

        states, _context = self._encode_middle(
            bicameral=True,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return states

    def _forward(self, *, bicameral: bool, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> BicameralOutput:
        states, context = self._encode_middle(
            bicameral=bicameral,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = states.combined
        branch_a = states.branch_a
        branch_b = states.branch_b
        hidden = self._run_layers(
            self.middle_end,
            len(self.wrapper.qwen.layers),
            hidden,
            causal_mask=context["causal_mask"],
            position_ids=context["position_ids"],
            cache_position=context["cache_position"],
            position_embeddings=context["position_embeddings"],
        )
        hidden = self.wrapper.qwen.norm(hidden)
        logits = self.wrapper.lm_head(hidden)
        return BicameralOutput(
            logits=logits,
            hidden_states=hidden,
            branch_a=branch_a,
            branch_b=branch_b,
            metrics={
                "combiner_saturation_fraction": self.core.combiner.last_saturation_fraction,
                "combiner_deviation_rms": self.core.combiner.last_deviation_rms,
            },
        )

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> BicameralOutput:
        return self._forward(
            bicameral=True, input_ids=input_ids, attention_mask=attention_mask
        )

    def base_path(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> BicameralOutput:
        return self._forward(
            bicameral=False, input_ids=input_ids, attention_mask=attention_mask
        )
