"""Pure P3.5 stabilization, reader, and estimator-repair contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from eval.cache_paper2_phase3_agreement_oracle import analytic_oracle_directions
from models.paper2_dc2_student import Phase3StudentModules, ProbeControlState


P35_SOURCE_STEP = 4_000
P35_LANDING_STEPS = 400
P35_LOOK_STEPS = (4_100, 4_200, 4_300, 4_400)
P35_BASE_LR = 3e-4
P35_EMA_DECAY = 0.995
P35_TRAINING_RUNG = 0
P35_PRIMARY_EVAL_CEILING = 0.02
P35_SECONDARY_EVAL_CEILING = 0.08
P35_CONTROL_PROBES = 4


@dataclass(frozen=True)
class P35LandingContract:
    source_step: int = P35_SOURCE_STEP
    landing_steps: int = P35_LANDING_STEPS
    schedule: str = "cosine_to_zero"
    base_learning_rate: float = P35_BASE_LR
    ema_decay: float = P35_EMA_DECAY
    runtime_controller_frozen: bool = True
    training_rung: int = P35_TRAINING_RUNG
    primary_evaluation_ceiling: float = P35_PRIMARY_EVAL_CEILING
    secondary_evaluation_ceiling: float = P35_SECONDARY_EVAL_CEILING
    score_looks: tuple[int, ...] = P35_LOOK_STEPS
    ema_primary: bool = True
    raw_secondary: bool = True

    def validate(self) -> None:
        expected = {
            "source_step": 4000,
            "landing_steps": 400,
            "schedule": "cosine_to_zero",
            "base_learning_rate": 3e-4,
            "ema_decay": 0.995,
            "runtime_controller_frozen": True,
            "training_rung": 0,
            "primary_evaluation_ceiling": 0.02,
            "secondary_evaluation_ceiling": 0.08,
            "score_looks": (4100, 4200, 4300, 4400),
            "ema_primary": True,
            "raw_secondary": True,
        }
        if asdict(self) != expected:
            raise RuntimeError(f"P3.5 landing contract changed: {asdict(self)}")


def landing_learning_rate(step: int) -> float:
    """Cosine decay from the banked P3.4 rate to exactly zero."""

    if not P35_SOURCE_STEP < int(step) <= P35_SOURCE_STEP + P35_LANDING_STEPS:
        raise ValueError("P3.5 landing step must be in 4001..4400")
    progress = (int(step) - P35_SOURCE_STEP) / P35_LANDING_STEPS
    return P35_BASE_LR * 0.5 * (1.0 + math.cos(math.pi * progress))


def initialize_ema(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value.detach().float().clone() for name, value in state.items()}


@torch.no_grad()
def update_ema(
    ema: Mapping[str, torch.Tensor],
    state: Mapping[str, torch.Tensor],
    *,
    decay: float = P35_EMA_DECAY,
) -> None:
    if set(ema) != set(state):
        raise RuntimeError("P3.5 EMA and trainable-state keys changed")
    if not 0.0 < float(decay) < 1.0:
        raise ValueError("EMA decay must be between zero and one")
    for name in ema:
        ema[name].mul_(float(decay)).add_(state[name].detach().float(), alpha=1.0 - float(decay))


def set_p35_trainable(
    module: Phase3StudentModules, *, arm: str
) -> dict[str, nn.Parameter]:
    if arm not in ("stabilized", "probe_reader"):
        raise ValueError("P3.5 arm must be stabilized or probe_reader")
    if arm == "probe_reader" and not isinstance(module.control, ProbeControlState):
        raise RuntimeError("Arm R requires the detached multi-probe control reader")
    if arm == "stabilized" and isinstance(module.control, ProbeControlState):
        raise RuntimeError("Arm S must retain the mean-pool control reader")
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    for name, parameter in module.named_parameters():
        if name.startswith(("bridge.", "control.")):
            parameter.requires_grad_(True)
    trainable = {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    if not trainable or any(
        name.startswith(("flow.", "draft.", "initializer.")) for name in trainable
    ):
        raise RuntimeError("P3.5 trainable surface crossed the charter boundary")
    return trainable


def serving_source_tokens(
    hidden: torch.Tensor, lm_head_weight: torch.Tensor
) -> torch.Tensor:
    """Pinned BF16 serving reader used by both cache construction and audit."""

    if hidden.ndim != 2 or lm_head_weight.ndim != 2:
        raise ValueError("serving reader requires [rows, hidden] and [vocab, hidden]")
    if hidden.shape[-1] != lm_head_weight.shape[-1]:
        raise ValueError("serving reader hidden dimensions do not match")
    return (
        hidden.to(torch.bfloat16) @ lm_head_weight.to(torch.bfloat16).T
    ).argmax(dim=-1)


def repaired_oracle_payload(
    *,
    prior: Mapping[str, Any],
    selected_hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> dict[str, Any]:
    source_tokens = serving_source_tokens(selected_hidden, lm_head_weight).cpu()
    target_tokens = prior["target_tokens"].long().cpu()
    if len(source_tokens) != len(target_tokens):
        raise ValueError("repaired oracle source and target rows differ")
    directions = analytic_oracle_directions(
        lm_head_weight=lm_head_weight.cpu(),
        source_tokens=source_tokens,
        target_tokens=target_tokens,
    )
    result = dict(prior)
    result.update(
        {
            "kind": "paper2_phase3_serving_oracle_direction_cache_v2",
            "status": "complete_exact_bf16_serving_reader_identity",
            "directions": directions.to(torch.bfloat16),
            "source_tokens": source_tokens.to(torch.int32),
            "source_reader": "bf16_serving_matmul_v1",
            "source_anchor_identity_rows": int(len(source_tokens)),
            "source_anchor_identity_rate": 1.0,
            "prior_cache_kind": str(prior.get("kind")),
        }
    )
    return result


def assert_source_anchor_identity(
    *, cache: Mapping[str, Any], selected_hidden: torch.Tensor, lm_head_weight: torch.Tensor
) -> dict[str, Any]:
    observed = serving_source_tokens(selected_hidden, lm_head_weight).cpu().to(torch.int32)
    expected = cache["source_tokens"].cpu().to(torch.int32)
    matched = int((observed == expected).sum())
    rows = int(expected.numel())
    receipt = {
        "rows": rows,
        "matched_rows": matched,
        "mismatched_rows": rows - matched,
        "identity_rate": matched / max(1, rows),
        "reader": "bf16_serving_matmul_v1",
    }
    if matched != rows:
        raise RuntimeError(f"P3.5 source-anchor identity failed: {receipt}")
    return receipt


def reanchored_directions(
    *,
    current_hidden: torch.Tensor,
    target_tokens: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_tokens = serving_source_tokens(current_hidden, lm_head_weight)
    directions = analytic_oracle_directions(
        lm_head_weight=lm_head_weight,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
    )
    return source_tokens, directions


def load_p35_direction_lookup(
    cache: Mapping[str, Any],
) -> tuple[dict[str, int], torch.Tensor]:
    if cache.get("kind") != "paper2_phase3_serving_oracle_direction_cache_v2":
        raise RuntimeError("P3.5 requires the exact serving-reader oracle cache")
    if float(cache.get("source_anchor_identity_rate", 0.0)) != 1.0:
        raise RuntimeError("P3.5 oracle cache lacks 100% source-anchor identity")
    record_ids = [str(value) for value in cache["record_ids"]]
    directions = cache["directions"].float()
    if directions.shape != (len(record_ids), 896):
        raise RuntimeError("P3.5 oracle direction cache shape changed")
    return {record_id: index for index, record_id in enumerate(record_ids)}, directions


def margin_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("P3.5 margin summary requires rows")
    by_battery: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_battery.setdefault(str(row["battery"]), []).append(row)

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        token = [float(value) for row in selected for value in row["answer_token_margins"]]
        row_minimum = [float(row["answer_token_margin_minimum"]) for row in selected]
        return {
            "rows": len(selected),
            "tokens": len(token),
            "mean_answer_token_margin": sum(token) / len(token),
            "mean_row_minimum_margin": sum(row_minimum) / len(row_minimum),
            "minimum_row_minimum_margin": min(row_minimum),
        }

    return {
        "pooled": summarize(rows),
        "by_battery": {
            name: summarize(selected) for name, selected in sorted(by_battery.items())
        },
    }
