"""Runtime primitives for the registered Stage 2A memory screen."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from models.paper2_dc2_student import Phase3StudentModules
from models.sidecar_v2 import (
    FingerprintContentMemory,
    FingerprintMemoryReadout,
    LiteralNGramMemory,
    ScratchpadMemoryInjection,
    deterministic_value_permutation,
)


STAGE2A_STEPS = 1_200
STAGE2A_BATCH_SIZE = 128
STAGE2A_LEARNING_RATE = 5e-4
STAGE2A_WARMUP_STEPS = 50
STAGE2A_LANDING_START = 1_081
STAGE2A_EMA_DECAY = 0.999
STAGE2A_AMPLITUDE_LOW = 0.02
STAGE2A_AMPLITUDE_HIGH = 0.11
STAGE2A_READ_AMPLITUDE = 0.05
STAGE2A_FLOW_LOOPS = 4


@dataclass(frozen=True)
class Stage2AMemoryReadout:
    value: torch.Tensor
    compatibility_gate: torch.Tensor | None
    slot_indices: torch.Tensor | None
    slot_scores: torch.Tensor | None
    slot_weights: torch.Tensor | None


class Stage2AMemorySystem(nn.Module):
    """One registered memory arm and its post-initializer scratch writer."""

    def __init__(
        self,
        *,
        arm: str,
        memory_slots: int,
        memory_keys: torch.Tensor,
        teacher_values: torch.Tensor,
        seed: int,
    ) -> None:
        super().__init__()
        if arm not in ("t3a", "t3b", "shuffled", "random"):
            raise ValueError(f"unknown Stage 2A arm: {arm}")
        if memory_keys.shape != (memory_slots, 128):
            raise ValueError("Stage 2A fingerprint keys must be [memory_slots, 128]")
        if teacher_values.shape != (memory_slots, 128):
            raise ValueError("Stage 2A teacher values must be [memory_slots, 128]")
        self.arm = arm
        self.memory_slots = int(memory_slots)
        if arm == "t3b":
            table_count = 4
            if self.memory_slots % table_count:
                raise ValueError("T3b aggregate slot count must divide across four tables")
            self.reader = LiteralNGramMemory(
                value_dim=128,
                num_slots=self.memory_slots // table_count,
                ngram_sizes=(2, 3),
                hashes_per_ngram=2,
                seed=seed,
            )
            self.aggregate_table_slots = sum(
                int(table.num_embeddings) for table in self.reader.tables
            )
            if self.aggregate_table_slots != self.memory_slots:
                raise RuntimeError("T3b parameter budget no longer matches T3a values")
        else:
            values = teacher_values.detach().clone()
            trainable_values = arm == "t3a"
            if arm == "shuffled":
                values = deterministic_value_permutation(values, seed=20_260_817)
            elif arm == "random":
                generator = torch.Generator(device="cpu").manual_seed(20_260_817)
                centered = values.float() - values.float().mean(dim=0, keepdim=True)
                scale = centered.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
                values = (
                    torch.randn(values.shape, generator=generator) * scale
                    + values.float().mean(dim=0, keepdim=True)
                ).to(dtype=teacher_values.dtype)
            self.reader = FingerprintContentMemory(
                keys=memory_keys,
                values=values,
                top_k=8,
                temperature=0.07,
                trainable_values=trainable_values,
            )
            if not trainable_values:
                self.reader.values.requires_grad_(False)
        self.injection = ScratchpadMemoryInjection(
            memory_dim=128,
            scratch_dim=128,
            n_slots=8,
            seed=seed,
        )

    def read_fingerprint(
        self,
        query: torch.Tensor,
        *,
        excluded_slot_indices: torch.Tensor | None = None,
    ) -> Stage2AMemoryReadout:
        if isinstance(self.reader, LiteralNGramMemory):
            raise RuntimeError("T3b requires literal prefix-token addressing")
        readout: FingerprintMemoryReadout = self.reader(
            query, excluded_slot_indices=excluded_slot_indices
        )
        return Stage2AMemoryReadout(
            value=readout.value,
            compatibility_gate=readout.compatibility_gate,
            slot_indices=readout.slot_indices,
            slot_scores=readout.slot_scores,
            slot_weights=readout.slot_weights,
        )

    def read_literal(
        self, token_ids: torch.Tensor, prefix_positions: torch.Tensor
    ) -> Stage2AMemoryReadout:
        if not isinstance(self.reader, LiteralNGramMemory):
            raise RuntimeError("fingerprint arms require layer-6 addressing")
        values, _audit = self.reader(token_ids)
        if prefix_positions.ndim != 1 or prefix_positions.shape[0] != token_ids.shape[0]:
            raise ValueError("literal prefix positions must have one index per row")
        selected = values[
            torch.arange(values.shape[0], device=values.device), prefix_positions.long()
        ]
        return Stage2AMemoryReadout(
            value=selected,
            compatibility_gate=None,
            slot_indices=None,
            slot_scores=None,
            slot_weights=None,
        )

    def allowed_trainable(self) -> dict[str, nn.Parameter]:
        allowed_prefixes = ("injection.",)
        if self.arm == "t3a":
            allowed_prefixes += ("reader.values", "reader.compatibility_projection.")
        elif self.arm in ("shuffled", "random"):
            allowed_prefixes += ("reader.compatibility_projection.",)
        elif self.arm == "t3b":
            allowed_prefixes += ("reader.tables.",)
        trainable: dict[str, nn.Parameter] = {}
        for name, parameter in self.named_parameters():
            permitted = any(
                name == prefix or name.startswith(prefix) for prefix in allowed_prefixes
            )
            parameter.requires_grad_(permitted)
            if permitted:
                trainable[name] = parameter
        if not trainable:
            raise RuntimeError("Stage 2A arm has no trainable parameters")
        return trainable


def canonical_fingerprint_query(
    layer6: torch.Tensor, *, student_mean: torch.Tensor, student_basis: torch.Tensor
) -> torch.Tensor:
    if student_mean.ndim == 2 and student_mean.shape[0] == 1:
        student_mean = student_mean.squeeze(0)
    if layer6.ndim != 2 or student_mean.ndim != 1 or student_basis.ndim != 2:
        raise ValueError("fingerprint canonicalization received invalid ranks")
    if layer6.shape[1] != student_mean.shape[0] or student_basis.shape[0] != layer6.shape[1]:
        raise ValueError("fingerprint canonicalization widths differ")
    return (layer6.float() - student_mean.float()) @ student_basis.float()


@torch.no_grad()
def exact_prefix_features(
    sidecar: Phase3StudentModules,
    hidden: torch.Tensor,
    prefix_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce initializer and context on each causal answer prefix exactly."""

    if hidden.ndim != 2 or hidden.shape[1] != sidecar.initializer.hidden_size:
        raise ValueError("cached final hidden must be [tokens, hidden_size]")
    positions = prefix_positions.long()
    if positions.ndim != 1 or positions.numel() < 1:
        raise ValueError("each row requires answer-prefix positions")
    if bool(((positions < 0) | (positions >= hidden.shape[0])).any()):
        raise ValueError("answer-prefix position is outside cached hidden sequence")
    initializer = sidecar.initializer
    values = hidden.float()
    normalized = initializer.hidden_norm(values)
    anchors = initializer.anchors.float()
    query = initializer.query(anchors)
    keys = initializer.key(normalized)
    projected_values = initializer.value(normalized)
    scores = query @ keys.T / math.sqrt(initializer.latent_dim)
    scratch_rows = []
    for position in positions.tolist():
        weights = torch.softmax(scores[:, : position + 1], dim=-1)
        read = weights @ projected_values[: position + 1]
        scratch_rows.append(anchors + initializer.output(read))
    scratch = torch.stack(scratch_rows)
    cumulative = values.cumsum(dim=0)
    contexts = cumulative[positions] / (positions + 1).to(values.dtype)[:, None]
    current_hidden = values[positions]
    return scratch, contexts, current_hidden


def memory_augmented_logits(
    *,
    sidecar: Phase3StudentModules,
    memory_system: Stage2AMemorySystem,
    scratch0: torch.Tensor,
    contexts: torch.Tensor,
    current_hidden: torch.Tensor,
    memory_value: torch.Tensor,
    lm_head_weight: torch.Tensor,
    amplitude: float,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Run the frozen K=4 sidecar from a trainable post-initializer write."""

    scratch = memory_system.injection(
        scratch0, memory_value, amplitude_ceiling=float(amplitude)
    )
    flow = sidecar.flow(scratch, contexts, steps=STAGE2A_FLOW_LOOPS)
    innovation = flow.updates[-1].float().square().mean(dim=-1).sqrt().mean(dim=1)
    control = sidecar.control(
        scratch=flow.state,
        previous=None,
        innovation_norm=innovation,
        student_entropy=current_hidden.new_zeros((current_hidden.shape[0],)),
        top2_margin=current_hidden.new_zeros((current_hidden.shape[0],)),
        position_bucket=current_hidden.new_zeros(
            (current_hidden.shape[0],), dtype=torch.long
        ),
    )
    compact = torch.stack((torch.zeros_like(current_hidden), current_hidden), dim=1)
    write_mask = torch.zeros(
        (current_hidden.shape[0], 2, 1), dtype=torch.bool, device=current_hidden.device
    )
    write_mask[:, 1] = True
    bridge = sidecar.bridge(
        h0=compact,
        previous=compact,
        scratch=flow.state,
        control_state=control,
        loop_index=STAGE2A_FLOW_LOOPS - 1,
        active=True,
        write_position_mask=write_mask,
    )
    augmented = bridge.hidden[:, 1]
    logits = augmented.to(lm_head_weight.dtype) @ lm_head_weight.T
    return logits, {
        "memory_write_rms": (scratch - scratch0).float().square().mean(dim=(1, 2)).sqrt(),
        "position_gate": bridge.position_gate[:, 1, 0],
        "writeback_ratio": bridge.realized_writeback_ratio[:, 1],
    }


def stage2a_learning_rate(step: int) -> float:
    if not 1 <= int(step) <= STAGE2A_STEPS:
        raise ValueError("Stage 2A step must be in 1..1200")
    if step <= STAGE2A_WARMUP_STEPS:
        return STAGE2A_LEARNING_RATE * step / STAGE2A_WARMUP_STEPS
    if step < STAGE2A_LANDING_START:
        return STAGE2A_LEARNING_RATE
    progress = (step - STAGE2A_LANDING_START + 1) / (
        STAGE2A_STEPS - STAGE2A_LANDING_START + 1
    )
    return STAGE2A_LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))


def initialize_stage2a_ema(
    parameters: Mapping[str, nn.Parameter],
) -> dict[str, torch.Tensor]:
    return {name: value.detach().float().clone() for name, value in parameters.items()}


@torch.no_grad()
def update_stage2a_ema(
    ema: Mapping[str, torch.Tensor], parameters: Mapping[str, nn.Parameter]
) -> None:
    if set(ema) != set(parameters):
        raise RuntimeError("Stage 2A EMA parameter schema changed")
    for name, value in parameters.items():
        ema[name].mul_(STAGE2A_EMA_DECAY).add_(
            value.detach().float(), alpha=1.0 - STAGE2A_EMA_DECAY
        )


def tensor_digest(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_sidecar_digest(sidecar: Phase3StudentModules) -> str:
    return tensor_digest({name: value for name, value in sidecar.state_dict().items()})


def assert_frozen_sidecar(sidecar: Phase3StudentModules, expected_digest: str) -> None:
    observed = frozen_sidecar_digest(sidecar)
    if observed != expected_digest:
        raise RuntimeError(
            f"Stage 2A frozen sidecar changed: expected={expected_digest} observed={observed}"
        )
