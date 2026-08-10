"""CPU build receipt for the Phase 3 per-position bridge gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.paper2_dc2_student import Phase3StudentModules
from training.paper2_phase3_migration import (
    PHASE3_NEW_GATE_PARAMETERS,
    migrate_phase2_trainable_state,
    trainable_state,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_build(output_summary: Path) -> dict[str, Any]:
    torch.manual_seed(20260809)
    embedding = nn.Embedding(257, 896)
    embedding.requires_grad_(False)
    module = Phase3StudentModules(tied_embedding=embedding, hidden_size=896)

    phase2_state = trainable_state(module)
    for name in PHASE3_NEW_GATE_PARAMETERS:
        phase2_state.pop(name)
    phase2_state["bridge.gate_logits"] = torch.tensor([-4.2, -3.9, -3.6, -3.3])
    migration = migrate_phase2_trainable_state(module, phase2_state)

    hidden = torch.randn(2, 12, 896)
    previous_logits = torch.randn(2, 4, 257)
    inactive = module(hidden=hidden, previous_logits=previous_logits, steps=0)
    active = module(hidden=hidden, previous_logits=previous_logits, steps=2)
    expected_scalar = torch.sigmoid(module.bridge.gate_logits[1])
    expected_positions = expected_scalar.expand_as(active.bridge.position_gate[:, 1:])

    gate_mask = torch.ones_like(active.bridge.delta[..., :1])
    gate_mask[:, 0] = 0
    scalar_reference_hidden = (
        hidden
        + active.bridge.rho * (hidden - hidden)
        + expected_scalar * gate_mask * active.bridge.delta
    )
    migration_max_abs = float(
        (active.hidden - scalar_reference_hidden).abs().max().detach()
    )
    assertions = {
        "phase2_scalar_bank_preserved": torch.equal(
            module.bridge.gate_logits.detach().cpu(), phase2_state["bridge.gate_logits"]
        ),
        "position_weights_zero_initialized": migration["new_parameters_are_zero"],
        "control_projection_zero_initialized": bool(
            torch.count_nonzero(module.bridge.gate_control.weight.detach()) == 0
        ),
        "position_uniform_at_migration": torch.equal(
            active.bridge.position_gate[:, 1:], expected_positions
        ),
        "position_zero_closed": bool(
            torch.count_nonzero(active.bridge.position_gate[:, 0]) == 0
        ),
        "migration_writeback_equivalent": migration_max_abs == 0.0,
        "zero_loop_hidden_bit_exact": torch.equal(inactive.hidden, hidden),
        "zero_loop_logits_bit_exact": torch.equal(inactive.logits, previous_logits),
        "phase3_parameter_count_established": (
            migration["phase3_trainable_parameter_count"] == 1_185_973
        ),
        "optimizer_absent": True,
        "training_steps_zero": True,
    }
    failed = [name for name, passed in assertions.items() if not bool(passed)]
    if failed:
        raise RuntimeError(f"Phase 3 gate build assertions failed: {failed}")
    result = {
        "kind": "paper2_phase3_position_gate_build_receipt_v1",
        "status": "complete_build_only_no_training",
        "architecture": {
            "gate_inputs": ["per_position_hidden", "attended_scratch", "control_state"],
            "scalar_migration": "bridge.gate_logits retained as per-loop bias",
            "new_parameter_names": list(PHASE3_NEW_GATE_PARAMETERS),
            "phase2_trainable_parameters_historical": 1_184_917,
            "phase3_trainable_parameters": migration["phase3_trainable_parameter_count"],
            "added_trainable_parameters": (
                migration["phase3_trainable_parameter_count"] - 1_184_917
            ),
        },
        "migration": migration,
        "migration_max_abs_writeback_difference": migration_max_abs,
        "assertions": assertions,
        "training_started": False,
        "optimizer_steps": 0,
        "do_not_claim": [
            "the untrained per-position gate predicts teachability",
            "the synthetic migration receipt replaces checkpoint-integrated migration",
        ],
    }
    write_json(output_summary, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_summary", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_build(args.output_summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
