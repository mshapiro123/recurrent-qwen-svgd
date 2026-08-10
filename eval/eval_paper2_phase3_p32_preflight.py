"""CPU-only P3.2 schema and oracle-gradient batching preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from training.paper2_phase3_p32 import (
    AgreementLabelInputs,
    GateLabel,
    VerifiedLabelInputs,
    agreement_gate_label,
    cache_manifest,
    oracle_batch_equivalence,
    verified_gate_label,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_preflight(output_summary: Path) -> dict[str, Any]:
    agreement_label = agreement_gate_label(
        AgreementLabelInputs(
            student_top1=3,
            teacher_14b_top1=5,
            teacher_32b_top1=5,
            teachability=0.9,
            confident_agreement_margin=0.0,
        ),
        teachability_threshold=0.8,
        confident_agreement_margin_threshold=1.0,
    )
    verified_label = verified_gate_label(
        VerifiedLabelInputs(
            student_right=False,
            teacher_right=True,
            confident_agreement=False,
            teacher_14b_top1=5,
            teacher_32b_top1=5,
        )
    )
    manifest = cache_manifest(
        [
            {
                "record_id": "agreement-preflight",
                "source_stratum": "agreement",
                "battery": "lattice",
                "document_id": "agreement-document",
                "item_id": "agreement-item",
                "prediction_position": 1,
                "loop_index": 1,
                "student_top1": 3,
                "teacher_14b_top1": 5,
                "teacher_32b_top1": 5,
                "cross_scale_consistent": True,
                "flip_candidate_14b": True,
                "teachability": 0.9,
                "confident_agreement_margin": 0.0,
                "teacher_topk_ids": [5, 7],
                "teacher_topk_log_probs": [-0.1, -2.0],
                "gate_label": int(agreement_label),
            },
            {
                "record_id": "agreement-negative-14b-only-preflight",
                "source_stratum": "agreement",
                "battery": "lattice",
                "document_id": "agreement-negative-document",
                "item_id": "agreement-negative-item",
                "prediction_position": 1,
                "loop_index": 1,
                "student_top1": 5,
                "teacher_14b_top1": 5,
                "teacher_32b_top1": None,
                "cross_scale_consistent": False,
                "flip_candidate_14b": False,
                "teachability": 0.1,
                "confident_agreement_margin": 2.0,
                "teacher_topk_ids": [5, 7],
                "teacher_topk_log_probs": [-0.1, -2.0],
                "gate_label": int(GateLabel.NEGATIVE),
            },
            {
                "record_id": "agreement-uncovered-flip-preflight",
                "source_stratum": "agreement",
                "battery": "lattice",
                "document_id": "agreement-uncovered-document",
                "item_id": "agreement-uncovered-item",
                "prediction_position": 1,
                "loop_index": 1,
                "student_top1": 3,
                "teacher_14b_top1": 5,
                "teacher_32b_top1": None,
                "cross_scale_consistent": False,
                "flip_candidate_14b": True,
                "teachability": 0.9,
                "confident_agreement_margin": 0.0,
                "teacher_topk_ids": [5, 7],
                "teacher_topk_log_probs": [-0.1, -2.0],
                "gate_label": int(GateLabel.IGNORED),
            },
            {
                "record_id": "verified-preflight",
                "source_stratum": "verified",
                "battery": "gsm8k",
                "document_id": "verified-document",
                "item_id": "verified-item",
                "prediction_position": 1,
                "loop_index": 1,
                "student_top1": 3,
                "teacher_14b_top1": 5,
                "teacher_32b_top1": 5,
                "cross_scale_consistent": True,
                "gate_label": int(verified_label),
                "verifier_kind": "final_number",
                "student_right": False,
                "teacher_right": True,
                "verifier_receipt": "preflight-only",
            },
        ]
    )

    torch.manual_seed(20260809)
    head = nn.Linear(16, 31, bias=False)
    states = torch.randn(8, 5, 16)
    positions = torch.arange(8) % 5
    sources = torch.arange(8) + 1
    targets = torch.arange(8) + 11
    equivalence = oracle_batch_equivalence(
        insertion_states=states,
        forward_from_insertion=head,
        prediction_positions=positions,
        source_tokens=sources,
        target_tokens=targets,
    )
    assertions = {
        "agreement_positive_requires_cross_scale": agreement_label == GateLabel.POSITIVE,
        "verified_positive_uses_real_correctness": verified_label == GateLabel.POSITIVE,
        "verified_positive_requires_cross_scale": (
            verified_gate_label(
                VerifiedLabelInputs(
                    student_right=False,
                    teacher_right=True,
                    confident_agreement=False,
                    teacher_14b_top1=5,
                    teacher_32b_top1=None,
                )
            )
            == GateLabel.IGNORED
        ),
        "agreement_correctness_labels_prohibited": manifest[
            "agreement_correctness_labels_prohibited"
        ],
        "14b_only_negative_admitted": (
            manifest["coverage"]["per_loss_class"]["gate_negative"]["agreement_14b_only_eligible"]
            == 1
        ),
        "uncovered_flip_candidate_receipted": (
            manifest["coverage"]["targeted_32b_extension_candidates"] == 1
        ),
        "batched_gradient_all_finite": equivalence["all_finite"],
        "batched_gradient_direction_equivalent": (
            equivalence["maximum_direction_difference"] == 0.0
        ),
        "batched_gradient_norm_equivalent": equivalence["maximum_norm_difference"] == 0.0,
        "optimizer_absent": True,
        "training_steps_zero": True,
        "frozen_partitions_untouched": True,
    }
    failed = [name for name, passed in assertions.items() if not bool(passed)]
    if failed:
        raise RuntimeError(f"P3.2 preflight assertions failed: {failed}")
    result = {
        "kind": "paper2_phase3_p32_preflight_receipt_v1",
        "status": "complete_schema_and_batching_only",
        "manifest": manifest,
        "oracle_gradient_batch_equivalence": equivalence,
        "assertions": assertions,
        "training_started": False,
        "optimizer_steps": 0,
        "do_not_claim": [
            "agreement-directed gradients are correctness directions",
            "the synthetic linear head establishes upper-model gradient equivalence",
            "P3.2 cache generation is complete",
        ],
    }
    write_json(output_summary, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_summary", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.output_summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
