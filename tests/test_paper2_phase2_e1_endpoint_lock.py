from __future__ import annotations

import json
from pathlib import Path

import torch

from eval.prepare_paper2_phase2_e1_endpoint_lock import KIND, prepare_endpoint_lock
from training.paper2_phase2_e1_confirmation import sha256_file


def test_endpoint_lock_preparation_is_integrity_only(tmp_path: Path) -> None:
    checkpoints = {}
    expected = {}
    for seed in (0, 1):
        for arm in ("full_a2", "draft_only_control"):
            name = f"seed_{seed}_{arm}"
            path = tmp_path / f"{name}.pt"
            torch.save(
                {
                    "kind": "paper2_phase2_option_b_arm_v1",
                    "name": name,
                    "seed": seed,
                    "arm": arm,
                    "step": 20_000,
                    "abort_reason": None,
                    "trainable_state": {"weight": torch.tensor([seed + 1.0])},
                    "frozen_parameter_hash_before": "frozen",
                    "frozen_parameter_hash_after": "frozen",
                },
                path,
            )
            checkpoints[name] = path
            expected[name] = {"path": str(path), "sha256": sha256_file(path)}
    registration = tmp_path / "draft.json"
    registration.write_text(json.dumps({"checkpoints": expected}), encoding="utf-8")
    output = tmp_path / "receipt.json"
    result = prepare_endpoint_lock(
        registration=registration, checkpoints=checkpoints, output=output
    )
    assert result["kind"] == KIND
    assert result["eval_d_touched"] is False
    assert result["outcome_scores_computed"] is False
    assert result["read_once_scoring_spent"] is False
    assert result["optimizer_constructed"] is False
    assert len(result["endpoints"]) == 4
    assert all(
        len(row["semantic_trainable_state_digest"]) == 64
        for row in result["endpoints"].values()
    )
