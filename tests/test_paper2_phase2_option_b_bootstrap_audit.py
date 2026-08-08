from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eval.eval_paper2_phase2_option_b_bootstrap_audit import (
    REQUIRED_STEPS,
    _bootstrap_document_means,
    audit,
    ols_slope,
    registered_e1_support,
    separated_positive_intervals,
)
from training.paper2_phase2_matched_alpha import document_partition


def test_registered_e1_support_requires_ci_not_positive_point_estimate() -> None:
    assert not registered_e1_support(
        endpoint_relative_gain=0.004,
        late_slope_ci=(-0.0001, 0.0005),
    )
    assert registered_e1_support(
        endpoint_relative_gain=0.004,
        late_slope_ci=(0.00001, 0.0005),
    )
    assert registered_e1_support(
        endpoint_relative_gain=0.01,
        late_slope_ci=(-0.001, 0.001),
    )


def test_document_bootstrap_keeps_paired_arm_difference_exact() -> None:
    control = np.asarray(
        [[1.0, 2.0], [3.0, 4.0], [8.0, 9.0], [10.0, 11.0]], dtype=np.float64
    )
    full = control + 0.25
    draws = _bootstrap_document_means(
        trajectories={"full": full, "control": control},
        document_ids=["a", "a", "b", "b"],
        replicates=2_000,
        seed=7,
        chunk_size=31,
    )
    np.testing.assert_allclose(draws["full"] - draws["control"], 0.25, atol=1e-12)


def test_ols_slope_uses_thousands_of_updates() -> None:
    steps = np.asarray([10_000, 11_000, 12_000], dtype=np.float64)
    trajectories = np.asarray([[2.0, 2.2, 2.4], [1.0, 0.9, 0.8]], dtype=np.float64)
    np.testing.assert_allclose(ols_slope(trajectories, steps), [0.2, -0.1], atol=1e-12)


def test_separated_intervals_is_stricter_than_positive_contrast() -> None:
    assert separated_positive_intervals(dose_ci=(0.0, 0.1), fresh_ci=(0.11, 0.2))
    assert not separated_positive_intervals(dose_ci=(0.0, 0.1), fresh_ci=(0.09, 0.2))


def test_end_to_end_audit_uses_saved_rows_and_corrects_reading(tmp_path: Path) -> None:
    documents = [f"doc-{index}" for index in range(40)]
    manifest = tmp_path / "sample_manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "anchor_index": anchor,
                    "document_id": document,
                    "horizon": horizon,
                }
            )
            + "\n"
            for anchor, document in enumerate(documents)
            for horizon in (1, 2, 3, 4)
        ),
        encoding="utf-8",
    )
    stage0a_summary = tmp_path / "stage0a.json"
    stage0a_summary.write_text(
        json.dumps(
            {
                "manifest": {
                    "sample_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
                }
            }
        ),
        encoding="utf-8",
    )
    mask = document_partition(documents, evaluation_fraction=0.2, seed=20260804)
    selected = int(mask.sum())
    assert selected > 1
    private = tmp_path / "private"
    arms = []
    for seed in (0, 1):
        for arm in ("full_a2", "draft_only_control"):
            name = f"seed_{seed}_{arm}"
            arm_dir = private / name
            arm_dir.mkdir(parents=True)
            history = []
            for step in REQUIRED_STEPS:
                x = step / 1000.0
                control = 2.0 + 0.0001 * x
                value = control if arm == "draft_only_control" else control + 0.001 + 0.0002 * x
                accepted = torch.full((selected,), value, dtype=torch.float32)
                torch.save({"accepted_length": accepted}, arm_dir / f"rows_fixed_evaluation_step_{step:05d}.pt")
                history.append(
                    {
                        "step": step,
                        "evaluations": {
                            "fixed_evaluation": {"mean_accepted_length": float(accepted.mean())}
                        },
                    }
                )
            arms.append({"seed": seed, "arm": arm, "history": history})
    source_summary = tmp_path / "source.json"
    source_summary.write_text(
        json.dumps(
            {
                "kind": "paper2_phase2_option_b_matrix_v1",
                "status": "complete",
                "arms": arms,
                "population": {"existing_evaluation_anchors": selected},
                "scripted_reading": {"e1_support_in_both_seeds": True},
            }
        ),
        encoding="utf-8",
    )
    result = audit(
        source_summary_path=source_summary,
        stage0a_summary_path=stage0a_summary,
        manifest_path=manifest,
        private_root=private,
        replicates=1_000,
        bootstrap_seed=9,
    )
    assert result["corrected_scripted_reading"]["e1_support_in_both_seeds"] is True
    assert result["corrected_scripted_reading"]["writeback_retained_for_e1"] is True
    assert result["bootstrap"]["evaluation_anchors"] == selected
    assert result["seeds"][0]["full_second_half_exposure_slope_eal_per_1000"][
        "document_bootstrap_95_ci"
    ][0] > 0.0
