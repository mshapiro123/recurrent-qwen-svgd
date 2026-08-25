from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from eval.cache_paper2_bicameral_w2p_d4 import state_digest
from analysis.analyze_paper2_bicameral_w2p_phase_d import (
    FS2_PRIME_CONTROL,
    FS2_PRIME_FEATURE_BLOCKS,
    build_registered_feature_sets,
)
from training.paper2_bicameral_w2p import (
    conditional_row_cosine,
    deterministic_derangement,
    deterministic_stratified_folds,
    normalized_row_risk,
    select_crossfitted_map,
    validate_deployment_features,
)


AUTHORITY_SHA256 = "f89b45ef100fa46536dd93a3ef936aa8c9cfa1fc624b401b4bfc0d2b50bc2aa4"
RULINGS_SHA256 = "34352161cb69612bfc996658fab0f2d24eed381cc3895eda99a7c5a3d2e835fd"


def test_charter_and_machine_lock_are_bound() -> None:
    charter = Path("docs/STRATEGY_BICAMERAL_W2P_CONDITIONAL_MIXER_CHARTER_20260825.md")
    assert charter.stat().st_size == 13699
    assert hashlib.sha256(charter.read_bytes()).hexdigest() == AUTHORITY_SHA256
    lock = json.loads(Path("training/paper2_bicameral_w2p_lock.json").read_text())
    assert lock["authority"]["sha256"] == AUTHORITY_SHA256
    rulings = Path("docs/STRATEGY_BICAMERAL_W2P_D4_RULINGS_20260825.md")
    assert rulings.stat().st_size == 12618
    assert hashlib.sha256(rulings.read_bytes()).hexdigest() == RULINGS_SHA256
    assert lock["rulings"]["sha256"] == RULINGS_SHA256
    assert lock["phase_d"]["input_provenance"] == "student_prompt_only"
    assert lock["phase_d"]["secondary_target_binding"]["status"] == "RESOLVED_DIAGNOSTIC_ONLY"
    assert lock["phase_d"]["secondary_target_binding"]["resolved_family"] == "l0d"
    assert lock["phase_d"]["secondary_target_binding"]["gate_eligible"] is False
    assert lock["phase_d"]["selection_rule"] == "nested_blockwise_inner_cv_then_joint_refit"
    assert lock["phase_d"]["standing_law"] == "SL-3"
    assert lock["fs2"]["status"] == "BLOCKED_SOURCE_CONFLICT"
    assert lock["fs2_prime"]["feature_blocks"] == list(FS2_PRIME_FEATURE_BLOCKS)
    assert lock["fs2_prime"]["matched_single_stream_control"] == FS2_PRIME_CONTROL
    assert lock["step2_training_authorized"] is False


def test_fs2_prime_uses_only_registered_prompt_trajectory_blocks() -> None:
    generator = torch.Generator().manual_seed(17)
    sites = {}
    for site in (8, 12, 16, 18):
        sites[site] = {
            name: torch.randn((5, 7), generator=generator)
            for name in ("base", "branch_a", "branch_b")
        }
    cache = {"sites": sites}
    features = build_registered_feature_sets(cache)
    assert [tuple(value.shape) for value in features["fs1_md"]] == [(5, 7), (5, 7)]
    assert [tuple(value.shape) for value in features["fs2_prime"]] == [
        (5, 7),
        (5, 7),
        (5, 42),
        (5, 42),
    ]
    assert [tuple(value.shape) for value in features["fs0_prime"]] == [(5, 49)]
    permutation = torch.tensor([1, 2, 3, 4, 0])
    shuffled = build_registered_feature_sets(cache, branch_b_permutation=permutation)
    assert torch.equal(shuffled["fs0_prime"][0], features["fs0_prime"][0])
    assert not torch.equal(shuffled["fs2_prime"][0], features["fs2_prime"][0])


def test_leak_boundary_rejects_forced_target_features() -> None:
    validate_deployment_features(
        {
            "input_provenance": "student_prompt_only",
            "gold_answer_used": False,
            "teacher_forward_used": False,
            "oracle_routing_used": False,
        }
    )
    with pytest.raises(RuntimeError, match="leak-boundary"):
        validate_deployment_features(
            {
                "input_provenance": "prompt_plus_gold_forced_target",
                "gold_answer_used": True,
                "teacher_forward_used": False,
                "oracle_routing_used": False,
            }
        )


def test_stratified_folds_and_derangement_are_deterministic() -> None:
    labels = ["a"] * 17 + ["b"] * 15
    first = deterministic_stratified_folds(labels, folds=4, seed=9)
    second = deterministic_stratified_folds(labels, folds=4, seed=9)
    assert torch.equal(first, second)
    assert set(first.tolist()) == {0, 1, 2, 3}
    order = deterministic_derangement(len(labels), tag="test", seed=11)
    assert sorted(order.tolist()) == list(range(len(labels)))
    assert torch.all(order != torch.arange(len(labels)))


def test_crossfitted_reduced_rank_map_recovers_conditional_signal() -> None:
    generator = torch.Generator().manual_seed(23)
    rows = 72
    useful = torch.randn((rows, 10), generator=generator)
    decorative = torch.randn((rows, 10), generator=generator)
    weights = torch.randn((10, 12), generator=generator)
    target = useful @ weights + 0.03 * torch.randn((rows, 12), generator=generator)
    labels = ["a" if index % 3 else "b" for index in range(rows)]
    fit = select_crossfitted_map(
        [useful, decorative],
        target,
        labels,
        seed=31,
        rank_options=((4, 8), (2, 4)),
        ridge_options=((1e-4, 1e-2), (1e-4, 1e-2)),
    )
    risk = float(normalized_row_risk(fit["prediction"], target).mean())
    cosine = float(conditional_row_cosine(fit["prediction"], target).mean())
    assert risk < 0.35
    assert cosine > 0.8
    assert fit["selected"]["ranks"][0] in (4, 8)
    assert fit["grid"]["selection_rule"] == "nested_blockwise_inner_cv_then_joint_refit"
    assert len(fit["grid"]["outer_folds"]) == 4


def test_row_risk_is_normalized_against_population_mean() -> None:
    target = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    perfect = normalized_row_risk(target, target)
    mean = normalized_row_risk(torch.zeros_like(target), target)
    assert torch.equal(perfect, torch.zeros_like(perfect))
    assert torch.allclose(mean.mean(), torch.tensor(1.0))


def test_d4_target_is_wired_and_forward_only() -> None:
    bootstrap = Path("colab/CURRENT_A100_BOOTSTRAP_CELL.py").read_text(encoding="utf-8")
    launcher = Path("colab/STAGE5_PAPER2_BICAMERAL_W2P_D4_CELL.py").read_text(encoding="utf-8")
    runner = Path("colab/run_stage5_paper2_bicameral_w2p_d4.py").read_text(encoding="utf-8")
    assert '"paper2_bicameral_w2p_d4"' in bootstrap
    assert "paper2_bicameral_w2p_d4_v1" in launcher
    assert "no optimizer no teacher no sealed evaluation" in launcher
    assert "optimizer_constructed" in runner
    assert "confirm_scored" in runner
    assert "eval_e_scored" in runner
    assert "--wall_seconds_cap" in Path("eval/cache_paper2_bicameral_w2p_d4.py").read_text()


def test_d4_state_digest_accepts_scalar_parameters() -> None:
    module = nn.Module()
    module.register_parameter("scalar", nn.Parameter(torch.tensor(1.0)))
    first = state_digest(module)
    module.scalar.data.fill_(2.0)
    assert state_digest(module) != first
