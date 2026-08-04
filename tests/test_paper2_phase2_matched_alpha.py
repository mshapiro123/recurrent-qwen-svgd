from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn

from eval.eval_paper2_phase2_matched_alpha_decision import decide
from models.paper2_dc2_student import Phase2StudentModules, ResidualDraftHead
from training.paper2_phase2_matched_alpha import (
    alpha_transform,
    build_adamw_groups,
    distribution_overlap,
    document_partition,
    expected_accepted_length,
    normalize_sparse_with_tail,
    paired_bootstrap_interval,
    practical_equivalence,
    quality_noninferior,
    trust_saturated,
)
from training.run_paper2_phase2_matched_alpha import (
    _assert_zero_loop_identity,
    _gradient_atlas,
    _losses,
)


def test_document_partition_is_stable_and_document_isolated() -> None:
    documents = ["a", "a", "b", "c", "c"]
    first = document_partition(documents, evaluation_fraction=0.3, seed=17)
    second = document_partition(documents, evaluation_fraction=0.3, seed=17)
    assert torch.equal(first, second)
    assert bool(first[0]) == bool(first[1])
    assert bool(first[3]) == bool(first[4])


def test_alpha_transform_changes_only_registered_scaling() -> None:
    raw = torch.tensor([[1.0, 2.0]])
    basis = torch.eye(2)
    eigenvalues = torch.tensor([1.0, 4.0])
    assert torch.equal(alpha_transform(raw, basis, eigenvalues, 0.0), raw)
    assert torch.allclose(alpha_transform(raw, basis, eigenvalues, 1.0), torch.tensor([[1.0, 1.0]]))


def test_overlap_and_expected_accepted_length_match_closed_form() -> None:
    p = torch.log(torch.tensor([[[0.6, 0.4], [0.5, 0.5]]]))
    q = torch.log(torch.tensor([[[0.5, 0.5], [0.25, 0.75]]]))
    overlap = distribution_overlap(p, q)
    assert torch.allclose(overlap, torch.tensor([[0.9, 0.75]]))
    assert torch.allclose(expected_accepted_length(overlap), torch.tensor([1.575]))


def test_sparse_normalization_includes_tail_and_masks_padding() -> None:
    logits = torch.tensor([[[0.0, 20.0]]])
    mask = torch.tensor([[[True, False]]])
    normalized = normalize_sparse_with_tail(logits, torch.tensor([[0.0]]), mask)
    assert normalized.shape == (1, 1, 3)
    assert torch.allclose(normalized.exp().sum(-1), torch.ones(1, 1))
    assert float(normalized[0, 0, 1]) == float("-inf")


def test_quality_gate_uses_point_and_wilson_floors() -> None:
    assert quality_noninferior(1998, 2000)
    assert not quality_noninferior(1993, 2000)


class TinyRegisteredModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.bias = nn.Parameter(torch.ones(2))
        self.norm = nn.RMSNorm(2)
        self.scalar = nn.Parameter(torch.ones(()))


def test_optimizer_excludes_bias_norm_and_scalar_from_decay() -> None:
    module = TinyRegisteredModule()
    groups = build_adamw_groups(module, weight_decay=0.01)
    decay_ids = {id(value) for value in groups[0]["params"]}
    no_decay_ids = {id(value) for value in groups[1]["params"]}
    assert id(module.weight) in decay_ids
    assert {id(module.bias), id(module.norm.weight), id(module.scalar)} <= no_decay_ids


def test_optimizer_excludes_registered_gate_scalar_banks() -> None:
    embedding = nn.Embedding.from_pretrained(torch.zeros(17, 16), freeze=True)
    module = Phase2StudentModules(tied_embedding=embedding, hidden_size=16)
    groups = build_adamw_groups(module, weight_decay=0.01)
    no_decay_ids = {id(value) for value in groups[1]["params"]}
    assert id(module.bridge.gate_logits) in no_decay_ids
    assert id(module.bridge.rho_logits) in no_decay_ids


def test_trust_saturation_requires_strict_majority_of_full_window() -> None:
    assert not trust_saturated([True] * 50 + [False] * 50)
    assert trust_saturated([True] * 51 + [False] * 49)
    assert not trust_saturated([True] * 99)


def test_sparse_draft_head_scores_only_requested_candidates() -> None:
    torch.manual_seed(4)
    embedding = nn.Embedding(23, 8)
    head = ResidualDraftHead(
        tied_embedding=embedding,
        latent_dim=4,
        control_dim=3,
        hidden_size=8,
        rank=2,
        horizons=4,
    )
    previous = torch.randn(2, 4, 5)
    candidate_ids = torch.randint(0, 23, (2, 4, 5))
    output = head(
        previous_logits=previous,
        scratch=torch.randn(2, 8, 4),
        control_state=torch.randn(2, 3),
        candidate_ids=candidate_ids,
    )
    assert output.logits.shape == previous.shape
    with pytest.raises(ValueError, match="share shape"):
        head(
            previous_logits=previous,
            scratch=torch.randn(2, 8, 4),
            control_state=torch.randn(2, 3),
            candidate_ids=candidate_ids[..., :4],
        )


def test_paired_equivalence_uses_whole_interval() -> None:
    mean, low, high = paired_bootstrap_interval(
        torch.tensor([0.01, -0.01, 0.0, 0.0]), seed=3, draws=500
    )
    assert mean == pytest.approx(0.0)
    assert practical_equivalence(
        difference_ci=(low, high), reference_mean=1.0, relative_band=0.02
    )
    assert not practical_equivalence(
        difference_ci=(-0.01, 0.03), reference_mean=1.0, relative_band=0.02
    )


def test_decision_defaults_to_half_inside_equivalence(tmp_path) -> None:
    arms = []
    for alpha, delta in ((0.0, -0.005), (0.5, 0.0), (1.0, 0.005)):
        for seed in (0, 1):
            rows_path = tmp_path / f"rows_{alpha}_{seed}.pt"
            torch.save({"accepted_length": torch.ones(40) + delta}, rows_path)
            digest = hashlib.sha256(rows_path.read_bytes()).hexdigest()
            arms.append(
                {
                    "alpha": alpha,
                    "seed": seed,
                    "status": "complete",
                    "final": {"quality_noninferior": True, "flow_validation_loss": 0.4},
                    "final_rows": {"path": str(rows_path), "sha256": digest},
                    "gradient_atlases": [{"module_gradient_norm_cv": {"refiner": 0.2, "bridge": 0.2, "heads": 0.2}}],
                    "clip_events": {"refiner": 0.1, "bridge": 0.1, "heads": 0.1},
                }
            )
    result = decide(
        {
            "protocol_lock_commit": "cf6747264e48e2de657eb2a1646f1e7c4f152ea5",
            "adequacy_precondition_met": True,
            "arms": arms,
        },
        bootstrap_draws=200,
    )
    assert result["status"] == "selected_dev_configuration"
    assert result["selected_alpha"] == 0.5


def test_sparse_training_path_uses_teacher_width_and_emits_atlas() -> None:
    torch.manual_seed(8)
    student_embedding = nn.Embedding(31, 16)
    student_embedding.weight.requires_grad_(False)
    teacher_embedding = nn.Embedding(31, 20)
    teacher_embedding.weight.requires_grad_(False)
    module = Phase2StudentModules(tied_embedding=student_embedding, hidden_size=16)
    candidates = torch.randint(0, 31, (2, 4, 5))
    batch = {
        "hidden": torch.randn(2, 4, 16),
        "target_scratch": torch.randn(2, 8, 128),
        "candidate_ids": candidates,
        "candidate_mask": torch.ones(2, 4, 5, dtype=torch.bool),
        "base_candidates": torch.log_softmax(torch.randn(2, 4, 6), -1)[..., :5],
        "base_tail": torch.log_softmax(torch.randn(2, 4, 6), -1)[..., 5],
        "teacher_candidates": torch.log_softmax(torch.randn(2, 4, 6), -1)[..., :5],
        "teacher_tail": torch.log_softmax(torch.randn(2, 4, 6), -1)[..., 5],
        "teacher_topk_ids": torch.randint(0, 31, (2, 4, 5)),
        "teacher_topk_log_probs": torch.log_softmax(torch.randn(2, 4, 5), -1),
        "position_bucket": torch.tensor([1, 3]),
    }
    decoder = torch.randn(128, 20)
    decoder_bias = torch.randn(20)
    losses, metrics = _losses(
        module=module,
        batch=batch,
        embedding=student_embedding,
        teacher_embedding=teacher_embedding,
        decoder=decoder,
        decoder_bias=decoder_bias,
    )
    assert all(torch.isfinite(value) for value in losses.values())
    assert metrics["probe_log"].shape == (2, 4, 5)
    assert _assert_zero_loop_identity(module=module, batch=batch)["bit_exact"]
    atlas = _gradient_atlas(
        module=module,
        batch=batch,
        embedding=student_embedding,
        teacher_embedding=teacher_embedding,
        decoder=decoder,
        decoder_bias=decoder_bias,
        seed=11,
    )
    assert atlas["loop"] == 1
    assert len(atlas["flow_step_jvp_gains"]) == 4
    assert set(atlas["module_gradient_norm_cv"]) == {"refiner", "bridge", "heads"}
