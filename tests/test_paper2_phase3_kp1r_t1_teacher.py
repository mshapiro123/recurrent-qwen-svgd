from __future__ import annotations

import math

import torch

from training.paper2_phase3_kp1r_t1_teacher import (
    battery_frequency_predictions,
    centered_gram,
    fit_orthogonal_procrustes,
    knowledge_margin_rows,
    linear_cka,
    linear_cka_from_grams,
    permute_within_battery,
    principal_angle_metrics,
    principal_angle_metrics_from_bases,
    sample_space_basis,
    stratified_alignment_split,
    summarize_margin,
    target_entropy_audit,
    transport_retrieval_metrics,
)


def test_target_entropy_audit_rejects_degenerate_generative_target() -> None:
    rows = [
        {"item_id": f"g{index}", "battery": "gsm8k"}
        for index in range(10)
    ]
    try:
        target_entropy_audit(rows, [220] * 10)
    except RuntimeError as error:
        assert "target degeneracy" in str(error)
    else:
        raise AssertionError("degenerate target should stop before scoring")


def test_frequency_control_is_train_only_battery_aware_and_deterministic() -> None:
    predicted = battery_frequency_predictions(
        [4, 4, 5, 7], ["a", "a", "a", "b"], ["a", "b", "c"]
    )
    assert predicted == [4, 7, 4]


def test_within_battery_permutation_preserves_label_multisets() -> None:
    labels = [1, 2, 3, 8, 9]
    batteries = ["a", "a", "a", "b", "b"]
    permuted = permute_within_battery(labels, batteries, seed=17)
    assert sorted(permuted[:3]) == [1, 2, 3]
    assert sorted(permuted[3:]) == [8, 9]
    assert permuted == permute_within_battery(labels, batteries, seed=17)


def test_margin_summary_requires_both_pooled_and_macro_intervals() -> None:
    margins = knowledge_margin_rows([1, 2, 3, 4], [0, 0, 0, 0], [1, 2, 3, 4])
    summary = summarize_margin(margins, ["a", "a", "b", "b"], seed=3, draws=200)
    assert summary["pooled_margin"] == 1.0
    assert summary["battery_macro_margin"] == 1.0
    assert summary["present_but_unread_gate"] is True


def test_linear_cka_is_rotation_invariant() -> None:
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(40, 8, generator=generator)
    q, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
    assert math.isclose(linear_cka(x, x @ q), 1.0, abs_tol=1e-10)
    assert math.isclose(
        linear_cka_from_grams(centered_gram(x), centered_gram(x @ q)),
        1.0,
        abs_tol=2e-7,
    )


def test_principal_angles_compare_subspaces_in_sample_space() -> None:
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(32, 6, generator=generator)
    q, _ = torch.linalg.qr(torch.randn(6, 6, generator=generator))
    metrics = principal_angle_metrics(x, x @ q)
    assert metrics["maximum_angle_degrees"] < 0.05
    bases = (
        sample_space_basis(x, rank=6),
        sample_space_basis(x @ q, rank=6),
    )
    assert principal_angle_metrics_from_bases(*bases)["maximum_angle_degrees"] < 0.05


def test_alignment_split_is_stable_and_contains_both_roles() -> None:
    ids = [f"r{index}" for index in range(20)]
    batteries = ["a"] * 10 + ["b"] * 10
    split = stratified_alignment_split(ids, batteries, seed=19, fit_fraction=0.6)
    assert split == stratified_alignment_split(ids, batteries, seed=19, fit_fraction=0.6)
    assert split.count("alignment_fit") == 12
    assert split.count("alignment_eval") == 8


def test_procrustes_transport_recovers_rotated_row_retrieval() -> None:
    generator = torch.Generator().manual_seed(23)
    student = torch.randn(80, 8, generator=generator)
    q, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
    teacher = student @ q
    rotation, student_mean, teacher_mean = fit_orthogonal_procrustes(
        student[:50], teacher[:50]
    )
    transported = (student[50:].double() - student_mean) @ rotation + teacher_mean
    torch.testing.assert_close(transported, teacher[50:].double(), atol=1e-6, rtol=1e-5)
    metrics = transport_retrieval_metrics(
        student_fit=student[:50],
        student_eval=student[50:],
        teacher_fit=teacher[:50],
        teacher_eval=teacher[50:],
        teacher_pca_dim=8,
    )
    assert metrics["top1_retrieval_accuracy"] == 1.0
