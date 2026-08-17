import torch

from training.sidecar_v2_data_spine import (
    aggregate_cluster_routing,
    apply_stage2a_firm_knowledge_rule,
    build_fingerprint_memory_manifest,
    canonicalize_expert_outputs,
    deterministic_k_medoids,
    fit_nondev_fingerprint_geometry,
    relevance_weighted_distance_matrix,
    ridge_low_rank_initialization,
    tensor_sha256,
)


def test_canonical_projection_and_hash_are_deterministic() -> None:
    outputs = torch.arange(48, dtype=torch.float64).reshape(3, 4, 4)
    projection = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    actual = canonicalize_expert_outputs(outputs, projection)
    torch.testing.assert_close(actual, outputs @ projection)
    assert tensor_sha256(actual) == tensor_sha256(actual.clone())


def test_functional_distance_is_symmetric_and_relevance_weighted() -> None:
    outputs = torch.tensor(
        [
            [[0.0], [0.0], [4.0]],
            [[0.0], [1.0], [4.0]],
        ]
    )
    routing = torch.tensor([[0.8, 0.2, 0.0], [0.8, 0.2, 0.0]])
    distances = relevance_weighted_distance_matrix(outputs, routing, epsilon=0.0)
    torch.testing.assert_close(distances, distances.T)
    torch.testing.assert_close(torch.diagonal(distances), torch.zeros(3))
    assert distances[0, 1] < distances[0, 2]


def test_k_medoids_recovers_two_functional_groups_deterministically() -> None:
    points = torch.tensor([0.0, 0.1, 10.0, 10.1]).unsqueeze(1)
    distances = torch.cdist(points, points).square()
    first = deterministic_k_medoids(distances, n_clusters=2)
    second = deterministic_k_medoids(distances, n_clusters=2)
    torch.testing.assert_close(first.assignments, second.assignments)
    torch.testing.assert_close(first.medoids, second.medoids)
    assert first.assignments[0] == first.assignments[1]
    assert first.assignments[2] == first.assignments[3]
    assert first.assignments[0] != first.assignments[2]


def test_cluster_routing_preserves_total_mass() -> None:
    routing = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    assignments = torch.tensor([0, 0, 1, 1])
    clustered = aggregate_cluster_routing(routing, assignments, n_clusters=2)
    torch.testing.assert_close(clustered, torch.tensor([[0.3, 0.7], [0.7, 0.3]]))
    torch.testing.assert_close(clustered.sum(1), routing.sum(1))


def test_ridge_low_rank_factorization_reconstructs_fitted_map() -> None:
    generator = torch.Generator().manual_seed(20260815)
    queries = torch.randn((32, 3), generator=generator, dtype=torch.float64)
    true_map = torch.tensor(
        [[2.0, 0.0], [0.0, -1.0], [0.5, 0.25]], dtype=torch.float64
    )
    targets = queries @ true_map
    fitted, b, a = ridge_low_rank_initialization(
        queries, targets, rank=2, ridge=1e-10
    )
    torch.testing.assert_close(b @ a, fitted, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(fitted, true_map, rtol=1e-8, atol=1e-8)


def test_fingerprint_manifest_is_deterministic_and_excludes_panel() -> None:
    rows = [
        {
            "battery": "gsm8k",
            "item_id": f"item-{index}",
            "document_id": f"doc-{index}",
            "content_sha256": f"sha-{index}",
            "firm": index != 5,
        }
        for index in range(12)
    ]
    panel = {("gsm8k", "item-1"), ("gsm8k", "item-2")}
    reserved = {("gsm8k", "item-3")}
    first, receipt = build_fingerprint_memory_manifest(
        rows,
        panel_item_ids=panel,
        reserved_item_ids=reserved,
        admitted_field="firm",
        slots=6,
        seed=13,
    )
    second, repeated = build_fingerprint_memory_manifest(
        reversed(rows),
        panel_item_ids=panel,
        reserved_item_ids=reserved,
        admitted_field="firm",
        slots=6,
        seed=13,
    )
    assert first == second
    assert receipt == repeated
    assert receipt["panel_overlap"] == 0
    assert receipt["reserved_overlap"] == 0
    assert receipt["reserved_rows_excluded"] == 1
    assert receipt["optimizer_steps"] == 0
    assert all((row["battery"], row["item_id"]) not in panel for row in first)
    assert all((row["battery"], row["item_id"]) not in reserved for row in first)


def test_fingerprint_manifest_refuses_thin_admitted_population() -> None:
    rows = [
        {
            "battery": "mmlu",
            "item_id": f"item-{index}",
            "document_id": f"doc-{index}",
            "content_sha256": f"sha-{index}",
            "firm": index < 2,
        }
        for index in range(5)
    ]
    try:
        build_fingerprint_memory_manifest(
            rows, panel_item_ids=set(), admitted_field="firm", slots=3
        )
    except RuntimeError as error:
        assert "firm-knowledge non-panel population" in str(error)
    else:
        raise AssertionError("thin admitted memory population was accepted")


def test_stage2a_firm_knowledge_requires_correctness_and_concurrence() -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64
    rows = [
        {
            "battery": "gsm8k",
            "item_id": "admit",
            "teacher_14b_correct": True,
            "teacher_14b_normalized_answer": "42",
            "teacher_32b_normalized_answer": "42",
            "teacher_14b_output_sha256": digest_a,
            "teacher_32b_output_sha256": digest_b,
            "correctness_reader": "gsm8k_numeric_equality_v1",
        },
        {
            "battery": "gsm8k",
            "item_id": "disagree",
            "teacher_14b_correct": True,
            "teacher_14b_normalized_answer": "42",
            "teacher_32b_normalized_answer": "41",
            "teacher_14b_output_sha256": digest_a,
            "teacher_32b_output_sha256": digest_b,
            "correctness_reader": "gsm8k_numeric_equality_v1",
        },
        {
            "battery": "gsm8k",
            "item_id": "incorrect",
            "teacher_14b_correct": False,
            "teacher_14b_normalized_answer": "41",
            "teacher_32b_normalized_answer": "41",
            "teacher_14b_output_sha256": digest_a,
            "teacher_32b_output_sha256": digest_b,
            "correctness_reader": "gsm8k_numeric_equality_v1",
        },
    ]
    materialized, receipt = apply_stage2a_firm_knowledge_rule(rows)
    assert [row["stage2a_firm_knowledge_admitted"] for row in materialized] == [
        True,
        False,
        False,
    ]
    assert receipt["admitted"] == 1
    assert receipt["probability_thresholds"] is None
    assert receipt["counts_by_battery"]["gsm8k"]["family_disagreement"] == 1


def test_nondev_geometry_keeps_teacher_values_native_and_scores_holdout() -> None:
    generator = torch.Generator().manual_seed(9)
    student_fit = torch.randn(180, 128, generator=generator)
    rotation, _ = torch.linalg.qr(torch.randn(128, 128, generator=generator))
    teacher_fit = student_fit @ rotation
    student_holdout = torch.randn(40, 128, generator=generator)
    teacher_holdout = student_holdout @ rotation
    geometry, receipt = fit_nondev_fingerprint_geometry(
        student_fit=student_fit,
        teacher_fit=teacher_fit,
        student_holdout=student_holdout,
        teacher_holdout=teacher_holdout,
        rank=128,
    )
    assert geometry.teacher_values(teacher_holdout).shape == (40, 128)
    assert geometry.student_keys(student_holdout).shape == (40, 128)
    assert receipt["top1_retrieval"] == 1.0
    assert receipt["teacher_values_coordinate_system"] == "teacher_pca"
    assert receipt["diagnostic_rotation_live_path"] is False
    assert receipt["dev_rows_used"] == 0
