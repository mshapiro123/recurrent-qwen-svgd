import torch

from training.sidecar_v2_data_spine import (
    aggregate_cluster_routing,
    build_fingerprint_memory_manifest,
    canonicalize_expert_outputs,
    deterministic_k_medoids,
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
    first, receipt = build_fingerprint_memory_manifest(
        rows, panel_item_ids=panel, admitted_field="firm", slots=6, seed=13
    )
    second, repeated = build_fingerprint_memory_manifest(
        reversed(rows), panel_item_ids=panel, admitted_field="firm", slots=6, seed=13
    )
    assert first == second
    assert receipt == repeated
    assert receipt["panel_overlap"] == 0
    assert receipt["optimizer_steps"] == 0
    assert all((row["battery"], row["item_id"]) not in panel for row in first)


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
