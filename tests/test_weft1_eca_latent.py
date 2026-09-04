from __future__ import annotations

import copy
import hashlib
import json

import pytest
import torch

from analysis.weft1_eca_latent import (
    DELIVERED_AUTHORITIES,
    ECAReceiptIdentityError,
    ECA_HORIZONS,
    ECA_REPLICAS,
    ECA_RULES,
    ECA_SELF_HASH_FIELD,
    ECACellSpec,
    SMOKE_PROFILE,
    _canonical_json_bytes,
    _exclusive_writer_lock,
    _with_self_hash,
    assigned_grid,
    authority_receipts,
    derived_seed_receipt,
    eca_step,
    executed_k_at_step,
    k_values,
    minibatch_order_sha256,
    registered_grid,
    ridge_probe_matrix,
    run_campaign,
    run_cell,
    shard_directory_name,
    validate_cell_receipt,
    verify_campaign,
)


def test_delivered_eca_authorities_are_byte_exact() -> None:
    observed = authority_receipts()

    assert tuple(
        (row["path"], row["bytes"], row["sha256"]) for row in observed
    ) == DELIVERED_AUTHORITIES


def test_registered_grid_is_exactly_the_ratified_72_cells() -> None:
    grid = registered_grid()

    assert len(grid) == 72
    assert len(set(grid)) == 72
    assert {cell.rule for cell in grid} == set(ECA_RULES)
    assert {cell.tau for cell in grid} == set(ECA_HORIZONS)
    assert {cell.replica for cell in grid} == set(ECA_REPLICAS)
    for tau in ECA_HORIZONS:
        assert {cell.k for cell in grid if cell.tau == tau} == set(k_values(tau))
        assert all(2 * cell.tau + 1 in (9, 17, 33) for cell in grid if cell.tau == tau)


def _reference_step(x: torch.Tensor, rule: int) -> torch.Tensor:
    result = torch.empty_like(x)
    for example in range(x.shape[0]):
        for cell in range(x.shape[1]):
            left = int(x[example, (cell - 1) % x.shape[1]])
            center = int(x[example, cell])
            right = int(x[example, (cell + 1) % x.shape[1]])
            bit = 4 * left + 2 * center + right
            result[example, cell] = (rule >> bit) & 1
    return result


@pytest.mark.parametrize("rule", (0, 15, 30, 54, 110, 255))
def test_eca_step_matches_independent_wolfram_reference(rule: int) -> None:
    generator = torch.Generator().manual_seed(9182)
    x = torch.randint(0, 2, (7, 32), dtype=torch.uint8, generator=generator)

    assert torch.equal(eca_step(x, rule), _reference_step(x, rule))


def test_named_streams_pair_k_arms_but_separate_replicas() -> None:
    shallow = derived_seed_receipt(ECACellSpec(54, 8, 1, 0))
    deep = derived_seed_receipt(ECACellSpec(54, 8, 16, 0))
    other_rule = derived_seed_receipt(ECACellSpec(110, 8, 16, 0))
    other_replica = derived_seed_receipt(ECACellSpec(54, 8, 16, 1))

    assert shallow == deep
    assert shallow["data_train"] == other_rule["data_train"]
    assert shallow["model_init"] == other_rule["model_init"]
    assert shallow["train_order"] != other_rule["train_order"]
    assert all(shallow[name] != other_replica[name] for name in shallow)
    assert all(
        str(row["source_key"]).startswith("weft.preflight.eca_latent.")
        for row in shallow.values()
    )


def test_curriculum_is_only_for_k_at_least_four_and_returns_executed_depth() -> None:
    boundaries = (200, 400, 700)

    assert [executed_k_at_step(2, step, boundaries) for step in (0, 250, 900)] == [
        2,
        2,
        2,
    ]
    assert [
        executed_k_at_step(16, step, boundaries)
        for step in (0, 199, 200, 399, 400, 699, 700, 1199)
    ] == [1, 1, 2, 2, 4, 4, 16, 16]


def test_ridge_probe_emits_full_k_by_tau_matrix_on_separate_splits() -> None:
    fit_targets = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.uint8,
    )
    eval_targets = torch.tensor(
        [[0, 1, 0], [1, 0, 1], [1, 1, 0]], dtype=torch.uint8
    )
    # One spatial cell per independent example.  F^0 is unused by the probe;
    # F^1..F^3 are linearly embedded in every hidden visit.
    fit_states = (torch.zeros(8, 1, dtype=torch.uint8),) + tuple(
        fit_targets[:, index : index + 1] for index in range(3)
    )
    eval_states = (torch.zeros(3, 1, dtype=torch.uint8),) + tuple(
        eval_targets[:, index : index + 1] for index in range(3)
    )
    fit_features = (2.0 * fit_targets.float() - 1.0).unsqueeze(1)
    eval_features = (2.0 * eval_targets.float() - 1.0).unsqueeze(1)

    matrix = ridge_probe_matrix(
        (fit_features, fit_features),
        fit_states,
        (eval_features, eval_features),
        eval_states,
        ridge=1e-6,
    )

    assert len(matrix) == 2
    assert all(len(row) == 3 for row in matrix)
    assert all(value == pytest.approx(1.0) for row in matrix for value in row)


def test_smoke_cell_is_exactly_replayable_and_obeys_receipt_semantics() -> None:
    spec = ECACellSpec(54, 4, 8, 0)

    first = run_cell(spec, profile=SMOKE_PROFILE)
    second = run_cell(spec, profile=SMOKE_PROFILE)

    assert first == second
    assert first["kernel_size"] == 9
    assert first["scientific_status"] == "unregistered_configuration_not_scientific"
    assert first["instrument_status"] == "analysis_pending"
    assert first["eval_population"]["curve_and_terminal_are_identical"] is True
    assert first["terminal_bpc"] == first["eval_curve"][-1]["eval_bpc"]
    assert all(
        point["executed_k"] == point["scored_k"]
        for point in first["eval_curve"]
    )
    assert [point["executed_k"] for point in first["eval_curve"]] == [1, 2, 4, 8]
    assert len(first["probe"]["matrix"]) == 8
    assert all(len(row) == 4 for row in first["probe"]["matrix"])
    assert first["probe"]["classification"] is None
    assert first["probe"]["fit_dataset_sha256"] != first["probe"]["eval_dataset_sha256"]
    losses = [point["eval_bpc"] for point in first["eval_curve"]]
    deltas = [point["delta_prediction_tokens"] for point in first["eval_curve"]]
    expected_area = sum((loss - losses[-1]) * delta for loss, delta in zip(losses, deltas))
    assert first["preq_area"]["preq_area"] == pytest.approx(expected_area)
    assert first[ECA_SELF_HASH_FIELD] == hashlib.sha256(
        _canonical_json_bytes(
            {key: value for key, value in first.items() if key != ECA_SELF_HASH_FIELD}
        )
    ).hexdigest()


def test_model_and_minibatch_fingerprints_are_k_paired_and_machine_verifiable() -> None:
    shallow_spec = ECACellSpec(54, 4, 1, 0)
    deep_spec = ECACellSpec(54, 4, 8, 0)

    shallow = run_cell(shallow_spec, profile=SMOKE_PROFILE)
    deep = run_cell(deep_spec, profile=SMOKE_PROFILE)

    assert shallow["model_initial_state_sha256"] == deep["model_initial_state_sha256"]
    assert shallow["minibatch_order"]["sha256"] == deep["minibatch_order"]["sha256"]
    assert shallow["minibatch_order"]["sha256"] == minibatch_order_sha256(
        shallow_spec, SMOKE_PROFILE
    )


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    unhashed = {key: value for key, value in payload.items() if key != ECA_SELF_HASH_FIELD}
    return _with_self_hash(unhashed)


def test_semantic_tampering_fails_even_with_a_recomputed_self_hash() -> None:
    spec = ECACellSpec(54, 4, 8, 0)
    original = run_cell(spec, profile=SMOKE_PROFILE)
    mutations = []

    terminal_accuracy = copy.deepcopy(original)
    terminal_accuracy["terminal_accuracy"] = 0.25
    mutations.append(terminal_accuracy)

    eval_accuracy = copy.deepcopy(original)
    eval_accuracy["eval_curve"][0]["eval_accuracy"] = 1.5
    mutations.append(eval_accuracy)

    probe_range = copy.deepcopy(original)
    probe_range["probe"]["matrix"][0][0] = -0.1
    mutations.append(probe_range)

    eval_step = copy.deepcopy(original)
    eval_step["eval_curve"][0]["step"] = 2
    mutations.append(eval_step)

    eval_delta = copy.deepcopy(original)
    eval_delta["eval_curve"][0]["delta_prediction_tokens"] += 1
    mutations.append(eval_delta)

    executed_k = copy.deepcopy(original)
    executed_k["eval_curve"][0]["executed_k"] = 8
    executed_k["eval_curve"][0]["scored_k"] = 8
    mutations.append(executed_k)

    curriculum = copy.deepcopy(original)
    curriculum["curriculum"]["enabled"] = False
    mutations.append(curriculum)

    dataset = copy.deepcopy(original)
    dataset["dataset_sha256"]["train"] = "0" * 64
    mutations.append(dataset)

    model = copy.deepcopy(original)
    model["model_initial_state_sha256"] = "0" * 64
    mutations.append(model)

    order = copy.deepcopy(original)
    order["minibatch_order"]["sha256"] = "0" * 64
    mutations.append(order)

    scientific_claim = copy.deepcopy(original)
    scientific_claim["scientific_status"] = "registered_measurement"
    mutations.append(scientific_claim)

    for payload in mutations:
        with pytest.raises(ECAReceiptIdentityError):
            validate_cell_receipt(
                _rehash(payload),
                expected_identity=original["cell_identity_sha256"],
                expected_spec=spec,
            )


def test_campaign_resume_is_noop_and_corruption_fails_closed(tmp_path) -> None:
    shard_count = 72
    shard_index = 0
    spec = assigned_grid(shard_index, shard_count)[0]
    output = tmp_path / "campaign"

    first_manifest = run_campaign(
        output,
        profile=SMOKE_PROFILE,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    shard_root = output / "shards" / shard_directory_name(shard_index, shard_count)
    receipt_path = shard_root / spec.filename
    before = receipt_path.read_bytes()
    before_sha = hashlib.sha256(before).hexdigest()
    second_manifest = run_campaign(
        output,
        profile=SMOKE_PROFILE,
        shard_index=shard_index,
        shard_count=shard_count,
    )

    assert first_manifest == second_manifest
    assert receipt_path.read_bytes() == before
    assert second_manifest["status"] == "shard_complete"
    assert second_manifest["instrument_status"] == "analysis_pending"
    assert second_manifest["completed_cells"] == 1
    assert second_manifest["total_campaign_cells"] == 72
    assert second_manifest["cell_receipts"][spec.filename]["sha256"] == before_sha

    (shard_root / "manifest.json").unlink()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["terminal_bpc"] += 1.0
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ECAReceiptIdentityError, match="self-hash"):
        run_campaign(
            output,
            profile=SMOKE_PROFILE,
            shard_index=shard_index,
            shard_count=shard_count,
        )


def test_full_campaign_identity_is_constant_across_shard_counts(tmp_path) -> None:
    first = run_campaign(
        tmp_path / "one",
        profile=SMOKE_PROFILE,
        shard_index=0,
        shard_count=72,
    )
    second = run_campaign(
        tmp_path / "two",
        profile=SMOKE_PROFILE,
        shard_index=0,
        shard_count=36,
    )

    assert first["campaign_identity_sha256"] == second["campaign_identity_sha256"]
    assert first["total_campaign_cells"] == second["total_campaign_cells"] == 72
    assert first["shard_cells"] == 1
    assert second["shard_cells"] == 2


def test_same_shard_rejects_a_concurrent_writer(tmp_path) -> None:
    output = tmp_path / "campaign"
    shard_index, shard_count = 0, 72
    lock = (
        output
        / "shards"
        / shard_directory_name(shard_index, shard_count)
        / ".writer.lock"
    )

    with _exclusive_writer_lock(lock, {"test": "held"}):
        with pytest.raises(ECAReceiptIdentityError, match="concurrent"):
            run_campaign(
                output,
                profile=SMOKE_PROFILE,
                shard_index=shard_index,
                shard_count=shard_count,
            )


def test_campaign_rejects_an_unexpected_cell_receipt(tmp_path) -> None:
    output = tmp_path / "campaign"
    shard_index, shard_count = 0, 72
    shard_root = output / "shards" / shard_directory_name(shard_index, shard_count)
    shard_root.mkdir(parents=True)
    (shard_root / "cell_unregistered.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ECAReceiptIdentityError, match="unexpected"):
        run_campaign(
            output,
            profile=SMOKE_PROFILE,
            shard_index=shard_index,
            shard_count=shard_count,
        )


def test_aggregate_verifier_requires_and_preserves_analysis_pending(tmp_path) -> None:
    output = tmp_path / "campaign"
    run_campaign(output, profile=SMOKE_PROFILE, shard_count=1)

    aggregate = verify_campaign(output, profile=SMOKE_PROFILE, shard_count=1)

    assert aggregate["status"] == "analysis_pending"
    assert aggregate["instrument_status"] == "analysis_pending"
    assert aggregate["execution_status"] == "all_72_registered_cells_verified"
    assert aggregate["scientific_status"] == "unregistered_configuration_not_scientific"
    assert aggregate["total_campaign_cells"] == 72
    assert len(aggregate["cell_receipts"]) == 72
