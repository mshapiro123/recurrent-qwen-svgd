from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import math

import pytest

from training.weft1_gtok_a1_contract import (
    ScreenCorpusReceiptV2,
    StratumFloorReceiptV2,
)
from training.weft1_gtok_contract import (
    FlatAdamWRecipe,
    GTOK_EXECUTION_AUTHORITY_CHAIN_V2,
    GTOK_HELDOUT_BYTE_TARGET,
    GTOK_PROXY_TOPOLOGY_SHA256,
    GTOK_SCREEN_HELDOUT_STRATUM_TARGETS,
    GTOK_SCREEN_TRAIN_STRATUM_TARGETS,
    GTOK_STRATA,
    GTOK_TRAINING_BYTE_BUDGET,
    GTOK_VOCABULARY_ARMS,
    StratumNllReceipt,
    a1_flat_adamw_recipe,
    execution_authority_v2_bound_sha256,
    sha256_bytes,
)
from training.weft1_gtok_v2_contract import (
    ArmCalibrationProjectionV2,
    BpbMilestoneReceiptV2,
    CampaignComputeReceiptV2,
    ComputeAttemptReceiptV2,
    ComputeConfirmationRunV2,
    FrozenScreenCorpusV2,
    GTOK_A2_BINDINGS_SHA256,
    GTOK_AMENDMENT_A2_SHA256,
    GTOK_AMENDMENT_A3_SHA256,
    GTOK_CALIBRATION_MAX_STEPS,
    GTOK_FIRST_BOUNDARY_BYTES,
    GTOK_RELEASE_CLOSE_SHA256,
    GTOK_SECOND_BOUNDARY_BYTES,
    GTOK_SELECTOR_LITERAL_BINDING_SHA256_V2,
    GTOK_TERMINAL_BUDGET,
    GTOK_TERMINAL_METRIC,
    GTOK_TRIPWIRE_A100_MICROSECONDS,
    GTOK_V2_AUTHORITY_CHAIN,
    GTokRunReceiptV2,
    GTokV2Stop,
    PreflightProjectionReceiptV2,
    RuntimeTripwireSnapshotV2,
    SelectionComparisonV2,
    TokenizerArmReceiptV2,
    VocabExtBasisV2,
    ValidatedComputeConfirmationV2,
    ValidatedGTokMatrixV2,
    VocabularyAdmissibilityReceiptV2,
    VocabularyFreezeArtifactV2,
    enforce_runtime_tripwire_v2,
    gtok_v2_bound_sha256,
    mint_vocabulary_freeze_v2,
    select_vocabulary_v2,
    validate_complete_gtok_matrix_v2,
    validate_compute_confirmation_v2,
    validate_selection_receipt_v2,
)


SEEDS = (101, 202)
DEFAULT_TERMINAL_BPBS = {
    (16_384, 101): 1.20,
    (16_384, 202): 1.22,
    (24_576, 101): 1.10,
    (24_576, 202): 1.12,
    (32_768, 101): 1.00,
    (32_768, 202): 1.02,
    (49_152, 101): 0.95,
    (49_152, 202): 0.97,
}


def _hash(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


def _floors(stream: str, *, shortfall: int = 1) -> tuple[StratumFloorReceiptV2, ...]:
    targets = dict(
        GTOK_SCREEN_TRAIN_STRATUM_TARGETS
        if stream == "T"
        else GTOK_SCREEN_HELDOUT_STRATUM_TARGETS
    )
    return tuple(
        StratumFloorReceiptV2(
            stream=stream,
            stratum=stratum,
            target_bytes=targets[stratum],
            realized_bytes=targets[stratum] - shortfall,
            ordered_document_ids_sha256=_hash(f"{stream}-{stratum}-order"),
            boundary_document_id_sha256=_hash(f"{stream}-{stratum}-boundary"),
            next_document_byte_count=shortfall + 1,
        )
        for stratum in GTOK_STRATA
    )


def _corpus() -> FrozenScreenCorpusV2:
    floors = ScreenCorpusReceiptV2(
        training=_floors("T"),
        heldout=_floors("H"),
        training_stream_sha256=_hash("training-stream"),
        heldout_stream_sha256=_hash("heldout-stream"),
        document_overlap_count=0,
        cluster_overlap_count=0,
    )
    return FrozenScreenCorpusV2(
        full_corpus_manifest_sha256=_hash("full-corpus-manifest"),
        screen_submanifest_sha256=_hash("screen-submanifest"),
        corpus_freeze_receipt_sha256=_hash("p-b-freeze"),
        d1_d6_gate_bundle_sha256=_hash("d1-d6"),
        decontamination_receipt_sha256=_hash("decon"),
        floors=floors,
    )


def _tokenizers(corpus: FrozenScreenCorpusV2) -> tuple[TokenizerArmReceiptV2, ...]:
    return tuple(
        TokenizerArmReceiptV2(
            vocab_size=vocab_size,
            tokenizer_json_sha256=_hash(f"tokenizer-json-{vocab_size}"),
            merges_sha256=_hash(f"merges-{vocab_size}"),
            token_inventory_sha256=_hash(f"token-inventory-{vocab_size}"),
            reserved_inventory_sha256=_hash("reserved-inventory"),
            pretokenizer_regex_sha256=_hash("pretokenizer-regex"),
            fit_stream_sha256=corpus.training_stream_sha256,
            full_corpus_manifest_sha256=corpus.full_corpus_manifest_sha256,
            double_fit_receipt_sha256=_hash(f"double-fit-{vocab_size}"),
            byte_round_trip_receipt_sha256=_hash("byte-round-trip"),
            token_inventory_count=vocab_size,
        )
        for vocab_size in GTOK_VOCABULARY_ARMS
    )


def _strata(corpus: FrozenScreenCorpusV2, bpb: float) -> tuple[StratumNllReceipt, ...]:
    counts = dict(corpus.heldout_denominator_signature)
    return tuple(
        StratumNllReceipt(
            stratum=stratum,
            nll_nats=bpb * math.log(2.0) * counts[stratum],
            raw_byte_count=counts[stratum],
        )
        for stratum in GTOK_STRATA
    )


def _observations(
    corpus: FrozenScreenCorpusV2, terminal_bpb: float
) -> tuple[BpbMilestoneReceiptV2, ...]:
    terminal = corpus.training_realized_bytes
    return (
        BpbMilestoneReceiptV2(
            label="after_1b",
            optimizer_step=10,
            previous_training_raw_bytes=GTOK_FIRST_BOUNDARY_BYTES - 100_000,
            training_raw_bytes=GTOK_FIRST_BOUNDARY_BYTES + 100_000,
            heldout_stream_sha256=corpus.heldout_stream_sha256,
            strata=_strata(corpus, terminal_bpb + 0.20),
        ),
        BpbMilestoneReceiptV2(
            label="after_2b",
            optimizer_step=20,
            previous_training_raw_bytes=GTOK_SECOND_BOUNDARY_BYTES - 100_000,
            training_raw_bytes=GTOK_SECOND_BOUNDARY_BYTES + 100_000,
            heldout_stream_sha256=corpus.heldout_stream_sha256,
            strata=_strata(corpus, terminal_bpb + 0.10),
        ),
        BpbMilestoneReceiptV2(
            label="terminal_realized_T",
            optimizer_step=40,
            previous_training_raw_bytes=GTOK_SECOND_BOUNDARY_BYTES + 100_000,
            training_raw_bytes=terminal,
            heldout_stream_sha256=corpus.heldout_stream_sha256,
            strata=_strata(corpus, terminal_bpb),
        ),
    )


def _runs(
    corpus: FrozenScreenCorpusV2,
    tokenizers: tuple[TokenizerArmReceiptV2, ...],
    terminal_bpbs: dict[tuple[int, int], float] | None = None,
) -> tuple[GTokRunReceiptV2, ...]:
    values = DEFAULT_TERMINAL_BPBS if terminal_bpbs is None else terminal_bpbs
    by_vocab = {item.vocab_size: item for item in tokenizers}
    return tuple(
        GTokRunReceiptV2(
            vocab_size=vocab_size,
            seed=seed,
            frozen_screen_corpus_sha256=corpus.receipt_sha256,
            tokenizer_receipt_sha256=by_vocab[vocab_size].receipt_sha256,
            initialization_recipe_sha256=_hash("initialization-recipe"),
            initialization_seed=10_000 + seed,
            shared_initial_state_sha256=_hash(f"shared-state-{seed}"),
            data_order_seed=20_000 + seed,
            data_order_sha256=_hash(f"data-order-{seed}"),
            compute_attempt_id=f"base-{vocab_size}-{seed}",
            measured_a100_microseconds=100_000_000,
            measured_flops=1_000_000_000_000 + vocab_size,
            optimizer=a1_flat_adamw_recipe(),
            observations=_observations(corpus, values[(vocab_size, seed)]),
        )
        for vocab_size in GTOK_VOCABULARY_ARMS
        for seed in SEEDS
    )


def _preflight(
    scope: str,
    vocabularies: tuple[int, ...],
    *,
    prior_campaign_a100_microseconds: int = 0,
    prior_event_ledger_sha256: str | None = None,
    calibration_a100_microseconds: int = 10_000_000,
) -> PreflightProjectionReceiptV2:
    calibrations = tuple(
        ArmCalibrationProjectionV2(
            scope=scope,
            vocab_size=vocab_size,
            calibration_attempt_id=f"{scope}-calibration-{vocab_size}",
            calibration_steps=GTOK_CALIBRATION_MAX_STEPS,
            measured_tokens=1_000,
            measured_a100_microseconds=calibration_a100_microseconds,
            planned_tokens_per_run=10_000,
            projected_run_a100_microseconds=(
                calibration_a100_microseconds * 10
            ),
        )
        for vocab_size in tuple(sorted(vocabularies))
    )
    return PreflightProjectionReceiptV2(
        scope=scope,
        prior_campaign_a100_microseconds=prior_campaign_a100_microseconds,
        prior_event_ledger_sha256=prior_event_ledger_sha256,
        calibrations=calibrations,
        projected_campaign_a100_microseconds=(
            prior_campaign_a100_microseconds
            + sum(item.projected_scope_a100_microseconds for item in calibrations)
        ),
    )


def _campaign(
    preflight: PreflightProjectionReceiptV2,
    selected_rows,
    *,
    predecessor_campaign_sha256: str | None,
    event_ledger_label: str,
    extra_attempts: tuple[ComputeAttemptReceiptV2, ...] = (),
) -> CampaignComputeReceiptV2:
    by_vocab = {item.vocab_size: item for item in preflight.calibrations}
    calibrations = tuple(
        ComputeAttemptReceiptV2(
            attempt_id=item.calibration_attempt_id,
            scope=preflight.scope,
            kind="calibration",
            vocab_size=item.vocab_size,
            seed=None,
            consumed_a100_microseconds=item.measured_a100_microseconds,
            status="completed",
            calibration_projection_sha256=item.receipt_sha256,
            projected_run_a100_microseconds=(
                item.projected_run_a100_microseconds
            ),
            watchdog_limit_a100_microseconds=(
                2 * item.projected_run_a100_microseconds
            ),
        )
        for item in preflight.calibrations
    )
    selected_attempts = tuple(
        ComputeAttemptReceiptV2(
            attempt_id=row.compute_attempt_id,
            scope=preflight.scope,
            kind="full_run",
            vocab_size=row.vocab_size,
            seed=row.seed,
            consumed_a100_microseconds=row.measured_a100_microseconds,
            status="completed",
            calibration_projection_sha256=by_vocab[row.vocab_size].receipt_sha256,
            projected_run_a100_microseconds=(
                by_vocab[row.vocab_size].projected_run_a100_microseconds
            ),
            watchdog_limit_a100_microseconds=(
                2 * by_vocab[row.vocab_size].projected_run_a100_microseconds
            ),
        )
        for row in selected_rows
    )
    attempts = (*calibrations, *selected_attempts, *extra_attempts)
    consumed = preflight.prior_campaign_a100_microseconds + sum(
        item.consumed_a100_microseconds for item in attempts
    )
    event_ledger_sha256 = _hash(event_ledger_label)
    snapshot = RuntimeTripwireSnapshotV2(
        event_ledger_sha256=event_ledger_sha256,
        cumulative_a100_microseconds=consumed,
        pending_attempt_ids=(),
        running_attempt_ids=(),
        hard_abort_attempt_ids=(),
        hard_abort_and_report=False,
        return_to_strategy=False,
    )
    return CampaignComputeReceiptV2(
        scope=preflight.scope,
        predecessor_campaign_sha256=predecessor_campaign_sha256,
        preflight=preflight,
        attempts=attempts,
        event_ledger_sha256=event_ledger_sha256,
        consumed_a100_microseconds=consumed,
        selected_run_a100_microseconds=sum(
            row.measured_a100_microseconds for row in selected_rows
        ),
        runtime_snapshot=snapshot,
        all_attempts_accounted=True,
    )


def _matrix(
    terminal_bpbs: dict[tuple[int, int], float] | None = None,
):
    corpus = _corpus()
    tokenizers = _tokenizers(corpus)
    runs = _runs(corpus, tokenizers, terminal_bpbs)
    preflight = _preflight("base_screen", GTOK_VOCABULARY_ARMS)
    compute = _campaign(
        preflight,
        runs,
        predecessor_campaign_sha256=None,
        event_ledger_label="base-event-ledger",
    )
    return validate_complete_gtok_matrix_v2(
        runs,
        corpus=corpus,
        tokenizers=tokenizers,
        compute=compute,
    )


def _admissibility() -> tuple[VocabularyAdmissibilityReceiptV2, ...]:
    return tuple(
        VocabularyAdmissibilityReceiptV2(
            vocab_size=vocab_size,
            vocabulary_parameter_count=vocab_size * 1_024,
            target_parameter_count=305_800_000,
        )
        for vocab_size in GTOK_VOCABULARY_ARMS
    )


def _confirmation_runs(matrix, selection):
    pair = selection.compute_confirmation_pair
    common_flops = 900_000_000_000
    base_by_key = {(run.vocab_size, run.seed): run for run in matrix.runs}
    rows = []
    for vocab_size in pair:
        for seed in matrix.seeds:
            bpb = 0.90 + (0.02 if seed == 202 else 0.0)
            if vocab_size == pair[1]:
                bpb += 0.05
            rows.append(
                ComputeConfirmationRunV2(
                    vocab_size=vocab_size,
                    seed=seed,
                    base_run_receipt_sha256=base_by_key[(vocab_size, seed)].receipt_sha256,
                    compute_attempt_id=f"confirmation-{vocab_size}-{seed}",
                    common_flop_budget=common_flops,
                    measured_flops=common_flops,
                    heldout_stream_sha256=matrix.corpus.heldout_stream_sha256,
                    strata=_strata(matrix.corpus, bpb),
                    measured_a100_microseconds=50_000_000,
                )
            )
    return tuple(rows)


def _confirmation_compute(matrix, selection, rows) -> CampaignComputeReceiptV2:
    preflight = _preflight(
        "confirmation",
        tuple(sorted(selection.compute_confirmation_pair)),
        prior_campaign_a100_microseconds=(
            matrix.compute.consumed_a100_microseconds
        ),
        prior_event_ledger_sha256=matrix.compute.event_ledger_sha256,
        calibration_a100_microseconds=5_000_000,
    )
    return _campaign(
        preflight,
        rows,
        predecessor_campaign_sha256=matrix.compute.receipt_sha256,
        event_ledger_label="confirmation-event-ledger",
    )


def _confirmation(matrix, selection):
    rows = _confirmation_runs(matrix, selection)
    compute = _confirmation_compute(matrix, selection, rows)
    return validate_compute_confirmation_v2(
        rows,
        matrix=matrix,
        selection=selection,
        compute=compute,
    )


def _basis(matrix, selected_vocab_size: int) -> VocabExtBasisV2:
    tokenizer = next(
        item for item in matrix.tokenizers if item.vocab_size == selected_vocab_size
    )
    return VocabExtBasisV2(
        vocab_size=tokenizer.vocab_size,
        tokenizer_json_sha256=tokenizer.tokenizer_json_sha256,
        merges_sha256=tokenizer.merges_sha256,
        token_inventory_sha256=tokenizer.token_inventory_sha256,
        reserved_inventory_sha256=tokenizer.reserved_inventory_sha256,
        pretokenizer_regex_sha256=tokenizer.pretokenizer_regex_sha256,
        full_corpus_manifest_sha256=matrix.corpus.full_corpus_manifest_sha256,
        screen_submanifest_sha256=matrix.corpus.screen_submanifest_sha256,
    )


def test_v2_authority_is_forward_only_and_legacy_hash_domain_is_unchanged() -> None:
    assert GTOK_V2_AUTHORITY_CHAIN == (
        *GTOK_EXECUTION_AUTHORITY_CHAIN_V2,
        GTOK_AMENDMENT_A2_SHA256,
        GTOK_AMENDMENT_A3_SHA256,
        GTOK_RELEASE_CLOSE_SHA256,
        GTOK_A2_BINDINGS_SHA256,
    )
    payload = {"receipt": 1}
    legacy = execution_authority_v2_bound_sha256("weft1_example_v2", payload)
    current = gtok_v2_bound_sha256("weft1_gtok_v2_example", payload)
    assert current != legacy
    assert len(GTOK_SELECTOR_LITERAL_BINDING_SHA256_V2) == 64
    with pytest.raises(TypeError, match="factory-minted"):
        ValidatedGTokMatrixV2()
    with pytest.raises(TypeError, match="factory-minted"):
        ValidatedComputeConfirmationV2()
    with pytest.raises(TypeError, match="factory-minted"):
        VocabularyFreezeArtifactV2()


def test_matrix_binds_independent_targets_floor_terminal_and_ten_blocks() -> None:
    matrix = _matrix()

    assert matrix.status == "GREEN_COMPLETE_EVIDENCE"
    assert matrix.corpus.training_target_bytes == GTOK_TRAINING_BYTE_BUDGET
    assert matrix.corpus.heldout_target_bytes == GTOK_HELDOUT_BYTE_TARGET
    assert matrix.corpus.training_realized_bytes == GTOK_TRAINING_BYTE_BUDGET - 4
    assert matrix.corpus.heldout_realized_bytes == GTOK_HELDOUT_BYTE_TARGET - 4
    assert len(matrix.runs) == 8
    assert all(run.executing_block_count == 10 for run in matrix.runs)
    assert all(run.model_topology_sha256 == GTOK_PROXY_TOPOLOGY_SHA256 for run in matrix.runs)
    assert all(
        run.terminal.training_raw_bytes == matrix.corpus.training_realized_bytes
        for run in matrix.runs
    )
    assert all(run.checkpoint_retained is False for run in matrix.runs)
    assert all(run.optimizer.muon_enabled is False for run in matrix.runs)
    assert all(run.optimizer == a1_flat_adamw_recipe() for run in matrix.runs)
    assert matrix.runs[0].optimizer.hyperparameters == (
        ("betas", (0.9, 0.95)),
        ("eps", 1e-8),
        ("gradient_clip_norm", 1.0),
        ("learning_rate", 3e-4),
        ("weight_decay", 0.1),
    )
    assert matrix.runs[0].optimizer.schedule == (
        ("batch_sequence_length", 2_048),
        ("batch_sequences", 256),
        ("compute_dtype", "bfloat16"),
        ("decay", "cosine"),
        ("final_learning_rate_fraction", Fraction(1, 10)),
        ("loss_reduction_dtype", "float32"),
        ("master_weight_dtype", "float32"),
        ("warmup_fraction", Fraction(1, 100)),
    )
    first = matrix.runs[0].observations
    assert first[0].previous_training_raw_bytes < 1_000_000_000 <= first[0].training_raw_bytes
    assert first[1].previous_training_raw_bytes < 2_000_000_000 <= first[1].training_raw_bytes


def test_matrix_rejects_target_instead_of_realized_terminal_and_wrong_optimizer() -> None:
    matrix = _matrix()
    runs = matrix.runs
    wrong_terminal = replace(
        runs[0].terminal,
        training_raw_bytes=GTOK_TRAINING_BYTE_BUDGET,
    )
    wrong_run = replace(runs[0], observations=(*runs[0].observations[:-1], wrong_terminal))
    with pytest.raises(ValueError, match="terminal milestone must equal realized T"):
        validate_complete_gtok_matrix_v2(
            (wrong_run, *runs[1:]),
            corpus=matrix.corpus,
            tokenizers=matrix.tokenizers,
            compute=matrix.compute,
        )

    recipe = a1_flat_adamw_recipe()
    changed = FlatAdamWRecipe(
        hyperparameters=tuple(
            (name, 1e-4 if name == "learning_rate" else value)
            for name, value in recipe.hyperparameters
        ),
        schedule=recipe.schedule,
    )
    with pytest.raises(ValueError, match="exact flat A1 AdamW"):
        replace(runs[0], optimizer=changed)
    with pytest.raises(ValueError, match="retain model checkpoints"):
        replace(runs[0], checkpoint_retained=True)


def test_selector_requires_agreed_strict_seed_order_and_echoes_pairwise_math() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())

    assert selection.agreed_strict_terminal_order == (49_152, 32_768, 24_576, 16_384)
    assert selection.compute_confirmation_pair == (49_152, 32_768)
    assert selection.selected_vocab_size == 49_152
    assert len(selection.comparisons) == 3
    for index, comparison in enumerate(selection.comparisons):
        assert comparison.comparison_index == index
        assert comparison.metric == GTOK_TERMINAL_METRIC
        assert comparison.budget == GTOK_TERMINAL_BUDGET
        expected = math.sqrt(
            (
                comparison.incumbent_sample_sd**2
                + comparison.challenger_sample_sd**2
            )
            / 2.0
        )
        assert comparison.s_hat == expected
        assert comparison.two_s_hat == 2.0 * expected
        assert comparison.three_s_hat == 3.0 * expected
        assert comparison.displaced is (
            comparison.delta_bpb > comparison.three_s_hat
        )


def test_asymmetric_band_retains_smaller_arm_and_reports_2s_diagnostic() -> None:
    values = {
        (16_384, 101): 1.30,
        (16_384, 202): 1.34,
        (24_576, 101): 1.20,
        (24_576, 202): 1.24,
        (32_768, 101): 1.10,
        (32_768, 202): 1.14,
        (49_152, 101): 1.085,
        (49_152, 202): 1.125,
    }
    selection = select_vocabulary_v2(_matrix(values), admissibility=_admissibility())

    assert selection.agreed_strict_terminal_order[0] == 49_152
    assert selection.selected_vocab_size == 32_768
    final = selection.comparisons[-1]
    assert final.challenger_vocab == 49_152
    assert final.tie_diagnostic is True
    assert final.displaced is False
    assert final.incumbent_vocab_after == 32_768


def test_seed_order_split_and_exact_tie_stop_before_pooling() -> None:
    split = dict(DEFAULT_TERMINAL_BPBS)
    split[(49_152, 202)] = 1.03
    with pytest.raises(GTokV2Stop, match="total orders disagree"):
        select_vocabulary_v2(_matrix(split), admissibility=_admissibility())

    tied = dict(DEFAULT_TERMINAL_BPBS)
    tied[(49_152, 202)] = tied[(32_768, 202)]
    with pytest.raises(GTokV2Stop, match="not strict"):
        select_vocabulary_v2(_matrix(tied), admissibility=_admissibility())


def test_selector_mutations_fail_for_order_pair_metric_budget_estimator_and_strictness() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    first = selection.comparisons[0]

    changed_order = replace(
        selection,
        agreed_strict_terminal_order=tuple(reversed(selection.agreed_strict_terminal_order)),
    )
    with pytest.raises(ValueError, match="deterministic recomputation"):
        validate_selection_receipt_v2(changed_order, matrix=matrix)

    with pytest.raises(ValueError, match="wrong arm pair|incumbent transition"):
        replace(first, challenger_vocab=32_768)
    with pytest.raises(ValueError, match="terminal pooled BPB"):
        replace(first, metric="mean_validation_loss")
    with pytest.raises(ValueError, match="terminal pooled BPB"):
        replace(first, budget="target_4b")
    with pytest.raises(ValueError, match="pairwise equal-n pooled"):
        replace(
            first,
            s_hat=first.s_hat * 2.0,
            two_s_hat=first.two_s_hat * 2.0,
            three_s_hat=first.three_s_hat * 2.0,
        )
    with pytest.raises(ValueError, match="operator must remain strict"):
        replace(first, displacement_operator=">=")
    with pytest.raises(ValueError, match="preserve traversal order"):
        replace(selection, comparisons=tuple(reversed(selection.comparisons)))

    exact_boundary = SelectionComparisonV2(
        comparison_index=0,
        incumbent_vocab_before=16_384,
        challenger_vocab=24_576,
        incumbent_mean_bpb=4.0,
        challenger_mean_bpb=1.0,
        incumbent_sample_sd=1.0,
        challenger_sample_sd=1.0,
        s_hat=1.0,
        delta_bpb=3.0,
        two_s_hat=2.0,
        three_s_hat=3.0,
        tie_diagnostic=False,
        displaced=False,
        incumbent_vocab_after=16_384,
    )
    with pytest.raises(ValueError, match="strictly >"):
        replace(
            exact_boundary,
            displaced=True,
            incumbent_vocab_after=24_576,
        )


def test_confirmation_uses_registered_pair_equal_flops_and_stops_on_reversal() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    confirmation = _confirmation(matrix, selection)

    assert confirmation.status == "GREEN_NO_REVERSAL"
    assert confirmation.pair == selection.compute_confirmation_pair
    assert len({run.measured_flops for run in confirmation.runs}) == 1

    rows = _confirmation_runs(matrix, selection)
    reversed_row = next(
        run
        for run in rows
        if (run.vocab_size, run.seed)
        == (selection.compute_confirmation_pair[0], matrix.seeds[0])
    )
    changed = replace(reversed_row, strata=_strata(matrix.corpus, 1.20))
    mutated = tuple(changed if run is reversed_row else run for run in rows)
    compute = _confirmation_compute(matrix, selection, mutated)
    with pytest.raises(GTokV2Stop, match="reversed or tied"):
        validate_compute_confirmation_v2(
            mutated,
            matrix=matrix,
            selection=selection,
            compute=compute,
        )


def test_preflight_binds_100_step_calibration_and_halts_before_full_launch() -> None:
    preflight = _preflight("base_screen", GTOK_VOCABULARY_ARMS)
    assert all(
        item.calibration_steps <= GTOK_CALIBRATION_MAX_STEPS
        for item in preflight.calibrations
    )
    with pytest.raises(ValueError, match="may not exceed 100 steps"):
        replace(
            preflight.calibrations[0],
            calibration_steps=GTOK_CALIBRATION_MAX_STEPS + 1,
        )
    with pytest.raises(GTokV2Stop, match="before full-run launch"):
        replace(preflight, full_run_launch_count_at_projection=1)

    over = ArmCalibrationProjectionV2(
        scope="base_screen",
        vocab_size=16_384,
        calibration_attempt_id="over-projection-calibration",
        calibration_steps=1,
        measured_tokens=1,
        measured_a100_microseconds=1,
        planned_tokens_per_run=GTOK_TRIPWIRE_A100_MICROSECONDS,
        projected_run_a100_microseconds=GTOK_TRIPWIRE_A100_MICROSECONDS,
    )
    with pytest.raises(GTokV2Stop, match="halt before full launch"):
        PreflightProjectionReceiptV2(
            scope="base_screen",
            prior_campaign_a100_microseconds=0,
            prior_event_ledger_sha256=None,
            calibrations=(over,),
            projected_campaign_a100_microseconds=(
                over.projected_scope_a100_microseconds
            ),
        )


def test_campaign_charges_calibration_retries_and_enforces_2x_watchdog() -> None:
    corpus = _corpus()
    tokenizers = _tokenizers(corpus)
    runs = _runs(corpus, tokenizers)
    preflight = _preflight("base_screen", GTOK_VOCABULARY_ARMS)
    projection = preflight.calibrations[0]
    retry = ComputeAttemptReceiptV2(
        attempt_id="base-retry",
        scope="base_screen",
        kind="full_run",
        vocab_size=projection.vocab_size,
        seed=SEEDS[0],
        consumed_a100_microseconds=50_000_000,
        status="failed",
        calibration_projection_sha256=projection.receipt_sha256,
        projected_run_a100_microseconds=(
            projection.projected_run_a100_microseconds
        ),
        watchdog_limit_a100_microseconds=(
            2 * projection.projected_run_a100_microseconds
        ),
    )
    campaign = _campaign(
        preflight,
        runs,
        predecessor_campaign_sha256=None,
        event_ledger_label="base-with-retry",
        extra_attempts=(retry,),
    )
    assert campaign.consumed_a100_microseconds == sum(
        item.consumed_a100_microseconds for item in campaign.attempts
    )
    assert retry in campaign.attempts
    with pytest.raises(ValueError, match="calibration, retries, and all attempts"):
        replace(
            campaign,
            consumed_a100_microseconds=campaign.consumed_a100_microseconds - 1,
        )
    with pytest.raises(ValueError, match="every failed, retried, and completed"):
        replace(campaign, all_attempts_accounted=False)

    limit = 2 * projection.projected_run_a100_microseconds
    with pytest.raises(GTokV2Stop, match="hard watchdog abort"):
        replace(
            retry,
            consumed_a100_microseconds=limit + 1,
            status="completed",
        )
    boundary = replace(
        retry,
        attempt_id="base-boundary",
        consumed_a100_microseconds=limit,
        status="completed",
    )
    assert boundary.hard_abort_issued is False
    aborted = replace(
        retry,
        attempt_id="base-watchdog-abort",
        consumed_a100_microseconds=limit + 1,
        status="aborted_watchdog",
        hard_abort_issued=True,
    )
    assert aborted.hard_abort_issued is True


def test_runtime_crossing_hard_aborts_pending_and_running_work() -> None:
    with pytest.raises(ValueError, match="hard-abort every pending and running"):
        RuntimeTripwireSnapshotV2(
            event_ledger_sha256=_hash("crossed-ledger"),
            cumulative_a100_microseconds=GTOK_TRIPWIRE_A100_MICROSECONDS + 1,
            pending_attempt_ids=("pending",),
            running_attempt_ids=("running",),
            hard_abort_attempt_ids=("running",),
            hard_abort_and_report=True,
            return_to_strategy=True,
        )
    crossed = RuntimeTripwireSnapshotV2(
        event_ledger_sha256=_hash("crossed-ledger"),
        cumulative_a100_microseconds=GTOK_TRIPWIRE_A100_MICROSECONDS + 1,
        pending_attempt_ids=("pending",),
        running_attempt_ids=("running",),
        hard_abort_attempt_ids=("pending", "running"),
        hard_abort_and_report=True,
        return_to_strategy=True,
    )
    with pytest.raises(GTokV2Stop, match="pending/running work hard-aborted"):
        enforce_runtime_tripwire_v2(crossed)


def test_confirmation_compute_extends_base_with_projection_and_full_ledger() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    rows = _confirmation_runs(matrix, selection)
    compute = _confirmation_compute(matrix, selection, rows)
    confirmation = validate_compute_confirmation_v2(
        rows,
        matrix=matrix,
        selection=selection,
        compute=compute,
    )
    assert compute.predecessor_campaign_sha256 == matrix.compute.receipt_sha256
    assert (
        compute.preflight.prior_campaign_a100_microseconds
        == matrix.compute.consumed_a100_microseconds
    )
    assert confirmation.compute_campaign_receipt_sha256 == compute.receipt_sha256
    confirmation_scope_projection = sum(
        item.projected_scope_a100_microseconds
        for item in compute.preflight.calibrations
    )
    with pytest.raises(GTokV2Stop, match="halt before full launch"):
        replace(
            compute.preflight,
            prior_campaign_a100_microseconds=(
                GTOK_TRIPWIRE_A100_MICROSECONDS
            ),
            projected_campaign_a100_microseconds=(
                GTOK_TRIPWIRE_A100_MICROSECONDS
                + confirmation_scope_projection
            ),
        )
    with pytest.raises(ValueError, match="does not extend the base"):
        validate_compute_confirmation_v2(
            rows,
            matrix=matrix,
            selection=selection,
            compute=replace(
                compute,
                predecessor_campaign_sha256=_hash("wrong-predecessor"),
            ),
        )


def test_freeze_minter_requires_green_matrix_confirmation_and_exact_vocab_ext_basis() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    confirmation = _confirmation(matrix, selection)
    basis = _basis(matrix, selection.selected_vocab_size)

    artifact = mint_vocabulary_freeze_v2(
        matrix=matrix,
        selection=selection,
        confirmation=confirmation,
        basis=basis,
    )
    assert artifact.status == "FROZEN_GREEN"
    assert artifact.selected_vocab_size == selection.selected_vocab_size
    assert artifact.vocab_ext_basis.existing_token_ids_never_renumbered is True
    assert len(artifact.receipt_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        artifact.selected_vocab_size = 16_384  # type: ignore[misc]

    with pytest.raises(TypeError, match="factory-validated compute confirmation"):
        mint_vocabulary_freeze_v2(
            matrix=matrix,
            selection=selection,
            confirmation=None,  # type: ignore[arg-type]
            basis=basis,
        )
    with pytest.raises(ValueError, match="selected tokenizer basis"):
        mint_vocabulary_freeze_v2(
            matrix=matrix,
            selection=selection,
            confirmation=confirmation,
            basis=replace(basis, merges_sha256=_hash("wrong-merges")),
        )
    with pytest.raises(ValueError, match="never renumber"):
        replace(basis, existing_token_ids_never_renumbered=False)


def test_admissibility_is_exact_rung_b_fraction_and_guards_confirmation_pair() -> None:
    matrix = _matrix()
    rows = list(_admissibility())
    rows[-1] = VocabularyAdmissibilityReceiptV2(
        vocab_size=49_152,
        vocabulary_parameter_count=60_000_001,
        target_parameter_count=300_000_000,
    )
    assert rows[-1].fraction > Fraction(1, 5)
    with pytest.raises(GTokV2Stop, match="confirmation arm fails"):
        select_vocabulary_v2(matrix, admissibility=tuple(rows))
