from __future__ import annotations

from dataclasses import asdict, FrozenInstanceError, replace
from fractions import Fraction
import math

import pytest

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
    canonical_json_bytes,
    canonical_sha256,
    execution_authority_v2_bound_sha256,
    sha256_bytes,
)
from training.weft1_gtok_campaign_v2 import (
    confirmation_seed_rows_for_vocabulary_v2,
)
from training.weft1_corpus_materialize_a3 import ConfirmationConsumerOrderV4
from training.weft1_gtok_confirmation_v2 import (
    BaseRunFlopEvidenceV2,
    BaseStepFlopV2,
    CONFIRMATION_BINDING_SHA256_V2,
    ConfirmationArmFlopPlanV2,
    ConfirmationExecutionPlanV2,
)
from training.weft1_gtok_training_v2 import (
    ConfirmationTrainingPlanV2,
    FLOP_BINDING_SHA256_V2,
)
from training.weft1_gtok_v2_contract import (
    A2FirstFitGroupReceiptV2,
    A2FirstFitScreenReceiptV2,
    ArmCalibrationProjectionV2,
    BpbMilestoneReceiptV2,
    CampaignComputeReceiptV2,
    ConfirmationArmFlopSourceEnvelopeV2,
    ConfirmationAttemptLaunchEnvelopeV2,
    ConfirmationBaseRunFlopSourceEnvelopeV2,
    ComputeAttemptReceiptV2,
    ComputeConfirmationRunV2,
    ConfirmationEvidenceClosureV2,
    ConfirmationExecutionPlanEnvelopeV2,
    ConfirmationFreshEvidenceJoinV2,
    ConfirmationLifecycleEventEvidenceV2,
    ConfirmationOrderEnvelopeV2,
    ConfirmationRetryArtifactEnvelopeV2,
    FrozenScreenCorpusV2,
    GTOK_A2_BINDINGS_SHA256,
    GTOK_AMENDMENT_A2_SHA256,
    GTOK_AMENDMENT_A3_SHA256,
    GTOK_CALIBRATION_MAX_STEPS,
    GTOK_CONFIRMATION_SEMANTICS_SHA256,
    GTOK_FIRST_BOUNDARY_BYTES,
    GTOK_RELEASE_CLOSE_SHA256,
    GTOK_RHO_BPB_DECIMAL_PLACES,
    GTOK_RHO_BPB_SCALE,
    GTOK_SECOND_BOUNDARY_BYTES,
    GTOK_SELECTION_CONFIRMATION_AUTHORITY_CHAIN,
    GTOK_SELECTION_CONFIRMATION_AUTHORITY_SHA256,
    GTOK_SELECTION_CONFIRMATION_LITERAL_BINDING_SHA256_V2,
    GTOK_SEMANTICS_AMENDMENT_S1_SHA256,
    GTOK_SEMANTICS_AMENDMENT_S2_SHA256,
    GTOK_SELECTOR_LITERAL_BINDING_SHA256_V2,
    GTOK_TERMINAL_BUDGET,
    GTOK_TERMINAL_METRIC,
    GTOK_TRIPWIRE_A100_MICROSECONDS,
    GTOK_V2_AUTHORITY_CHAIN,
    GTokRunReceiptV2,
    GTokV2Stop,
    PrecalibrationReplayAttemptReceiptV2,
    PreflightProjectionReceiptV2,
    RuntimeTripwireSnapshotV2,
    SelectionComparisonV2,
    TokenizerArmReceiptV2,
    VocabExtBasisV2,
    ValidatedComputeConfirmationV2,
    ValidatedGTokMatrixV2,
    VocabularyAdmissibilityReceiptV2,
    VocabularyFreezeArtifactV2,
    compute_event_ledger_sha256_v2,
    enforce_runtime_tripwire_v2,
    gtok_v2_bound_sha256,
    mint_vocabulary_freeze_v2,
    select_vocabulary_v2,
    validate_complete_gtok_matrix_v2,
    validate_compute_confirmation_v2,
    validate_selection_receipt_v2,
)


SEEDS = (101, 202)
_BASE_OPTIMIZER_STEPS = 400
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


def _groups(
    stream: str, *, shortfall: int = 1
) -> tuple[A2FirstFitGroupReceiptV2, ...]:
    targets = dict(
        GTOK_SCREEN_TRAIN_STRATUM_TARGETS
        if stream == "T"
        else GTOK_SCREEN_HELDOUT_STRATUM_TARGETS
    )
    return tuple(
        A2FirstFitGroupReceiptV2(
            stream=stream,
            stratum=stratum,
            target_bytes=targets[stratum],
            realized_bytes=targets[stratum] - shortfall,
            deficit_bytes=shortfall,
            document_count=10,
            ordered_raw_content_ids_sha256=_hash(f"{stream}-{stratum}-order"),
        )
        for stratum in GTOK_STRATA
    )


def _corpus() -> FrozenScreenCorpusV2:
    first_fit = A2FirstFitScreenReceiptV2(
        groups=(*_groups("T"), *_groups("H")),
        training_framed_stream_sha256=_hash("training-stream"),
        heldout_framed_stream_sha256=_hash("heldout-stream"),
        document_overlap_count=0,
        cluster_overlap_count=0,
    )
    return FrozenScreenCorpusV2(
        full_corpus_manifest_sha256=_hash("full-corpus-manifest"),
        screen_submanifest_sha256=_hash("screen-submanifest"),
        d6_physical_evidence_sha256=_hash("physical-d6"),
        corpus_freeze_receipt_sha256=_hash("p-b-freeze"),
        d1_d6_gate_bundle_sha256=_hash("d1-d6"),
        decontamination_receipt_sha256=_hash("decon"),
        first_fit=first_fit,
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
            optimizer_step=100,
            previous_training_raw_bytes=GTOK_FIRST_BOUNDARY_BYTES - 100_000,
            training_raw_bytes=GTOK_FIRST_BOUNDARY_BYTES + 100_000,
            heldout_stream_sha256=corpus.heldout_stream_sha256,
            strata=_strata(corpus, terminal_bpb + 0.20),
        ),
        BpbMilestoneReceiptV2(
            label="after_2b",
            optimizer_step=200,
            previous_training_raw_bytes=GTOK_SECOND_BOUNDARY_BYTES - 100_000,
            training_raw_bytes=GTOK_SECOND_BOUNDARY_BYTES + 100_000,
            heldout_stream_sha256=corpus.heldout_stream_sha256,
            strata=_strata(corpus, terminal_bpb + 0.10),
        ),
        BpbMilestoneReceiptV2(
            label="terminal_realized_T",
            optimizer_step=_BASE_OPTIMIZER_STEPS,
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
            training_runtime_receipt_sha256=_hash("training-runtime"),
            code_closure_receipt_sha256=_hash("code-closure"),
            compute_attempt_id=f"base-{vocab_size}-{seed}",
            measured_a100_microseconds=100_000_000,
            measured_flops=1_000_000_000_000 + vocab_size,
            optimizer=a1_flat_adamw_recipe(),
            observations=_observations(corpus, values[(vocab_size, seed)]),
            stream_bytes=corpus.training_realized_bytes,
            stream_docs=10,
            stream_tokens=_BASE_OPTIMIZER_STEPS * 256 * 2_048,
            trained_tokens=_BASE_OPTIMIZER_STEPS * 256 * 2_048,
            dropped_tokens=0,
            trained_bytes=corpus.training_realized_bytes,
            dropped_bytes=0,
            trained_docs_full=10,
            dropped_docs=0,
        )
        for vocab_size in GTOK_VOCABULARY_ARMS
        for seed in SEEDS
    )


def test_run_receipt_binds_terminal_bpb_to_exact_trained_prefix() -> None:
    corpus = _corpus()
    run = _runs(corpus, _tokenizers(corpus))[0]
    trained_bytes = corpus.training_realized_bytes - 1
    observations = (
        *run.observations[:-1],
        replace(run.terminal, training_raw_bytes=trained_bytes),
    )
    exact = replace(
        run,
        observations=observations,
        stream_bytes=corpus.training_realized_bytes,
        stream_docs=3,
        stream_tokens=_BASE_OPTIMIZER_STEPS * 256 * 2_048 + 10,
        trained_tokens=_BASE_OPTIMIZER_STEPS * 256 * 2_048,
        dropped_tokens=10,
        trained_bytes=trained_bytes,
        dropped_bytes=1,
        trained_docs_full=1,
        boundary_doc_id="a" * 40,
        boundary_doc_consumed_tokens=5,
        dropped_docs=1,
    )
    assert exact.has_exact_stream_accounting
    assert exact.terminal.training_raw_bytes == exact.trained_bytes
    assert exact.dropped_token_fraction == pytest.approx(
        10 / (_BASE_OPTIMIZER_STEPS * 256 * 2_048 + 10)
    )
    with pytest.raises(ValueError, match="trained prefix"):
        replace(
            exact,
            observations=run.observations,
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
            measured_tokens=80 * 256 * 2_048,
            measured_a100_microseconds=calibration_a100_microseconds,
            planned_tokens_per_run=_BASE_OPTIMIZER_STEPS * 256 * 2_048,
            projected_run_a100_microseconds=(
                calibration_a100_microseconds * 5
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

    def selected_projection(row) -> int:
        projection = by_vocab[row.vocab_size]
        if preflight.scope != "confirmation":
            return projection.projected_run_a100_microseconds
        training_projection = (
            projection.measured_a100_microseconds * row.trained_tokens
            + projection.measured_tokens
            - 1
        ) // projection.measured_tokens
        return training_projection + (
            projection.measured_heldout_evaluation_a100_microseconds
            * projection.heldout_evaluations_per_full_run
        ) + (
            projection.measured_output_surface_a100_microseconds
            * projection.output_surface_benchmarks_per_full_run
        )

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
        if item.projection_source != "completed_base_calibration"
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
            projected_run_a100_microseconds=selected_projection(row),
            watchdog_limit_a100_microseconds=2 * selected_projection(row),
            execution_plan_sha256=(
                row.execution_plan_sha256
                if preflight.scope == "confirmation"
                else None
            ),
            planned_compute_token_slots=(
                row.trained_tokens
                if preflight.scope == "confirmation"
                else None
            ),
        )
        for row in selected_rows
    )
    attempts = (*calibrations, *selected_attempts, *extra_attempts)
    consumed = preflight.prior_campaign_a100_microseconds + sum(
        item.consumed_a100_microseconds for item in attempts
    )
    del event_ledger_label
    event_ledger_sha256 = compute_event_ledger_sha256_v2(attempts)
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


def _synthetic_flop_ledger(run) -> dict[str, object]:
    optimizer_steps = run.terminal.optimizer_step
    initial_flops = run.measured_flops - 2 * (optimizer_steps - 1)
    return {
        "shapes": (
            {
                "batch_rows": 256,
                "sequence_length": 2_048,
                "optimizer_phase": "initial",
                "occurrences": 1,
                "profiler_rows": (
                    {
                        "operator": "synthetic.initial",
                        "flops_per_occurrence": initial_flops - 1,
                    },
                ),
                "unsupported_rows": (
                    {
                        "family": "synthetic.initial.unsupported",
                        "flops_per_occurrence": 1,
                        "derivation": "synthetic=1",
                    },
                ),
                "zero_flop_profiler_operators": (),
            },
            {
                "batch_rows": 256,
                "sequence_length": 2_048,
                "optimizer_phase": "steady",
                "occurrences": optimizer_steps - 1,
                "profiler_rows": (
                    {
                        "operator": "synthetic.steady",
                        "flops_per_occurrence": 1,
                    },
                ),
                "unsupported_rows": (
                    {
                        "family": "synthetic.steady.unsupported",
                        "flops_per_occurrence": 1,
                        "derivation": "synthetic=1",
                    },
                ),
                "zero_flop_profiler_operators": (),
            },
        ),
        "optimizer_steps": optimizer_steps,
        "compute_token_slots": run.trained_tokens,
        "profiler_with_flops": True,
        "flop_binding_sha256": FLOP_BINDING_SHA256_V2,
    }


def _base_flop_source(matrix, vocab_size: int, seed: int):
    run = next(
        row for row in matrix.runs
        if row.vocab_size == vocab_size and row.seed == seed
    )
    ledger = _synthetic_flop_ledger(run)
    ledger_sha256 = canonical_sha256(ledger)
    initial_flops = run.measured_flops - 2 * (run.terminal.optimizer_step - 1)
    steps = (
        BaseStepFlopV2(
            optimizer_step=1,
            batch_rows=256,
            sequence_length=2_048,
            optimizer_phase="initial",
            measured_flops=initial_flops,
        ),
        *tuple(
            BaseStepFlopV2(
                optimizer_step=optimizer_step,
                batch_rows=256,
                sequence_length=2_048,
                optimizer_phase="steady",
                measured_flops=2,
            )
            for optimizer_step in range(2, run.terminal.optimizer_step + 1)
        ),
    )
    evidence = BaseRunFlopEvidenceV2(
        vocab_size=vocab_size,
        seed=seed,
        base_run_receipt_sha256=run.receipt_sha256,
        base_compute_attempt_id=run.compute_attempt_id,
        flop_ledger_sha256=ledger_sha256,
        steps=steps,
        measured_flops=run.measured_flops,
    )
    return ConfirmationBaseRunFlopSourceEnvelopeV2(
        flop_ledger_payload=ledger,
        flop_ledger_receipt_sha256=ledger_sha256,
        base_flop_evidence_payload=asdict(evidence),
        base_flop_evidence_receipt_sha256=evidence.receipt_sha256,
    )


def _arm_flop_source(matrix, selection, vocab_size: int):
    sources = tuple(
        _base_flop_source(matrix, vocab_size, seed) for seed in matrix.seeds
    )
    base_flops = tuple(
        int(row.base_flop_evidence_payload["measured_flops"])
        for row in sources
    )
    base_by_key = {(row.vocab_size, row.seed): row for row in matrix.runs}
    pair_means = {
        arm: sum(base_by_key[(arm, seed)].measured_flops for seed in matrix.seeds) // 2
        for arm in selection.compute_confirmation_pair
    }
    target = min(pair_means.values())
    byte_steps = base_by_key[(vocab_size, matrix.seeds[0])].terminal.optimizer_step
    arm_plan = ConfirmationArmFlopPlanV2(
        vocab_size=vocab_size,
        seeds=matrix.seeds,
        base_flops=(base_flops[0], base_flops[1]),
        base_flop_evidence_sha256s=(
            sources[0].base_flop_evidence_receipt_sha256,
            sources[1].base_flop_evidence_receipt_sha256,
        ),
        byte_matched_optimizer_steps=byte_steps,
        arm_mean_flops=pair_means[vocab_size],
        target_flops=target,
        planned_optimizer_steps=(target * byte_steps) // pair_means[vocab_size],
    )
    return ConfirmationArmFlopSourceEnvelopeV2(
        arm_plan_payload=asdict(arm_plan),
        arm_plan_receipt_sha256=arm_plan.receipt_sha256,
        base_runs=(sources[0], sources[1]),
    )


def _confirmation_plan(matrix, selection, seed_row):
    pair = selection.compute_confirmation_pair
    base_by_key = {(run.vocab_size, run.seed): run for run in matrix.runs}
    arm_mean_flops = {
        vocab_size: sum(base_by_key[(vocab_size, seed)].measured_flops for seed in matrix.seeds)
        // 2
        for vocab_size in pair
    }
    common_flops = min(arm_mean_flops.values())
    reused_vocab = min(pair, key=lambda value: arm_mean_flops[value])
    fresh_vocab = next(value for value in pair if value != reused_vocab)
    base_seed = matrix.seeds[seed_row.seed_slot]
    base_run = base_by_key[(fresh_vocab, base_seed)]
    order = ConfirmationConsumerOrderV4(
        confirmation_run_seed=seed_row.run_seed,
        data_order_seed=seed_row.data_order_seed,
        physical_d6_evidence_sha256=matrix.corpus.d6_physical_evidence_sha256,
        document_multiset_sha256=_hash(f"confirm-document-multiset-{fresh_vocab}"),
        ordered_raw_content_ids_sha256=_hash(
            f"confirm-data-order-{fresh_vocab}-{seed_row.seed_slot}"
        ),
        framed_payload_sha256=_hash(
            f"confirm-framed-payload-{fresh_vocab}-{seed_row.seed_slot}"
        ),
        document_count=10,
        retained_text_bytes=matrix.corpus.training_realized_bytes,
    )
    order_receipt_sha256 = order.receipt_sha256
    optimizer_steps = (
        common_flops * base_run.terminal.optimizer_step
    ) // arm_mean_flops[fresh_vocab]
    trained_tokens = optimizer_steps * 256 * 2_048
    training_plan = ConfirmationTrainingPlanV2(
        confirmation_order_receipt_sha256=order_receipt_sha256,
        optimizer_steps=optimizer_steps,
        global_batch_sequences=256,
        sequence_length=2_048,
        compute_token_slots=trained_tokens,
        valid_prediction_count=1,
        trained_bytes=matrix.corpus.training_realized_bytes - 1,
        trained_tokens=trained_tokens,
        trained_docs_full=9,
        boundary_doc_id=None,
        boundary_doc_consumed_tokens=None,
        stream_bytes=matrix.corpus.training_realized_bytes,
        stream_tokens=int(base_run.stream_tokens),
        stream_docs=10,
        dropped_bytes=1,
        dropped_tokens=int(base_run.stream_tokens) - trained_tokens,
        dropped_docs=1,
        packed_stream_sha256=_hash(
            f"confirm-packed-{fresh_vocab}-{seed_row.seed_slot}"
        ),
        calibration_prefix_compute_token_slots=100 * 256 * 2_048,
        calibration_prefix_valid_prediction_count=1,
        calibration_prefix_realized_raw_bytes=GTOK_FIRST_BOUNDARY_BYTES,
        calibration_prefix_document_count=1,
        calibration_prefix_packed_stream_sha256=_hash(
            f"confirm-calibration-packed-{fresh_vocab}-{seed_row.seed_slot}"
        ),
        bpb_checkpoint_steps=(100, 200, optimizer_steps),
    )
    return ConfirmationExecutionPlanV2(
        vocab_size=fresh_vocab,
        seed_slot=seed_row.seed_slot,
        registry_key=seed_row.registry_key,
        seed=seed_row.run_seed,
        initialization_seed=seed_row.initialization_seed,
        data_order_seed=seed_row.data_order_seed,
        data_order_sha256=_hash(
            f"confirm-data-order-{fresh_vocab}-{seed_row.seed_slot}"
        ),
        confirmation_order_receipt_sha256=order_receipt_sha256,
        physical_d6_evidence_sha256=matrix.corpus.d6_physical_evidence_sha256,
        document_multiset_sha256=_hash(f"confirm-document-multiset-{fresh_vocab}"),
        framed_payload_sha256=_hash(
            f"confirm-framed-payload-{fresh_vocab}-{seed_row.seed_slot}"
        ),
        order_document_count=10,
        order_retained_text_bytes=matrix.corpus.training_realized_bytes,
        target_flops=common_flops,
        arm_mean_flops=arm_mean_flops[fresh_vocab],
        byte_matched_optimizer_steps=base_run.terminal.optimizer_step,
        arm_flop_plan_sha256=_arm_flop_source(
            matrix, selection, fresh_vocab
        ).arm_plan_receipt_sha256,
        training_plan=training_plan,
        heldout_evaluation_steps=training_plan.bpb_checkpoint_steps,
    )


def _confirmation_observations(corpus, plan, terminal_bpb):
    return (
        BpbMilestoneReceiptV2(
            label="after_1b",
            optimizer_step=plan.heldout_evaluation_steps[0],
            previous_training_raw_bytes=GTOK_FIRST_BOUNDARY_BYTES - 100_000,
            training_raw_bytes=GTOK_FIRST_BOUNDARY_BYTES + 100_000,
            heldout_stream_sha256=corpus.heldout_stream_sha256,
            strata=_strata(corpus, terminal_bpb + 0.20),
        ),
        BpbMilestoneReceiptV2(
            label="after_2b",
            optimizer_step=plan.heldout_evaluation_steps[1],
            previous_training_raw_bytes=GTOK_SECOND_BOUNDARY_BYTES - 100_000,
            training_raw_bytes=GTOK_SECOND_BOUNDARY_BYTES + 100_000,
            heldout_stream_sha256=corpus.heldout_stream_sha256,
            strata=_strata(corpus, terminal_bpb + 0.10),
        ),
        BpbMilestoneReceiptV2(
            label="terminal_realized_T",
            optimizer_step=plan.heldout_evaluation_steps[2],
            previous_training_raw_bytes=GTOK_SECOND_BOUNDARY_BYTES + 100_000,
            training_raw_bytes=plan.training_plan.trained_bytes,
            heldout_stream_sha256=corpus.heldout_stream_sha256,
            strata=_strata(corpus, terminal_bpb),
        ),
    )


def _revalue_confirmation_observations(corpus, run, terminal_bpb):
    offsets = {"after_1b": 0.20, "after_2b": 0.10, "terminal_realized_T": 0.0}
    return tuple(
        replace(
            observation,
            strata=_strata(corpus, terminal_bpb + offsets[observation.label]),
        )
        for observation in run.observations
    )


def _confirmation_runs(matrix, selection):
    pair = selection.compute_confirmation_pair
    fresh_vocab = next(
        value
        for value in pair
        if value
        != min(
            pair,
            key=lambda vocab_size: sum(
                run.measured_flops
                for run in matrix.runs
                if run.vocab_size == vocab_size
            ),
        )
    )
    rows = []
    for seed_row in confirmation_seed_rows_for_vocabulary_v2(fresh_vocab):
        plan = _confirmation_plan(matrix, selection, seed_row)
        base_run = next(
            run
            for run in matrix.runs
            if run.vocab_size == fresh_vocab
            and run.seed == matrix.seeds[seed_row.seed_slot]
        )
        bpb = 0.90 + (0.02 if seed_row.seed_slot else 0.0)
        if fresh_vocab == pair[1]:
            bpb += 0.05
        training_plan = plan.training_plan
        rows.append(
            ComputeConfirmationRunV2(
                vocab_size=fresh_vocab,
                seed_slot=seed_row.seed_slot,
                registry_key=seed_row.registry_key,
                seed=seed_row.run_seed,
                initialization_seed=seed_row.initialization_seed,
                data_order_seed=seed_row.data_order_seed,
                data_order_sha256=plan.data_order_sha256,
                confirmation_order_receipt_sha256=(
                    plan.confirmation_order_receipt_sha256
                ),
                physical_d6_evidence_sha256=plan.physical_d6_evidence_sha256,
                document_multiset_sha256=plan.document_multiset_sha256,
                framed_payload_sha256=plan.framed_payload_sha256,
                execution_plan_sha256=plan.receipt_sha256,
                training_plan_sha256=training_plan.receipt_sha256,
                base_run_receipt_sha256=base_run.receipt_sha256,
                compute_attempt_id=f"confirmation-{fresh_vocab}-{seed_row.run_seed}",
                common_flop_budget=plan.target_flops,
                measured_flops=plan.target_flops,
                heldout_stream_sha256=matrix.corpus.heldout_stream_sha256,
                observations=_confirmation_observations(matrix.corpus, plan, bpb),
                measured_a100_microseconds=50_000_000,
                training_runtime_receipt_sha256=_hash("training-runtime"),
                code_closure_receipt_sha256=_hash("code-closure"),
                stream_bytes=training_plan.stream_bytes,
                stream_docs=training_plan.stream_docs,
                stream_tokens=training_plan.stream_tokens,
                trained_tokens=training_plan.trained_tokens,
                dropped_tokens=training_plan.dropped_tokens,
                trained_bytes=training_plan.trained_bytes,
                dropped_bytes=training_plan.dropped_bytes,
                trained_docs_full=training_plan.trained_docs_full,
                boundary_doc_id=training_plan.boundary_doc_id,
                boundary_doc_consumed_tokens=(
                    training_plan.boundary_doc_consumed_tokens
                ),
                dropped_docs=training_plan.dropped_docs,
            )
        )
    return tuple(rows)


def _confirmation_compute(matrix, selection, rows) -> CampaignComputeReceiptV2:
    fresh_vocab = rows[0].vocab_size
    base_projection = next(
        item for item in matrix.compute.preflight.calibrations
        if item.vocab_size == fresh_vocab
    )
    planned_tokens = rows[0].trained_tokens
    training_projection = (
        base_projection.measured_a100_microseconds * planned_tokens
        + base_projection.measured_tokens
        - 1
    ) // base_projection.measured_tokens
    projected_run = training_projection + (
        base_projection.measured_heldout_evaluation_a100_microseconds
        * base_projection.heldout_evaluations_per_full_run
    ) + (
        base_projection.measured_output_surface_a100_microseconds
        * base_projection.output_surface_benchmarks_per_full_run
    )
    inherited = ArmCalibrationProjectionV2(
        scope="confirmation",
        vocab_size=fresh_vocab,
        calibration_attempt_id=f"inherited-base-calibration-v{fresh_vocab}",
        calibration_steps=base_projection.calibration_steps,
        measured_tokens=base_projection.measured_tokens,
        measured_a100_microseconds=base_projection.measured_a100_microseconds,
        planned_tokens_per_run=planned_tokens,
        projected_run_a100_microseconds=projected_run,
        charged_calibration_a100_microseconds=0,
        measured_heldout_evaluation_a100_microseconds=(
            base_projection.measured_heldout_evaluation_a100_microseconds
        ),
        heldout_evaluations_per_full_run=(
            base_projection.heldout_evaluations_per_full_run
        ),
        measured_output_surface_a100_microseconds=(
            base_projection.measured_output_surface_a100_microseconds
        ),
        output_surface_benchmarks_per_full_run=(
            base_projection.output_surface_benchmarks_per_full_run
        ),
        projection_source="completed_base_calibration",
        projection_source_receipt_sha256=base_projection.receipt_sha256,
    )
    preflight = PreflightProjectionReceiptV2(
        scope="confirmation",
        prior_campaign_a100_microseconds=matrix.compute.consumed_a100_microseconds,
        prior_event_ledger_sha256=matrix.compute.event_ledger_sha256,
        calibrations=(inherited,),
        projected_campaign_a100_microseconds=(
            matrix.compute.consumed_a100_microseconds
            + inherited.projected_scope_a100_microseconds
        ),
    )
    return _campaign(
        preflight,
        rows,
        predecessor_campaign_sha256=matrix.compute.receipt_sha256,
        event_ledger_label="confirmation-event-ledger",
    )


def _confirmation_evidence_closure(
    matrix,
    selection,
    compute: CampaignComputeReceiptV2,
    rows: tuple[ComputeConfirmationRunV2, ...],
) -> ConfirmationEvidenceClosureV2:
    joins = []
    lifecycle_events = []
    execution_plans = []
    confirmation_orders = []
    attempt_launches = []
    for run in sorted(rows, key=lambda item: item.seed_slot):
        seed_row = confirmation_seed_rows_for_vocabulary_v2(run.vocab_size)[
            run.seed_slot
        ]
        plan = _confirmation_plan(matrix, selection, seed_row)
        assert plan.receipt_sha256 == run.execution_plan_sha256
        execution_plans.append(
            ConfirmationExecutionPlanEnvelopeV2(
                payload=asdict(plan),
                receipt_sha256=plan.receipt_sha256,
            )
        )
        order = ConfirmationConsumerOrderV4(
            confirmation_run_seed=plan.seed,
            data_order_seed=plan.data_order_seed,
            physical_d6_evidence_sha256=plan.physical_d6_evidence_sha256,
            document_multiset_sha256=plan.document_multiset_sha256,
            ordered_raw_content_ids_sha256=plan.data_order_sha256,
            framed_payload_sha256=plan.framed_payload_sha256,
            document_count=plan.order_document_count,
            retained_text_bytes=plan.order_retained_text_bytes,
        )
        assert order.receipt_sha256 == plan.confirmation_order_receipt_sha256
        confirmation_orders.append(
            ConfirmationOrderEnvelopeV2(
                payload=asdict(order),
                receipt_sha256=order.receipt_sha256,
                physical_sha256=sha256_bytes(
                    canonical_json_bytes(
                        {
                            "binding_sha256": CONFIRMATION_BINDING_SHA256_V2,
                            "payload": asdict(order),
                            "receipt_sha256": order.receipt_sha256,
                            "schema": "weft1_gtok_confirmation_consumer_order_v4",
                        }
                    )
                    + b"\n"
                ),
            )
        )
        attempt = next(
            row for row in compute.attempts
            if row.attempt_id == run.compute_attempt_id
        )
        launch_payload = {
            "attempt_id": attempt.attempt_id,
            "binding_sha256": CONFIRMATION_BINDING_SHA256_V2,
            "calibration_projection_sha256": (
                attempt.calibration_projection_sha256
            ),
            "execution_plan_sha256": plan.receipt_sha256,
            "logical_attempt_id": f"logical-{run.seed_slot}",
            "planned_compute_token_slots": attempt.planned_compute_token_slots,
            "projected_run_a100_microseconds": (
                attempt.projected_run_a100_microseconds
            ),
            "schema": "weft1_gtok_v2_confirmation_attempt_launch",
            "seed": run.seed,
            "vocab_size": run.vocab_size,
            "watchdog_limit_a100_microseconds": (
                attempt.watchdog_limit_a100_microseconds
            ),
        }
        attempt_launches.append(
            ConfirmationAttemptLaunchEnvelopeV2(
                payload=launch_payload,
                physical_sha256=sha256_bytes(
                    canonical_json_bytes(launch_payload) + b"\n"
                ),
            )
        )
        optimizer_steps = run.terminal.optimizer_step
        initial_flops = run.measured_flops - 2 * (optimizer_steps - 1)
        ledger = {
            "shapes": (
                {
                    "batch_rows": 256,
                    "sequence_length": 2_048,
                    "optimizer_phase": "initial",
                    "occurrences": 1,
                    "profiler_rows": (
                        {
                            "operator": "synthetic.initial",
                            "flops_per_occurrence": initial_flops - 1,
                        },
                    ),
                    "unsupported_rows": (
                        {
                            "family": "synthetic.initial.unsupported",
                            "flops_per_occurrence": 1,
                            "derivation": "synthetic=1",
                        },
                    ),
                    "zero_flop_profiler_operators": (),
                },
                {
                    "batch_rows": 256,
                    "sequence_length": 2_048,
                    "optimizer_phase": "steady",
                    "occurrences": optimizer_steps - 1,
                    "profiler_rows": (
                        {
                            "operator": "synthetic.steady",
                            "flops_per_occurrence": 1,
                        },
                    ),
                    "unsupported_rows": (
                        {
                            "family": "synthetic.steady.unsupported",
                            "flops_per_occurrence": 1,
                            "derivation": "synthetic=1",
                        },
                    ),
                    "zero_flop_profiler_operators": (),
                },
            ),
            "optimizer_steps": optimizer_steps,
            "compute_token_slots": run.trained_tokens,
            "profiler_with_flops": True,
            "flop_binding_sha256": FLOP_BINDING_SHA256_V2,
        }
        ledger_sha256 = canonical_sha256(ledger)
        attributed_step_flops = (
            plan.arm_mean_flops + plan.byte_matched_optimizer_steps // 2
        ) // plan.byte_matched_optimizer_steps
        burst = {
            "ordered_step_flops": (attributed_step_flops,) * 100,
            "prelaunch_arm_mean_flops": plan.arm_mean_flops,
            "byte_matched_optimizer_steps": plan.byte_matched_optimizer_steps,
        }
        burst_sha256 = gtok_v2_bound_sha256(
            "weft1_gtok_v2_confirmation_burst_flops",
            burst,
        )
        burst_evidence_sha256 = gtok_v2_bound_sha256(
            "weft1_gtok_v2_confirmation_physical_burst_evidence",
            {
                "burst_receipt_sha256": burst_sha256,
                "compute_attempt_id": run.compute_attempt_id,
                "execution_plan_sha256": run.execution_plan_sha256,
            },
        )
        flop_evidence_sha256 = gtok_v2_bound_sha256(
            "weft1_gtok_v2_confirmation_physical_flop_ledger_evidence",
            {
                "compute_attempt_id": run.compute_attempt_id,
                "execution_plan_sha256": run.execution_plan_sha256,
                "flop_ledger_receipt_sha256": ledger_sha256,
            },
        )
        completion = {
            "run": asdict(run),
            "flop_ledger": ledger,
            "execution_plan_sha256": run.execution_plan_sha256,
            "base_flop_evidence_sha256": plan.arm_flop_plan_sha256,
            "training_plan_sha256": run.training_plan_sha256,
            "heldout_evaluation_steps": tuple(
                item.optimizer_step for item in run.observations
            ),
            "burst_flop_receipt": burst,
            "physical_flop_ledger_sha256": flop_evidence_sha256,
            "physical_optimizer_steps": optimizer_steps,
            "training_runtime_receipt_sha256": (
                run.training_runtime_receipt_sha256
            ),
            "code_closure_receipt_sha256": run.code_closure_receipt_sha256,
            "checkpoint_retained": False,
        }
        terminal = ConfirmationLifecycleEventEvidenceV2(
            logical_attempt_id=f"logical-{run.seed_slot}",
            attempt_id=run.compute_attempt_id,
            scope="confirmation",
            kind="full_run",
            phase="TERMINAL",
            charged_a100_microseconds=run.measured_a100_microseconds,
            terminal_status="completed",
            completion_payload=completion,
            gpu_uuid_provenance=run.gpu_uuid_provenance,
        )
        lifecycle_events.extend(
            (
                ConfirmationLifecycleEventEvidenceV2(
                    logical_attempt_id=terminal.logical_attempt_id,
                    attempt_id=terminal.attempt_id,
                    scope=terminal.scope,
                    kind=terminal.kind,
                    phase="START",
                    charged_a100_microseconds=1,
                    terminal_status=None,
                    gpu_uuid_provenance=terminal.gpu_uuid_provenance,
                ),
                terminal,
            )
        )
        joins.append(
            ConfirmationFreshEvidenceJoinV2(
                vocab_size=run.vocab_size,
                seed_slot=run.seed_slot,
                fresh_run_receipt_sha256=run.receipt_sha256,
                confirmation_order_receipt_sha256=(
                    run.confirmation_order_receipt_sha256
                ),
                physical_d6_evidence_sha256=run.physical_d6_evidence_sha256,
                document_multiset_sha256=run.document_multiset_sha256,
                ordered_raw_content_ids_sha256=run.data_order_sha256,
                framed_payload_sha256=run.framed_payload_sha256,
                order_document_count=run.stream_docs,
                order_retained_text_bytes=run.stream_bytes,
                execution_plan_sha256=run.execution_plan_sha256,
                training_plan_sha256=run.training_plan_sha256,
                compute_attempt_id=run.compute_attempt_id,
                terminal_lifecycle_event_sha256=terminal.receipt_sha256,
                burst_receipt_sha256=burst_evidence_sha256,
                physical_flop_ledger_sha256=flop_evidence_sha256,
            )
        )
    lifecycle_tuple = tuple(lifecycle_events)
    return ConfirmationEvidenceClosureV2(
        compute_event_ledger_sha256=compute.event_ledger_sha256,
        lifecycle_ledger_sha256=gtok_v2_bound_sha256(
            "weft1_gtok_v2_confirmation_lifecycle_ledger",
            lifecycle_tuple,
        ),
        lifecycle_events=lifecycle_tuple,
        execution_plans=tuple(execution_plans),
        confirmation_orders=tuple(confirmation_orders),
        attempt_launches=tuple(attempt_launches),
        base_flop_sources=tuple(
            _arm_flop_source(matrix, selection, vocab_size)
            for vocab_size in sorted(selection.compute_confirmation_pair)
        ),
        fresh_joins=tuple(joins),
    )


def _confirmation(matrix, selection):
    rows = _confirmation_runs(matrix, selection)
    return _validate_confirmation(matrix, selection, rows)


def _validate_confirmation(matrix, selection, rows, compute=None):
    if compute is None:
        compute = _confirmation_compute(matrix, selection, rows)
    return validate_compute_confirmation_v2(
        rows,
        matrix=matrix,
        selection=selection,
        compute=compute,
        evidence_closure=_confirmation_evidence_closure(
            matrix,
            selection,
            compute,
            rows,
        ),
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
    assert GTOK_SELECTION_CONFIRMATION_AUTHORITY_CHAIN == (
        *GTOK_V2_AUTHORITY_CHAIN,
        GTOK_CONFIRMATION_SEMANTICS_SHA256,
        GTOK_SEMANTICS_AMENDMENT_S1_SHA256,
        GTOK_SEMANTICS_AMENDMENT_S2_SHA256,
    )
    assert len(GTOK_SELECTION_CONFIRMATION_AUTHORITY_SHA256) == 64
    assert len(GTOK_SELECTION_CONFIRMATION_LITERAL_BINDING_SHA256_V2) == 64
    assert GTOK_RHO_BPB_DECIMAL_PLACES == 6
    assert GTOK_RHO_BPB_SCALE == 1_000_000
    assert GTOK_SELECTION_CONFIRMATION_LITERAL_BINDING_SHA256_V2 != (
        GTOK_SELECTOR_LITERAL_BINDING_SHA256_V2
    )
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
    wrong_run = replace(
        runs[0],
        observations=(*runs[0].observations[:-1], wrong_terminal),
        stream_bytes=GTOK_TRAINING_BYTE_BUDGET,
        trained_bytes=GTOK_TRAINING_BYTE_BUDGET,
    )
    with pytest.raises(ValueError, match="declared run stream"):
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


def test_authoritative_matrix_rejects_legacy_run_without_q2_accounting() -> None:
    matrix = _matrix()
    run = matrix.runs[0]
    legacy = replace(
        run,
        stream_bytes=None,
        stream_docs=None,
        stream_tokens=None,
        trained_tokens=None,
        dropped_tokens=None,
        trained_bytes=None,
        dropped_bytes=None,
        trained_docs_full=None,
        boundary_doc_id=None,
        boundary_doc_consumed_tokens=None,
        dropped_docs=None,
    )
    assert not legacy.has_exact_stream_accounting
    with pytest.raises(ValueError, match="requires exact Q2 stream accounting"):
        validate_complete_gtok_matrix_v2(
            (legacy, *matrix.runs[1:]),
            corpus=matrix.corpus,
            tokenizers=matrix.tokenizers,
            compute=matrix.compute,
        )


def test_selector_requires_agreed_strict_seed_order_and_echoes_pairwise_math() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())

    assert selection.agreed_strict_terminal_order == (49_152, 32_768, 24_576, 16_384)
    assert selection.compute_confirmation_pair == (49_152, 32_768)
    assert selection.selected_vocab_size == 49_152
    assert selection.selection_confirmation_authority_sha256 == (
        GTOK_SELECTION_CONFIRMATION_AUTHORITY_SHA256
    )
    assert selection.selector_literal_binding_sha256 == (
        GTOK_SELECTION_CONFIRMATION_LITERAL_BINDING_SHA256_V2
    )
    assert tuple(row.rho_bpb_micros for row in selection.arm_statistics) == (
        1_210_000,
        1_110_000,
        1_010_000,
        960_000,
    )
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


def test_rho_uses_binary64_half_even_rounding_for_reporting_and_selection() -> None:
    values = {
        (16_384, 101): 1.4,
        (16_384, 202): 1.4,
        (24_576, 101): 1.3,
        (24_576, 202): 1.3,
        # These shortest decimal spellings are both half-micro cases. Their
        # exact binary64 values lie on opposite sides of their decimal
        # midpoints, so Decimal(str(value)) would produce the wrong result.
        (32_768, 101): float.fromhex("0x1.0000192a73711p+0"),
        (32_768, 202): float.fromhex("0x1.0000192a73711p+0"),
        (49_152, 101): float.fromhex("0x1.000008637bd06p+0"),
        (49_152, 202): float.fromhex("0x1.000008637bd06p+0"),
    }
    selection = select_vocabulary_v2(_matrix(values), admissibility=_admissibility())
    rho_by_vocab = {
        row.vocab_size: row.rho_bpb_micros for row in selection.arm_statistics
    }

    assert rho_by_vocab == {
        16_384: 1_400_000,
        24_576: 1_300_000,
        32_768: 1_000_001,
        49_152: 1_000_001,
    }
    assert selection.agreed_strict_terminal_order[:2] == (49_152, 32_768)
    assert selection.selected_vocab_size == 32_768
    assert selection.compute_confirmation_pair == (32_768, 49_152)
    final = selection.comparisons[-1]
    assert final.delta_bpb_micros == 0
    assert final.delta_bpb == 0.0
    assert final.displaced is False


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
    with pytest.raises(ValueError, match="forward semantics authority"):
        replace(
            selection,
            selection_confirmation_authority_sha256=_hash("wrong-semantics"),
        )
    with pytest.raises(ValueError, match="forward literal binding"):
        replace(
            selection,
            selector_literal_binding_sha256=_hash("wrong-literal-binding"),
        )

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
        incumbent_rho_bpb_micros=4_000_000,
        challenger_rho_bpb_micros=1_000_000,
        incumbent_sample_sd=1.0,
        challenger_sample_sd=1.0,
        s_hat=1.0,
        delta_bpb_micros=3_000_000,
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
    assert confirmation.selection_confirmation_authority_sha256 == (
        GTOK_SELECTION_CONFIRMATION_AUTHORITY_SHA256
    )
    assert len({run.measured_flops for run in confirmation.runs}) == 1

    rows = _confirmation_runs(matrix, selection)
    mutated = tuple(
        replace(
            run,
            observations=_revalue_confirmation_observations(
                matrix.corpus,
                run,
                1.30 + 0.01 * run.seed_slot,
            ),
        )
        for run in rows
    )
    compute = _confirmation_compute(matrix, selection, mutated)
    reversal = _validate_confirmation(matrix, selection, mutated, compute)
    assert reversal.status == "ESCALATE_REVERSAL"


def test_confirmation_reuses_two_base_slots_and_charges_only_two_fresh_slots() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    rows = _confirmation_runs(matrix, selection)
    compute = _confirmation_compute(matrix, selection, rows)
    confirmation = _validate_confirmation(matrix, selection, rows, compute)

    reused = tuple(
        slot
        for slot in confirmation.result_slots
        if slot.source == "reused_byte_matched"
    )
    fresh = tuple(
        slot
        for slot in confirmation.result_slots
        if slot.source == "fresh_confirmation"
    )
    assert len(confirmation.result_slots) == 4
    assert len(reused) == len(fresh) == 2
    assert {slot.seed_slot for slot in reused} == {0, 1}
    assert {slot.seed_slot for slot in fresh} == {0, 1}
    assert all(slot.compute_attempt_id is None for slot in reused)
    assert {slot.compute_attempt_id for slot in fresh} == {
        run.compute_attempt_id for run in rows
    }

    projections = compute.preflight.calibrations
    completed = tuple(
        attempt
        for attempt in compute.attempts
        if attempt.kind == "full_run" and attempt.status == "completed"
    )
    assert len(projections) == 1
    assert projections[0].projection_source == "completed_base_calibration"
    assert projections[0].vocab_size == confirmation.fresh_vocab_size
    assert len(completed) == 2
    assert {attempt.attempt_id for attempt in completed} == {
        run.compute_attempt_id for run in rows
    }
    assert not any(attempt.kind == "calibration" for attempt in compute.attempts)


def test_confirmation_opposite_slot_signs_escalate_seed_split() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    rows = _confirmation_runs(matrix, selection)
    assert selection.compute_confirmation_pair == (49_152, 32_768)
    mutated = tuple(
        replace(
            run,
            observations=_revalue_confirmation_observations(
                matrix.corpus,
                run,
                0.90 if run.seed_slot == 0 else 1.10,
            ),
        )
        for run in rows
    )
    confirmation = _validate_confirmation(matrix, selection, mutated)

    assert confirmation.slot_delta_bpb[0] < 0
    assert confirmation.slot_delta_bpb[1] > 0
    assert confirmation.status == "ESCALATE_SEED_SPLIT"


@pytest.mark.parametrize(
    ("fresh_runner_delta", "expected_status"),
    (
        (0.035, "GREEN_NO_REVERSAL"),
        (0.050, "ESCALATE_REVERSAL"),
    ),
)
def test_larger_vocab_runner_up_uses_three_s_hat_reversal_threshold(
    fresh_runner_delta: float,
    expected_status: str,
) -> None:
    matrix = _matrix(
        {
            (16_384, 101): 0.90,
            (16_384, 202): 0.92,
            (24_576, 101): 1.10,
            (24_576, 202): 1.12,
            (32_768, 101): 1.20,
            (32_768, 202): 1.22,
            (49_152, 101): 0.95,
            (49_152, 202): 0.97,
        }
    )
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    assert selection.compute_confirmation_pair == (16_384, 49_152)
    rows = _confirmation_runs(matrix, selection)
    mutated = tuple(
        replace(
            run,
            observations=_revalue_confirmation_observations(
                matrix.corpus,
                run,
                (0.90 if run.seed_slot == 0 else 0.92) - fresh_runner_delta,
            ),
        )
        for run in rows
    )
    confirmation = _validate_confirmation(matrix, selection, mutated)

    assert confirmation.runner_up_vocab_size > confirmation.winner_vocab_size
    assert confirmation.threshold_multiplier == 3
    assert confirmation.delta_bpb_micros > int(
        2 * confirmation.s_hat_c_bpb * GTOK_RHO_BPB_SCALE
    )
    if expected_status == "GREEN_NO_REVERSAL":
        assert confirmation.delta_bpb_micros <= int(
            3 * confirmation.s_hat_c_bpb * GTOK_RHO_BPB_SCALE
        )
    else:
        assert confirmation.delta_bpb_micros > int(
            3 * confirmation.s_hat_c_bpb * GTOK_RHO_BPB_SCALE
        )
    assert confirmation.status == expected_status


def test_smaller_vocab_runner_up_uses_two_s_hat_reversal_threshold() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    assert selection.compute_confirmation_pair == (49_152, 32_768)
    rows = _confirmation_runs(matrix, selection)
    mutated = tuple(
        replace(
            run,
            observations=_revalue_confirmation_observations(
                matrix.corpus,
                run,
                (1.00 if run.seed_slot == 0 else 1.02) + 0.035,
            ),
        )
        for run in rows
    )
    confirmation = _validate_confirmation(matrix, selection, mutated)

    assert confirmation.runner_up_vocab_size < confirmation.winner_vocab_size
    assert confirmation.threshold_multiplier == 2
    assert confirmation.delta_bpb_micros > int(
        2 * confirmation.s_hat_c_bpb * GTOK_RHO_BPB_SCALE
    )
    assert confirmation.status == "ESCALATE_REVERSAL"


def test_smaller_vocab_runner_at_exact_two_s_hat_does_not_reverse() -> None:
    values = dict(DEFAULT_TERMINAL_BPBS)
    values[(32_768, 101)] = 1.000
    values[(32_768, 202)] = 1.006
    matrix = _matrix(values)
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    assert selection.compute_confirmation_pair == (49_152, 32_768)
    rows = _confirmation_runs(matrix, selection)
    mutated = tuple(
        replace(
            run,
            observations=_revalue_confirmation_observations(
                matrix.corpus,
                run,
                1.009 if run.seed_slot == 0 else 1.017,
            ),
        )
        for run in rows
    )

    confirmation = _validate_confirmation(matrix, selection, mutated)

    assert confirmation.threshold_multiplier == 2
    assert confirmation.delta_bpb_micros == pytest.approx(
        2 * confirmation.s_hat_c_bpb * GTOK_RHO_BPB_SCALE,
        abs=1e-8,
    )
    assert all(delta > 0 for delta in confirmation.slot_delta_bpb)
    assert confirmation.status == "GREEN_NO_REVERSAL"


def test_larger_vocab_runner_at_exact_three_s_hat_does_not_reverse() -> None:
    matrix = _matrix(
        {
            (16_384, 101): 1.000,
            (16_384, 202): 1.006,
            (24_576, 101): 1.10,
            (24_576, 202): 1.12,
            (32_768, 101): 1.20,
            (32_768, 202): 1.22,
            (49_152, 101): 1.05,
            (49_152, 202): 1.07,
        }
    )
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    assert selection.compute_confirmation_pair == (16_384, 49_152)
    rows = _confirmation_runs(matrix, selection)
    mutated = tuple(
        replace(
            run,
            observations=_revalue_confirmation_observations(
                matrix.corpus,
                run,
                0.984 if run.seed_slot == 0 else 0.992,
            ),
        )
        for run in rows
    )

    confirmation = _validate_confirmation(matrix, selection, mutated)

    assert confirmation.threshold_multiplier == 3
    assert confirmation.delta_bpb_micros == pytest.approx(
        3 * confirmation.s_hat_c_bpb * GTOK_RHO_BPB_SCALE,
        abs=1e-8,
    )
    assert all(delta > 0 for delta in confirmation.slot_delta_bpb)
    assert confirmation.status == "GREEN_NO_REVERSAL"


def test_exact_zero_slot_delta_is_neither_seed_split_nor_reversal() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    rows = _confirmation_runs(matrix, selection)
    runner_up = selection.compute_confirmation_pair[1]
    base_by_key = {(run.vocab_size, run.seed): run for run in matrix.runs}
    mutated = tuple(
        replace(
            run,
            observations=(
                _revalue_confirmation_observations(
                    matrix.corpus,
                    run,
                    base_by_key[
                        (runner_up, matrix.seeds[run.seed_slot])
                    ].terminal.pooled_bpb,
                )
                if run.seed_slot == 0
                else _revalue_confirmation_observations(
                    matrix.corpus,
                    run,
                    1.04,
                )
            ),
        )
        for run in rows
    )

    confirmation = _validate_confirmation(matrix, selection, mutated)

    assert confirmation.slot_delta_bpb[0] == 0.0
    assert confirmation.slot_delta_bpb[1] > 0.0
    assert confirmation.status == "GREEN_NO_REVERSAL"


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
        calibration_steps=GTOK_CALIBRATION_MAX_STEPS,
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
    confirmation = _validate_confirmation(matrix, selection, rows, compute)
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
            evidence_closure=_confirmation_evidence_closure(
                matrix,
                selection,
                compute,
                rows,
            ),
        )


def test_confirmation_closure_rejects_terminal_only_lifecycle_preimages() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    rows = _confirmation_runs(matrix, selection)
    compute = _confirmation_compute(matrix, selection, rows)
    closure = _confirmation_evidence_closure(matrix, selection, compute, rows)
    terminals = tuple(
        event for event in closure.lifecycle_events if event.phase == "TERMINAL"
    )
    with pytest.raises(ValueError, match="lacks START"):
        replace(
            closure,
            lifecycle_events=terminals,
            lifecycle_ledger_sha256=gtok_v2_bound_sha256(
                "weft1_gtok_v2_confirmation_lifecycle_ledger",
                terminals,
            ),
        )


def test_confirmation_closure_rejects_unjoined_q3_order_preimage() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    rows = _confirmation_runs(matrix, selection)
    compute = _confirmation_compute(matrix, selection, rows)
    closure = _confirmation_evidence_closure(matrix, selection, compute, rows)
    first = closure.confirmation_orders[0]
    order = ConfirmationConsumerOrderV4(**first.payload)
    drifted = replace(order, ordered_raw_content_ids_sha256=_hash("drifted-q3-order"))
    drifted_order_artifact = {
        "binding_sha256": CONFIRMATION_BINDING_SHA256_V2,
        "payload": asdict(drifted),
        "receipt_sha256": drifted.receipt_sha256,
        "schema": "weft1_gtok_confirmation_consumer_order_v4",
    }
    drifted_envelope = ConfirmationOrderEnvelopeV2(
        payload=asdict(drifted),
        receipt_sha256=drifted.receipt_sha256,
        physical_sha256=sha256_bytes(
            canonical_json_bytes(drifted_order_artifact) + b"\n"
        ),
    )
    drifted_closure = replace(
        closure,
        confirmation_orders=(drifted_envelope, closure.confirmation_orders[1]),
    )
    with pytest.raises(ValueError, match="physical Q3 order preimage"):
        validate_compute_confirmation_v2(
            rows,
            matrix=matrix,
            selection=selection,
            compute=compute,
            evidence_closure=drifted_closure,
        )


def test_confirmation_completion_base_flop_preimage_must_match_plan() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    rows = _confirmation_runs(matrix, selection)
    compute = _confirmation_compute(matrix, selection, rows)
    closure = _confirmation_evidence_closure(matrix, selection, compute, rows)
    terminal_index = next(
        index
        for index, event in enumerate(closure.lifecycle_events)
        if event.phase == "TERMINAL" and event.attempt_id == rows[0].compute_attempt_id
    )
    terminal = closure.lifecycle_events[terminal_index]
    completion = dict(terminal.completion_payload)
    completion["base_flop_evidence_sha256"] = _hash("wrong-base-flop-plan")
    drifted_terminal = replace(terminal, completion_payload=completion)
    lifecycle = list(closure.lifecycle_events)
    lifecycle[terminal_index] = drifted_terminal
    joins = list(closure.fresh_joins)
    joins[0] = replace(
        joins[0],
        terminal_lifecycle_event_sha256=drifted_terminal.receipt_sha256,
    )
    lifecycle_tuple = tuple(lifecycle)
    drifted_closure = replace(
        closure,
        lifecycle_events=lifecycle_tuple,
        lifecycle_ledger_sha256=gtok_v2_bound_sha256(
            "weft1_gtok_v2_confirmation_lifecycle_ledger",
            lifecycle_tuple,
        ),
        fresh_joins=tuple(joins),
    )
    with pytest.raises(ValueError, match="physical burst/FLOP evidence"):
        validate_compute_confirmation_v2(
            rows,
            matrix=matrix,
            selection=selection,
            compute=compute,
            evidence_closure=drifted_closure,
        )


def test_confirmation_retry_artifact_rejects_wrong_authority_even_if_byte_exact() -> None:
    payload = {
        "attempt": {},
        "attempt_receipt_sha256": _hash("attempt"),
        "binding_sha256": _hash("wrong-confirmation-authority"),
        "correction_ordinal": 0,
        "failed_execution_plan_sha256": _hash("failed-plan"),
        "failed_optimizer_steps": 100,
        "failed_projected_run_a100_microseconds": 1,
        "failed_terminal_lifecycle_event_sha256": _hash("terminal"),
        "invalid_physical_flop_ledger": {},
        "invalid_physical_flop_ledger_sha256": _hash("ledger-evidence"),
        "invalid_flop_ledger_receipt_sha256": _hash("ledger"),
        "passed_burst_flop_receipt": {},
        "passed_burst_receipt_sha256": _hash("burst"),
        "passed_physical_burst_evidence_sha256": _hash("burst-evidence"),
        "realized_flops": 1,
        "retry_execution_plan": {},
        "retry_execution_plan_sha256": _hash("retry-plan"),
        "retry_projected_run_a100_microseconds": 1,
        "retry_steps": 101,
        "schema": "weft1_gtok_v2_invalid_confirmation_flop_band",
        "target_flops": 2,
    }
    physical_sha256 = sha256_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(ValueError, match="authority drifted"):
        ConfirmationRetryArtifactEnvelopeV2(
            payload=payload,
            physical_sha256=physical_sha256,
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


def test_precalibration_replay_attempts_are_projected_and_metered() -> None:
    binding_sha = _hash("replay-plan-binding")
    projection_sha = _hash("replay-pair-projection")
    first = PrecalibrationReplayAttemptReceiptV2(
        attempt_id="base-determinism-v16384-t17-r0",
        scope="base_screen",
        kind="determinism_replay",
        vocab_size=16_384,
        terminal_rows=17,
        representative_seed=SEEDS[0],
        replica_index=0,
        consumed_a100_microseconds=11,
        status="completed",
        replay_plan_binding_sha256=binding_sha,
    )
    second = PrecalibrationReplayAttemptReceiptV2(
        attempt_id="base-determinism-v16384-t17-r1",
        scope="base_screen",
        kind="determinism_replay",
        vocab_size=16_384,
        terminal_rows=17,
        representative_seed=SEEDS[0],
        replica_index=1,
        consumed_a100_microseconds=12,
        status="completed",
        replay_plan_binding_sha256=binding_sha,
        replay_pair_projection_sha256=projection_sha,
        projected_replica_a100_microseconds=11,
        watchdog_limit_a100_microseconds=22,
    )
    calibration = _preflight("base_screen", (16_384,)).calibrations
    preflight = PreflightProjectionReceiptV2(
        scope="base_screen",
        prior_campaign_a100_microseconds=0,
        prior_event_ledger_sha256=None,
        calibrations=calibration,
        projected_campaign_a100_microseconds=(
            23 + sum(row.projected_scope_a100_microseconds for row in calibration)
        ),
        precalibration_replay_attempts=(first, second),
        precalibration_replay_plan_set_sha256=_hash("replay-plan-set"),
        precalibration_replay_receipt_sha256s=(_hash("replay-pair-receipt"),),
        precalibration_replay_authority_sha256=_hash("replay-authority"),
    )
    assert preflight.projected_campaign_a100_microseconds == (
        first.consumed_a100_microseconds
        + second.consumed_a100_microseconds
        + sum(row.projected_scope_a100_microseconds for row in calibration)
    )
    with pytest.raises(ValueError, match="exactly 2x"):
        replace(second, watchdog_limit_a100_microseconds=23)


def test_campaign_event_ledger_is_recomputed_from_ordered_attempts() -> None:
    matrix = _matrix()
    compute = matrix.compute
    stale = _hash("stale-event-ledger")
    with pytest.raises(ValueError, match="ordered attempt receipts"):
        replace(
            compute,
            event_ledger_sha256=stale,
            runtime_snapshot=replace(
                compute.runtime_snapshot,
                event_ledger_sha256=stale,
            ),
        )
    with pytest.raises(ValueError, match="ordered attempt receipts"):
        replace(compute, attempts=tuple(reversed(compute.attempts)))


def test_base_and_confirmation_mints_reject_any_watchdog_abort() -> None:
    corpus = _corpus()
    tokenizers = _tokenizers(corpus)
    runs = _runs(corpus, tokenizers)
    preflight = _preflight("base_screen", GTOK_VOCABULARY_ARMS)
    projection = preflight.calibrations[0]
    base_abort = ComputeAttemptReceiptV2(
        attempt_id="base-watchdog-abort-before-mint",
        scope="base_screen",
        kind="full_run",
        vocab_size=projection.vocab_size,
        seed=SEEDS[0],
        consumed_a100_microseconds=(
            2 * projection.projected_run_a100_microseconds + 1
        ),
        status="aborted_watchdog",
        calibration_projection_sha256=projection.receipt_sha256,
        projected_run_a100_microseconds=projection.projected_run_a100_microseconds,
        watchdog_limit_a100_microseconds=(
            2 * projection.projected_run_a100_microseconds
        ),
        hard_abort_issued=True,
    )
    base_compute = _campaign(
        preflight,
        runs,
        predecessor_campaign_sha256=None,
        event_ledger_label="ignored",
        extra_attempts=(base_abort,),
    )
    with pytest.raises(GTokV2Stop, match="watchdog abort"):
        validate_complete_gtok_matrix_v2(
            runs,
            corpus=corpus,
            tokenizers=tokenizers,
            compute=base_compute,
        )

    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    confirmation_rows = _confirmation_runs(matrix, selection)
    healthy_compute = _confirmation_compute(matrix, selection, confirmation_rows)
    prior = healthy_compute.attempts[0]
    assert isinstance(prior, ComputeAttemptReceiptV2)
    confirmation_abort = replace(
        prior,
        attempt_id="confirmation-watchdog-abort-before-mint",
        consumed_a100_microseconds=prior.watchdog_limit_a100_microseconds + 1,
        status="aborted_watchdog",
        hard_abort_issued=True,
    )
    stopped_compute = _campaign(
        healthy_compute.preflight,
        confirmation_rows,
        predecessor_campaign_sha256=matrix.compute.receipt_sha256,
        event_ledger_label="ignored",
        extra_attempts=(confirmation_abort,),
    )
    closure = _confirmation_evidence_closure(
        matrix,
        selection,
        stopped_compute,
        confirmation_rows,
    )
    with pytest.raises(GTokV2Stop, match="watchdog abort"):
        validate_compute_confirmation_v2(
            confirmation_rows,
            matrix=matrix,
            selection=selection,
            compute=stopped_compute,
            evidence_closure=closure,
        )


def test_confirmation_nested_preimages_are_rehashed_at_mint_boundary() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    rows = _confirmation_runs(matrix, selection)
    compute = _confirmation_compute(matrix, selection, rows)
    closure = _confirmation_evidence_closure(matrix, selection, compute, rows)
    mutable_plan = closure.execution_plans[0].payload
    assert isinstance(mutable_plan, dict)
    mutable_plan["target_flops"] = int(mutable_plan["target_flops"]) + 1
    with pytest.raises(ValueError, match="SHA differs"):
        validate_compute_confirmation_v2(
            rows,
            matrix=matrix,
            selection=selection,
            compute=compute,
            evidence_closure=closure,
        )


def test_confirmation_physical_order_and_base_flop_preimages_are_fail_closed() -> None:
    matrix = _matrix()
    selection = select_vocabulary_v2(matrix, admissibility=_admissibility())
    rows = _confirmation_runs(matrix, selection)
    compute = _confirmation_compute(matrix, selection, rows)
    closure = _confirmation_evidence_closure(matrix, selection, compute, rows)
    with pytest.raises(ValueError, match="durable bytes"):
        replace(
            closure.confirmation_orders[0],
            physical_sha256=_hash("wrong-physical-order-artifact"),
        )
    raw_base = closure.base_flop_sources[0].base_runs[0].flop_ledger_payload
    assert isinstance(raw_base, dict)
    raw_base["optimizer_steps"] = int(raw_base["optimizer_steps"]) + 1
    with pytest.raises(ValueError, match="raw FLOP ledger SHA"):
        validate_compute_confirmation_v2(
            rows,
            matrix=matrix,
            selection=selection,
            compute=compute,
            evidence_closure=closure,
        )
