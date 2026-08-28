from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math

import pytest

from models.ablation_lm.config import AblationLMConfig
from training.weft1_gtok_contract import (
    AppendOnlyExtensionBasisReceipt,
    AppendOnlyCorpusExtensionReceipt,
    BaseTokenizerContractReceipt,
    BpbObservationReceipt,
    ByteRoundTripFixtureReceipt,
    CorpusAuditManifestReceipt,
    CorpusStratumByteStats,
    FlatAdamWRecipe,
    GTOK_A100_HOUR_TRIPWIRE,
    GTOK_AUTHORITY_CHAIN,
    GTOK_BPB_MILESTONE_BYTES,
    GTOK_BPB_MILESTONE_FRACTIONS,
    GTOK_CURRICULUM_AMENDMENT_SHA256,
    GTOK_CURRICULUM_DATA_SHA256,
    GTOK_CURRICULUM_DECISIONS_SHA256,
    GTOK_RATIFICATION_SHA256,
    GTokComputeReceipt,
    GTOK_ENGLISH_SCOPE_SHA256,
    GTOK_EXECUTION_AUTHORITY_CHAIN,
    GTOK_EXECUTION_HANDOFF_SHA256,
    GTOK_HANDOFF_SHA256,
    GTOK_PROXY_TOPOLOGY,
    GTOK_PROXY_TOPOLOGY_SHA256,
    GTOK_QWEN_ADJUDICATION_SHA256,
    GTOK_ROUND_TRIP_CATEGORIES,
    GTOK_RULINGS_SHA256,
    GTOK_SEED_COUNT,
    GTOK_STRATA,
    GTOK_TRAINING_BYTE_BUDGET,
    GTOK_TRAINING_BYTE_CEILING,
    GTOK_VOCABULARY_ARMS,
    GTokComputeEventReceipt,
    GTokExecutionBlocked,
    GTokProxyTopologyReceipt,
    GTokRunReceipt,
    StratumNllReceipt,
    TokenizerArtifactSnapshot,
    UNRESOLVED_GTOK_DECISIONS,
    assert_identical_flat_adamw,
    bits_per_byte,
    authority_bound_sha256,
    canonical_sha256,
    execution_authority_bound_sha256,
    require_gtok_execution_authority,
    sha256_bytes,
    validate_append_only_tokenizer_extension,
    validate_a100_hour_tripwire,
    validate_complete_gtok_bpb_receipts,
)


def _hash(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


def _gtok_proxy_config(vocab_size: int = 32_768) -> AblationLMConfig:
    return AblationLMConfig(
        vocab_size=vocab_size,
        n_prelude_layers=4,
        n_core_blocks=2,
        n_coda_layers=4,
    )


def _optimizer(*, learning_rate: float = 3e-4) -> FlatAdamWRecipe:
    return FlatAdamWRecipe(
        hyperparameters=(
            ("betas", (0.9, 0.95)),
            ("eps", 1e-8),
            ("learning_rate", learning_rate),
            ("weight_decay", 0.1),
        ),
        schedule=(("kind", "caller_bound_fixture"),),
    )


def _observation(
    fraction: Fraction,
    *,
    heldout_hash: str | None = None,
    general_raw_bytes: int = 9,
) -> BpbObservationReceipt:
    raw_bytes = (general_raw_bytes, 5, 3, 3)
    strata = tuple(
        StratumNllReceipt(
            stratum=name,
            nll_nats=math.log(2.0) * count,
            raw_byte_count=count,
        )
        for name, count in zip(GTOK_STRATA, raw_bytes, strict=True)
    )
    return BpbObservationReceipt(
        milestone_fraction=fraction,
        training_raw_bytes=GTOK_BPB_MILESTONE_BYTES[fraction],
        heldout_stream_sha256=_hash("heldout") if heldout_hash is None else heldout_hash,
        strata=strata,
    )


def _runs(*, optimizer: FlatAdamWRecipe | None = None) -> tuple[GTokRunReceipt, ...]:
    recipe = _optimizer() if optimizer is None else optimizer
    seeds = (101, 202)
    manifest = _corpus_manifest()
    return tuple(
        GTokRunReceipt(
            vocab_size=vocab_size,
            seed=seed,
            corpus_manifest_sha256=manifest.manifest_sha256,
            training_corpus_sha256=manifest.training_corpus_sha256,
            tokenizer_artifact_sha256=_hash(f"tokenizer-{vocab_size}"),
            model_topology_sha256=GTOK_PROXY_TOPOLOGY_SHA256,
            initialization_recipe_sha256=_hash("initialization-recipe"),
            initialization_seed=10_000 + seed,
            shared_initial_state_sha256=_hash(f"shared-initial-state-{seed}"),
            data_order_seed=20_000 + seed,
            data_order_sha256=_hash(f"order-{seed}"),
            compute_attempt_id=f"base-{vocab_size}-{seed}",
            measured_a100_hours=0.5,
            optimizer=recipe,
            observations=tuple(
                _observation(fraction) for fraction in GTOK_BPB_MILESTONE_FRACTIONS
            ),
        )
        for vocab_size in GTOK_VOCABULARY_ARMS
        for seed in seeds
    )


def _base_tokenizers(
    manifest: CorpusAuditManifestReceipt,
) -> tuple[BaseTokenizerContractReceipt, ...]:
    return tuple(
        BaseTokenizerContractReceipt(
            vocab_size=vocab_size,
            artifact_sha256=_hash(f"tokenizer-{vocab_size}"),
            corpus_manifest_sha256=manifest.manifest_sha256,
            fit_corpus_sha256=manifest.training_corpus_sha256,
            tokenizer_library="fixture-byte-bpe",
            tokenizer_version="0.0-test",
            pretokenizer_regex_sha256=_hash("literal-regex"),
            reserved_inventory_sha256=_hash("reserved-inventory"),
            token_inventory_sha256=_hash(f"token-inventory-{vocab_size}"),
            token_inventory_count=vocab_size,
        )
        for vocab_size in GTOK_VOCABULARY_ARMS
    )


def _compute(runs: tuple[GTokRunReceipt, ...]) -> GTokComputeReceipt:
    return GTokComputeReceipt(
        source_event_log_sha256=_hash("complete-scheduler-event-log"),
        events=tuple(
            GTokComputeEventReceipt(
                attempt_id=run.compute_attempt_id,
                event_index=event_index,
                scope="base_screen",
                status="completed",
                measured_a100_hours=run.measured_a100_hours,
                vocab_size=run.vocab_size,
                seed=run.seed,
            )
            for event_index, run in enumerate(runs)
        )
        + (
            GTokComputeEventReceipt(
                attempt_id="confirmation-0",
                event_index=len(runs),
                scope="confirmation",
                status="completed",
                measured_a100_hours=1.0,
            ),
        ),
    )


def _validate_bpb(
    runs: tuple[GTokRunReceipt, ...] | None = None,
):
    actual_runs = _runs() if runs is None else runs
    manifest = _corpus_manifest()
    return validate_complete_gtok_bpb_receipts(
        actual_runs,
        corpus_manifest=manifest,
        tokenizers=_base_tokenizers(manifest),
        compute=_compute(actual_runs),
    )


def _corpus_stats() -> tuple[CorpusStratumByteStats, ...]:
    rows = (
        ("general", 450, 0),
        ("code", 250, 2),
        ("mathematics", 150, 3),
        ("science_technical", 150, 1),
    )
    return tuple(
        CorpusStratumByteStats(
            name=name,
            raw_byte_count=raw_bytes,
            non_ascii_byte_count=non_ascii_bytes,
            non_ascii_fraction=Fraction(non_ascii_bytes, raw_bytes),
        )
        for name, raw_bytes, non_ascii_bytes in rows
    )


def _round_trip_fixture() -> ByteRoundTripFixtureReceipt:
    digest = _hash("adversarial-byte-fixture")
    return ByteRoundTripFixtureReceipt(
        fixture_id="all_required_categories",
        categories=GTOK_ROUND_TRIP_CATEGORIES,
        original_bytes_sha256=digest,
        round_trip_bytes_sha256=digest,
    )


def _corpus_manifest() -> CorpusAuditManifestReceipt:
    return CorpusAuditManifestReceipt(
        source_file_manifest_sha256=_hash("source-files"),
        document_split_manifest_sha256=_hash("document-split"),
        training_corpus_sha256=_hash("training-corpus"),
        heldout_corpus_sha256=_hash("heldout"),
        strata=_corpus_stats(),
        heldout_stratum_raw_byte_counts=(
            ("general", 9),
            ("code", 5),
            ("mathematics", 3),
            ("science_technical", 3),
        ),
        round_trip_fixtures=(_round_trip_fixture(),),
        training_raw_byte_count=980,
        heldout_raw_byte_count=20,
        heldout_fraction=Fraction(1, 50),
        document_overlap_count=0,
        language_filter_method="fixture_document_language_id",
        language_filter_threshold=Fraction(9, 10),
        language_filter_audit_sha256=_hash("language-filter-audit"),
    )


def _tokenizer_snapshot(
    *,
    artifact: str,
    token_labels: tuple[str, ...],
    merges: tuple[str, ...],
    regex: str = r"(?u)\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+",
    normalization: str = "identity_bytes",
    reserved_ids: tuple[int, ...] = (0, 1),
) -> TokenizerArtifactSnapshot:
    return TokenizerArtifactSnapshot(
        artifact_sha256=_hash(artifact),
        corpus_manifest_sha256=_hash(f"corpus-{artifact}"),
        tokenizer_library="fixture-byte-bpe",
        tokenizer_version="0.0-test",
        pretokenizer_regex=regex,
        pretokenizer_regex_sha256=sha256_bytes(regex.encode("utf-8")),
        normalization=normalization,
        reserved_token_ids=reserved_ids,
        id_to_token_bytes_sha256=tuple(_hash(label) for label in token_labels),
        id_to_token_metadata_sha256=tuple(
            _hash(f"metadata-{label}") for label in token_labels
        ),
        special_role_to_id=(("eos", 1), ("pad", 0)),
        merge_entries=merges,
        merge_table_sha256=canonical_sha256(merges),
    )


def _extension_pair() -> tuple[TokenizerArtifactSnapshot, TokenizerArtifactSnapshot]:
    parent = _tokenizer_snapshot(
        artifact="parent",
        token_labels=("pad", "eos", "a", "b"),
        merges=("61 62 -> 6162",),
    )
    child = _tokenizer_snapshot(
        artifact="child",
        token_labels=("pad", "eos", "a", "b", "ab"),
        merges=("61 62 -> 6162", "6162 63 -> 616263"),
    )
    return parent, child


def _extension_basis(parent: TokenizerArtifactSnapshot) -> AppendOnlyExtensionBasisReceipt:
    return AppendOnlyExtensionBasisReceipt(
        parent_artifact_sha256=parent.artifact_sha256,
        parent_merge_table_sha256=parent.merge_table_sha256,
        parent_corpus_manifest_sha256=parent.corpus_manifest_sha256,
        parent_pretokenizer_regex_sha256=parent.pretokenizer_regex_sha256,
        parent_token_id_manifest_sha256=parent.token_id_manifest_sha256,
    )


def _extension_corpus(
    parent: TokenizerArtifactSnapshot,
    child: TokenizerArtifactSnapshot,
) -> AppendOnlyCorpusExtensionReceipt:
    return AppendOnlyCorpusExtensionReceipt(
        parent_corpus_manifest_sha256=parent.corpus_manifest_sha256,
        added_corpus_manifest_sha256=_hash("extension-corpus-addition"),
        combined_corpus_manifest_sha256=child.corpus_manifest_sha256,
    )


def test_binding_authority_chain_and_screen_constants_are_exact() -> None:
    assert GTOK_HANDOFF_SHA256 == (
        "498f34b5966f0879c7f0a15ca8be02a603558781c35f59f03fb29cc9edd3eb02"
    )
    assert GTOK_RULINGS_SHA256 == (
        "167fc17da1ac71a8263f5e190dc07dcd681ed82c64123a3394dc2b47e42cf0d2"
    )
    assert GTOK_ENGLISH_SCOPE_SHA256 == (
        "19399342cb6233258ac2ba411b6dc1feaab101c3f3986d751b6debe20dee02d3"
    )
    assert GTOK_CURRICULUM_AMENDMENT_SHA256 == (
        "0221545d62f7ed189898abf56f1ca65be6683de4d8a396d80bae4a4a094065b5"
    )
    assert GTOK_RATIFICATION_SHA256 == (
        "c5df74297594e75697ffb71d8d05d75efcf94f7857d55ddd357043200efb6d3a"
    )
    assert GTOK_CURRICULUM_DATA_SHA256 == (
        "14f0ba5d32898d69413839b8e342cc74b858eef90e65079175e37968052dea22"
    )
    assert GTOK_QWEN_ADJUDICATION_SHA256 == (
        "6c2568d5ba7f8295c65493b863d0530e71ee78e2290455307b00bdcdee480a1f"
    )
    assert GTOK_CURRICULUM_DECISIONS_SHA256 == (
        "61fc7727e456d822f43613db602c0251344b64ea92c7b256af5f1fe560cd8b6d"
    )
    assert GTOK_EXECUTION_HANDOFF_SHA256 == (
        "2aecb64711a2bf2776c8d1940350bc5d42b335f60eb774ac1e941f470b9cf74c"
    )
    assert GTOK_AUTHORITY_CHAIN == (
        GTOK_HANDOFF_SHA256,
        GTOK_RULINGS_SHA256,
        GTOK_ENGLISH_SCOPE_SHA256,
        GTOK_CURRICULUM_AMENDMENT_SHA256,
        GTOK_RATIFICATION_SHA256,
    )
    assert GTOK_EXECUTION_AUTHORITY_CHAIN == (
        GTOK_HANDOFF_SHA256,
        GTOK_RATIFICATION_SHA256,
        GTOK_RULINGS_SHA256,
        GTOK_ENGLISH_SCOPE_SHA256,
        GTOK_CURRICULUM_AMENDMENT_SHA256,
        GTOK_CURRICULUM_DATA_SHA256,
        GTOK_QWEN_ADJUDICATION_SHA256,
        GTOK_CURRICULUM_DECISIONS_SHA256,
        GTOK_EXECUTION_HANDOFF_SHA256,
    )
    assert execution_authority_bound_sha256("fixture", {"value": 1}) != (
        authority_bound_sha256("fixture", {"value": 1})
    )
    assert GTOK_TRAINING_BYTE_BUDGET == 4_000_000_000
    assert GTOK_TRAINING_BYTE_CEILING == 4_000_000_000
    assert GTOK_A100_HOUR_TRIPWIRE == 12.0
    assert GTOK_VOCABULARY_ARMS == (16_384, 24_576, 32_768, 49_152)
    assert GTOK_SEED_COUNT == 2
    assert GTOK_BPB_MILESTONE_FRACTIONS == (
        Fraction(1, 4),
        Fraction(1, 2),
        Fraction(1, 1),
    )
    assert tuple(
        GTOK_BPB_MILESTONE_BYTES[fraction]
        for fraction in GTOK_BPB_MILESTONE_FRACTIONS
    ) == (1_000_000_000, 2_000_000_000, 4_000_000_000)
    assert validate_a100_hour_tripwire(12.0) == 12.0
    with pytest.raises(RuntimeError, match="12 A100-hour"):
        validate_a100_hour_tripwire(12.000_001)


def test_gtok_proxy_topology_is_exact_authority_bound_and_vocab_independent() -> None:
    assert GTOK_PROXY_TOPOLOGY_SHA256 == (
        "b7feb601cb71fc216d6198b57fdb0a345c601184210402be0407915473bf5bac"
    )
    assert GTOK_PROXY_TOPOLOGY.receipt_sha256 == authority_bound_sha256(
        "weft1_gtok_proxy_topology_v1",
        GTOK_PROXY_TOPOLOGY,
    )
    assert GTOK_PROXY_TOPOLOGY.receipt_sha256 != canonical_sha256(GTOK_PROXY_TOPOLOGY)
    assert (
        GTOK_PROXY_TOPOLOGY.n_prelude_layers,
        GTOK_PROXY_TOPOLOGY.n_core_blocks,
        GTOK_PROXY_TOPOLOGY.n_coda_layers,
    ) == (4, 2, 4)
    assert GTOK_PROXY_TOPOLOGY.vocabulary_binding == "gtok_run_arm"
    assert all(
        getattr(GTOK_PROXY_TOPOLOGY, name) is False
        for name in (
            "use_recurrence",
            "use_static_kv_core",
            "static_kv_midpoint_refresh",
            "use_front_hadamard_experts",
            "use_reentry_bridge",
            "use_scratch",
            "use_lane_carrier",
            "use_engram",
            "use_long_term_memory",
        )
    )
    for vocab_size in GTOK_VOCABULARY_ARMS:
        assert GTokProxyTopologyReceipt.from_config(
            _gtok_proxy_config(vocab_size)
        ) == GTOK_PROXY_TOPOLOGY


@pytest.mark.parametrize(
    "changes",
    (
        {"n_prelude_layers": 3, "n_core_blocks": 4, "n_coda_layers": 3},
        {"use_front_hadamard_experts": True},
        {"use_recurrence": True},
        {"use_recurrence": True, "use_static_kv_core": True},
        {
            "use_recurrence": True,
            "use_static_kv_core": True,
            "static_kv_midpoint_refresh": True,
        },
        {"use_recurrence": True, "use_reentry_bridge": True},
        {"use_scratch": True},
        {"use_scratch": True, "use_lane_carrier": True},
        {"use_engram": True},
        {"use_long_term_memory": True},
        {"z_loss_coefficient": 1e-4},
    ),
)
def test_gtok_proxy_topology_rejects_graph_drift(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="exact ratified 4/2/4 S0 graph"):
        GTokProxyTopologyReceipt.from_config(replace(_gtok_proxy_config(), **changes))


def test_gtok_runs_reject_a_common_but_wrong_topology_hash() -> None:
    with pytest.raises(ValueError, match="exact ratified G-TOK 4/2/4 S0 receipt"):
        replace(_runs()[0], model_topology_sha256=_hash("wrong-common-topology"))
    with pytest.raises(ValueError, match="exact ratified 4/2/4 S0 graph"):
        GTokProxyTopologyReceipt(use_recurrence=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact ratified 4/2/4 S0 graph"):
        GTokProxyTopologyReceipt(attention_dropout=0.1)


def test_bpb_uses_raw_bytes_and_pools_nll_before_division() -> None:
    observation = _observation(Fraction(1, 4))

    assert bits_per_byte(math.log(2.0) * 17, 17) == pytest.approx(1.0)
    assert observation.pooled_bpb == pytest.approx(1.0)
    assert observation.bpb_by_stratum == {
        stratum: pytest.approx(1.0) for stratum in GTOK_STRATA
    }
    assert observation.pooled_raw_byte_count == 20
    with pytest.raises(ValueError, match="positive integer"):
        bits_per_byte(1.0, 0)
    with pytest.raises(TypeError, match="numeric"):
        StratumNllReceipt("general", "1.0", 1)  # type: ignore[arg-type]
    huge_strata = tuple(
        StratumNllReceipt(stratum, 1e308, 1) for stratum in GTOK_STRATA
    )
    with pytest.raises(ValueError, match="pooled held-out NLL"):
        BpbObservationReceipt(
            milestone_fraction=Fraction(1, 4),
            training_raw_bytes=GTOK_BPB_MILESTONE_BYTES[Fraction(1, 4)],
            heldout_stream_sha256=_hash("heldout"),
            strata=huge_strata,
        )


def test_complete_receipts_require_every_arm_seed_and_milestone() -> None:
    runs = _runs()
    matrix = _validate_bpb(runs)

    assert matrix.schema == "weft1_gtok_bpb_matrix_v1"
    assert matrix.authority_chain == GTOK_AUTHORITY_CHAIN
    assert matrix.vocab_sizes == GTOK_VOCABULARY_ARMS
    assert matrix.seeds == (101, 202)
    assert len(matrix.runs) == 8
    assert matrix.heldout_denominator_signature == (
        ("general", 9),
        ("code", 5),
        ("mathematics", 3),
        ("science_technical", 3),
    )
    assert not hasattr(matrix, "winner")

    with pytest.raises(ValueError, match="exactly 8"):
        _validate_bpb(runs[:-1])
    with pytest.raises(ValueError, match="0.25/0.5/1.0"):
        replace(runs[0], observations=runs[0].observations[:-1])


def test_complete_receipts_require_identical_heldout_denominators_and_data_order() -> None:
    runs = _runs()
    changed_observation = _observation(
        Fraction(1, 4),
        general_raw_bytes=10,
    )
    changed_run = replace(
        runs[0],
        observations=(changed_observation, *runs[0].observations[1:]),
    )
    with pytest.raises(ValueError, match="identical raw-byte denominators"):
        _validate_bpb((changed_run, *runs[1:]))

    all_changed = tuple(
        replace(
            run,
            observations=tuple(
                _observation(observation.milestone_fraction, general_raw_bytes=10)
                for observation in run.observations
            ),
        )
        for run in runs
    )
    with pytest.raises(ValueError, match="manifested held-out strata"):
        _validate_bpb(all_changed)

    changed_order = replace(runs[0], data_order_sha256=_hash("different-order"))
    with pytest.raises(ValueError, match="identical data order"):
        _validate_bpb((changed_order, *runs[1:]))


def test_complete_receipts_require_independent_seed_variation() -> None:
    runs = _runs()
    same_order = tuple(
        replace(run, data_order_seed=20_101, data_order_sha256=_hash("order-101"))
        if run.seed == 202
        else run
        for run in runs
    )
    with pytest.raises(ValueError, match="distinct data orders"):
        _validate_bpb(same_order)

    same_initialization = tuple(
        replace(run, initialization_seed=10_101) if run.seed == 202 else run
        for run in runs
    )
    with pytest.raises(ValueError, match="distinct initialization seeds"):
        _validate_bpb(same_initialization)

    same_initial_state = tuple(
        replace(
            run,
            shared_initial_state_sha256=_hash("shared-initial-state-101"),
        )
        if run.seed == 202
        else run
        for run in runs
    )
    with pytest.raises(ValueError, match="distinct non-vocabulary initial states"):
        _validate_bpb(same_initial_state)

    mismatched_within_seed = (
        replace(runs[0], shared_initial_state_sha256=_hash("wrong-shared-state")),
        *runs[1:],
    )
    with pytest.raises(ValueError, match="same non-vocabulary initialization state"):
        _validate_bpb(mismatched_within_seed)


def test_complete_receipts_join_corpus_tokenizer_and_compute_evidence() -> None:
    runs = _runs()
    wrong_manifest = replace(runs[0], corpus_manifest_sha256=_hash("wrong-manifest"))
    with pytest.raises(ValueError, match="same frozen corpus"):
        _validate_bpb((wrong_manifest, *runs[1:]))

    wrong_tokenizer = replace(
        runs[0],
        tokenizer_artifact_sha256=_hash("wrong-tokenizer"),
    )
    with pytest.raises(ValueError, match="not joined to its arm contract"):
        _validate_bpb((wrong_tokenizer, *runs[1:]))

    mismatched_hours = replace(runs[0], measured_a100_hours=0.25)
    manifest = _corpus_manifest()
    with pytest.raises(ValueError, match="differs from the cumulative compute receipt"):
        validate_complete_gtok_bpb_receipts(
            (mismatched_hours, *runs[1:]),
            corpus_manifest=manifest,
            tokenizers=_base_tokenizers(manifest),
            compute=_compute(runs),
        )


def test_tokenizer_arms_differ_only_in_v_and_exclude_heldout_bytes() -> None:
    runs = _runs()
    manifest = _corpus_manifest()
    tokenizers = _base_tokenizers(manifest)

    same_artifact = (
        tokenizers[0],
        replace(tokenizers[1], artifact_sha256=tokenizers[0].artifact_sha256),
        *tokenizers[2:],
    )
    with pytest.raises(ValueError, match="distinct tokenizer artifact"):
        validate_complete_gtok_bpb_receipts(
            runs,
            corpus_manifest=manifest,
            tokenizers=same_artifact,
            compute=_compute(runs),
        )

    changed_algorithm = (
        tokenizers[0],
        replace(tokenizers[1], tokenizer_version="different-version"),
        *tokenizers[2:],
    )
    with pytest.raises(ValueError, match="differ only in vocabulary size"):
        validate_complete_gtok_bpb_receipts(
            runs,
            corpus_manifest=manifest,
            tokenizers=changed_algorithm,
            compute=_compute(runs),
        )

    heldout_contaminated = tuple(
        replace(tokenizer, fit_corpus_sha256=manifest.heldout_corpus_sha256)
        for tokenizer in tokenizers
    )
    with pytest.raises(ValueError, match="fit corpus must exclude"):
        validate_complete_gtok_bpb_receipts(
            runs,
            corpus_manifest=manifest,
            tokenizers=heldout_contaminated,
            compute=_compute(runs),
        )

    with pytest.raises(ValueError, match="inventory count"):
        replace(tokenizers[0], token_inventory_count=tokenizers[0].vocab_size - 1)


def test_compute_receipt_applies_the_tripwire_cumulatively() -> None:
    runs = _runs()
    with pytest.raises(RuntimeError, match="12 A100-hour"):
        GTokComputeReceipt(
            source_event_log_sha256=_hash("over-budget-event-log"),
            events=tuple(
                GTokComputeEventReceipt(
                    attempt_id=run.compute_attempt_id,
                    event_index=event_index,
                    scope="base_screen",
                    status="completed",
                    measured_a100_hours=1.5,
                    vocab_size=run.vocab_size,
                    seed=run.seed,
                )
                for event_index, run in enumerate(runs)
            )
            + (
                GTokComputeEventReceipt(
                    attempt_id="failed-retry",
                    event_index=len(runs),
                    scope="base_screen",
                    status="failed",
                    measured_a100_hours=0.001,
                    vocab_size=GTOK_VOCABULARY_ARMS[0],
                    seed=101,
                ),
            ),
        )


def test_compute_receipt_counts_failed_retries_without_selecting_them() -> None:
    runs = _runs()
    baseline = _compute(runs)
    retry = GTokComputeEventReceipt(
        attempt_id="failed-retry",
        event_index=len(baseline.events),
        scope="base_screen",
        status="failed",
        measured_a100_hours=0.25,
        vocab_size=GTOK_VOCABULARY_ARMS[0],
        seed=101,
    )
    compute = replace(
        baseline,
        source_event_log_sha256=_hash("event-log-with-failed-retry"),
        events=(*baseline.events, retry),
    )
    manifest = _corpus_manifest()
    matrix = validate_complete_gtok_bpb_receipts(
        runs,
        corpus_manifest=manifest,
        tokenizers=_base_tokenizers(manifest),
        compute=compute,
    )

    assert matrix.total_measured_a100_hours == pytest.approx(
        baseline.total_a100_hours + 0.25
    )

    completed_retry = replace(retry, status="completed")
    posthoc_compute = replace(
        compute,
        source_event_log_sha256=_hash("event-log-with-completed-retry"),
        events=(*baseline.events, completed_retry),
    )
    with pytest.raises(ValueError, match="exactly one completed attempt"):
        validate_complete_gtok_bpb_receipts(
            runs,
            corpus_manifest=manifest,
            tokenizers=_base_tokenizers(manifest),
            compute=posthoc_compute,
        )


def test_authority_hashed_semantic_fields_require_canonical_types() -> None:
    manifest = _corpus_manifest()
    tokenizer = _base_tokenizers(manifest)[0]
    run = _runs()[0]
    event = _compute(_runs()).events[0]

    with pytest.raises(TypeError, match="vocabulary size must be an exact integer"):
        replace(tokenizer, vocab_size=float(tokenizer.vocab_size))
    with pytest.raises(TypeError, match="byte_atom_count must be an exact integer"):
        replace(tokenizer, byte_atom_count=256.0)
    with pytest.raises(TypeError, match="bpe_dropout must be numeric"):
        replace(tokenizer, bpe_dropout="0")
    with pytest.raises(TypeError, match="run vocabulary size must be an exact integer"):
        replace(run, vocab_size=float(run.vocab_size))
    with pytest.raises(TypeError, match="compute event vocabulary size"):
        replace(event, vocab_size=float(event.vocab_size))
    with pytest.raises(TypeError, match="a100_hours must be numeric"):
        replace(event, measured_a100_hours="0.5")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        replace(event, scope="base_screen ")
    with pytest.raises(ValueError, match="status must be one of"):
        replace(event, status="succeeded")
    with pytest.raises(ValueError, match="require an arm and seed"):
        replace(event, vocab_size=None, seed=None)


def test_receipt_hashes_are_authority_and_schema_bound() -> None:
    manifest = _corpus_manifest()
    runs = _runs()
    compute = _compute(runs)
    matrix = _validate_bpb(runs)

    assert manifest.manifest_sha256 != canonical_sha256(manifest)
    assert manifest.manifest_sha256 == authority_bound_sha256(
        "weft1_gtok_corpus_audit_v1",
        manifest,
    )
    assert manifest.manifest_sha256 != authority_bound_sha256(
        "different_schema",
        manifest,
    )
    assert compute.events[0].receipt_sha256 == authority_bound_sha256(
        "weft1_gtok_compute_event_v1",
        compute.events[0],
    )
    assert compute.receipt_sha256 == authority_bound_sha256(
        "weft1_gtok_compute_ledger_v1",
        compute,
    )
    assert runs[0].receipt_sha256 == authority_bound_sha256(
        "weft1_gtok_run_v1",
        runs[0],
    )
    assert matrix.receipt_sha256 == authority_bound_sha256(matrix.schema, matrix)


def test_flat_adamw_is_identical_and_has_no_vocabulary_correction() -> None:
    recipe = _optimizer()
    assert assert_identical_flat_adamw((recipe, recipe)) is recipe

    with pytest.raises(ValueError, match="identical AdamW"):
        assert_identical_flat_adamw((recipe, _optimizer(learning_rate=1e-4)))
    signed_zero = FlatAdamWRecipe(
        hyperparameters=(
            ("betas", (0.9, 0.95)),
            ("eps", 1e-8),
            ("learning_rate", 3e-4),
            ("weight_decay", -0.0),
        ),
        schedule=recipe.schedule,
    )
    zero = FlatAdamWRecipe(
        hyperparameters=(
            ("betas", (0.9, 0.95)),
            ("eps", 1e-8),
            ("learning_rate", 3e-4),
            ("weight_decay", 0.0),
        ),
        schedule=recipe.schedule,
    )
    with pytest.raises(ValueError, match="identical AdamW"):
        assert_identical_flat_adamw((zero, signed_zero))
    with pytest.raises(ValueError, match="per-V"):
        FlatAdamWRecipe(
            hyperparameters=recipe.hyperparameters,
            schedule=recipe.schedule,
            vocabulary_size_correction="sqrt_v",
        )
    with pytest.raises(ValueError, match="Muon"):
        FlatAdamWRecipe(
            hyperparameters=recipe.hyperparameters,
            schedule=recipe.schedule,
            muon_enabled=True,
        )


def test_corpus_manifest_enforces_c1_c2_c3_without_reading_data() -> None:
    manifest = _corpus_manifest()

    assert len(manifest.manifest_sha256) == 64
    assert manifest.manifest_sha256 == authority_bound_sha256(
        "weft1_gtok_corpus_audit_v1",
        manifest,
    )
    assert tuple(item.name for item in manifest.strata) == GTOK_STRATA

    math_row = manifest.strata[2]
    with pytest.raises(ValueError, match="C1 requires non-ASCII bytes"):
        replace(
            manifest,
            strata=(
                *manifest.strata[:2],
                replace(
                    math_row,
                    non_ascii_byte_count=0,
                    non_ascii_fraction=Fraction(0, 1),
                ),
                manifest.strata[3],
            ),
        )
    with pytest.raises(ValueError, match="disagrees with byte counts"):
        replace(math_row, non_ascii_fraction=Fraction(1, 2))
    with pytest.raises(ValueError, match="prohibit byte-class filtering"):
        replace(manifest, byte_filtering_applied=True)
    with pytest.raises(ValueError, match="corpus hashes must differ"):
        replace(manifest, heldout_corpus_sha256=manifest.training_corpus_sha256)
    with pytest.raises(ValueError, match="document-level language filter"):
        replace(manifest, language_filter_method="none")


def test_c2_requires_exact_hashes_and_complete_adversarial_coverage() -> None:
    fixture = _round_trip_fixture()
    with pytest.raises(ValueError, match="round-trip hash differs"):
        replace(fixture, round_trip_bytes_sha256=_hash("changed"))

    incomplete = replace(fixture, categories=("greek",))
    with pytest.raises(ValueError, match="coverage is incomplete"):
        replace(_corpus_manifest(), round_trip_fixtures=(incomplete,))


def test_append_only_extension_preserves_ids_merges_and_basis_hashes() -> None:
    parent, child = _extension_pair()
    result = validate_append_only_tokenizer_extension(
        parent,
        child,
        _extension_basis(parent),
        _extension_corpus(parent, child),
    )

    assert result.preserved_token_ids == 4
    assert result.appended_token_ids == 1
    assert result.preserved_merges == 1
    assert result.appended_merges == 1
    assert result.segmentation_invariant is False


def test_append_only_extension_rejects_redefinition_and_merge_reordering() -> None:
    parent, child = _extension_pair()
    basis = _extension_basis(parent)

    changed_id = _tokenizer_snapshot(
        artifact="changed-id",
        token_labels=("pad", "changed-eos", "a", "b", "ab"),
        merges=child.merge_entries,
    )
    with pytest.raises(ValueError, match="renumbered or redefined"):
        validate_append_only_tokenizer_extension(
            parent,
            changed_id,
            basis,
            _extension_corpus(parent, changed_id),
        )

    changed_merges = _tokenizer_snapshot(
        artifact="changed-merges",
        token_labels=("pad", "eos", "a", "b", "ab"),
        merges=("6162 63 -> 616263", "61 62 -> 6162"),
    )
    with pytest.raises(ValueError, match="exact child prefix"):
        validate_append_only_tokenizer_extension(
            parent,
            changed_merges,
            basis,
            _extension_corpus(parent, changed_merges),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("pretokenizer_regex", r"\S+", "pre-tokenizer regex"),
        ("normalization", "nfkc", "normalization"),
        ("reserved_token_ids", (0, 1, 2), "reserved token IDs"),
    ),
)
def test_append_only_extension_rejects_changed_artifact_contract(
    field: str,
    value: object,
    message: str,
) -> None:
    parent, child = _extension_pair()
    kwargs = {
        "artifact": f"changed-{field}",
        "token_labels": ("pad", "eos", "a", "b", "ab"),
        "merges": child.merge_entries,
        "regex": child.pretokenizer_regex,
        "normalization": child.normalization,
        "reserved_ids": child.reserved_token_ids,
    }
    key = {
        "pretokenizer_regex": "regex",
        "normalization": "normalization",
        "reserved_token_ids": "reserved_ids",
    }[field]
    kwargs[key] = value
    changed = _tokenizer_snapshot(**kwargs)

    with pytest.raises(ValueError, match=message):
        validate_append_only_tokenizer_extension(
            parent,
            changed,
            _extension_basis(parent),
            _extension_corpus(parent, changed),
        )


def test_append_only_extension_requires_exact_parent_basis_hashes() -> None:
    parent, child = _extension_pair()
    forged = replace(
        _extension_basis(parent),
        parent_artifact_sha256=_hash("not-parent"),
    )

    with pytest.raises(ValueError, match="do not identify the parent"):
        validate_append_only_tokenizer_extension(
            parent,
            child,
            forged,
            _extension_corpus(parent, child),
        )


def test_append_only_extension_rejects_changed_token_metadata_and_roles() -> None:
    parent, child = _extension_pair()
    changed_metadata = replace(
        child,
        id_to_token_metadata_sha256=(
            _hash("changed-parent-metadata"),
            *child.id_to_token_metadata_sha256[1:],
        ),
    )
    with pytest.raises(ValueError, match="changed kind or AddedToken flags"):
        validate_append_only_tokenizer_extension(
            parent,
            changed_metadata,
            _extension_basis(parent),
            _extension_corpus(parent, changed_metadata),
        )

    changed_roles = replace(child, special_role_to_id=(("eos", 0), ("pad", 1)))
    with pytest.raises(ValueError, match="special token roles"):
        validate_append_only_tokenizer_extension(
            parent,
            changed_roles,
            _extension_basis(parent),
            _extension_corpus(parent, changed_roles),
        )


def test_append_only_extension_requires_joined_combined_corpus() -> None:
    parent, child = _extension_pair()
    wrong_extension = replace(
        _extension_corpus(parent, child),
        combined_corpus_manifest_sha256=_hash("wrong-combined-corpus"),
    )

    with pytest.raises(ValueError, match="not joined to the combined extension corpus"):
        validate_append_only_tokenizer_extension(
            parent,
            child,
            _extension_basis(parent),
            wrong_extension,
        )


def test_append_only_extension_requires_strict_corpus_and_artifact_growth() -> None:
    parent, child = _extension_pair()
    with pytest.raises(ValueError, match="strict parent-plus-addition"):
        replace(
            _extension_corpus(parent, child),
            combined_corpus_manifest_sha256=parent.corpus_manifest_sha256,
        )

    same_artifact = replace(child, artifact_sha256=parent.artifact_sha256)
    with pytest.raises(ValueError, match="distinct child artifact"):
        validate_append_only_tokenizer_extension(
            parent,
            same_artifact,
            _extension_basis(parent),
            _extension_corpus(parent, same_artifact),
        )


def test_tokenizer_snapshot_rejects_alias_ids_and_mutable_metadata() -> None:
    with pytest.raises(ValueError, match="may not alias"):
        _tokenizer_snapshot(
            artifact="alias-child",
            token_labels=("pad", "eos", "a", "a"),
            merges=("61 62 -> 6162",),
        )

    parent, _child = _extension_pair()
    with pytest.raises(TypeError, match="metadata manifest must be a tuple"):
        replace(
            parent,
            id_to_token_metadata_sha256=list(
                parent.id_to_token_metadata_sha256
            ),
        )


def test_append_only_bpe_requires_one_new_id_per_new_merge() -> None:
    parent, child = _extension_pair()
    extra_id = _tokenizer_snapshot(
        artifact="extra-id-child",
        token_labels=("pad", "eos", "a", "b", "ab", "abc"),
        merges=child.merge_entries,
    )

    with pytest.raises(ValueError, match="one appended token ID per merge"):
        validate_append_only_tokenizer_extension(
            parent,
            extra_id,
            _extension_basis(parent),
            _extension_corpus(parent, extra_id),
        )


def test_choice_dependent_actions_fail_closed_on_all_open_decisions() -> None:
    with pytest.raises(GTokExecutionBlocked) as raised:
        require_gtok_execution_authority("fit tokenizer")

    message = str(raised.value)
    assert "eight dense blocks in the execution handoff versus ten" in message
    assert "D-C-1 curriculum shape" not in message
    assert "D-C-2 tie rule" not in message
    assert "literal tokenizer library/version" in message
    assert "numerical AdamW hyperparameters" in message
    assert "undertrained-row norm threshold" in message
    assert "held-out denominator" in message
    assert len(UNRESOLVED_GTOK_DECISIONS) == 9
