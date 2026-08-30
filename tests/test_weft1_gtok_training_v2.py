from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer

import training.weft1_gtok_training_v2 as training
from training.weft1_gtok_tokenizer_a2 import fit_a2_tokenizer
from training.weft1_gtok_training_v2 import (
    CompleteFlopLedgerV2,
    MEASUREMENT_BINDING_SHA256_V2,
    PACKING_BINDING_SHA256_V2,
    PackedBatchV2,
    TrainingDocumentV2,
    _PhysicalFlopAccountantV2,
    _exact_token_prefix_raw_bytes_v2,
    _execute_optimizer_step_v2,
    build_flat_a1_adamw_v2,
    gtok_proxy_config_v2,
    iter_packed_global_batches_v2,
    learning_rate_for_step_v2,
    measure_output_surface_performance_v2,
    measure_tokenizer_corpus_metrics_v2,
    measure_vocabulary_fractions_v2,
    plan_training_stream_v2,
    shared_nonvocabulary_state_sha256_v2,
)


def _tokenizer() -> Tokenizer:
    texts = ["alpha beta gamma", "delta epsilon", "0123456789"] * 8
    return Tokenizer.from_str(
        fit_a2_tokenizer(texts, vocab_size=320, length=len(texts)).decode()
    )


def _document(text: str, stratum: str = "general") -> TrainingDocumentV2:
    raw = text.encode()
    return TrainingDocumentV2(
        raw_content_id=hashlib.sha1(raw).hexdigest(),  # noqa: S324
        text=text,
        stratum=stratum,
    )


def test_packer_is_causal_across_documents_and_credits_eos_batch() -> None:
    documents = (_document("a"), _document("b"))
    batches = tuple(
        iter_packed_global_batches_v2(
            documents,
            tokenizer=_tokenizer(),
            global_batch_sequences=2,
            sequence_length=8,
        )
    )
    assert len(batches) == 1
    batch = batches[0]
    # The terminal global batch executes only its physical sequence rows.
    assert batch.input_ids.shape == (1, 8)
    assert batch.completed_raw_bytes == len(b"a") + len(b"b")
    assert batch.completed_document_count == 2
    boundaries = batch.document_ids[:, :-1].ne(batch.document_ids[:, 1:])
    assert not bool(
        (
            boundaries
            & batch.attention_mask[:, :-1]
            & batch.attention_mask[:, 1:]
            & batch.document_ids[:, :-1].eq(batch.document_ids[:, 1:])
        ).any()
    )


def test_long_document_credit_waits_for_sequence_containing_eos() -> None:
    document = _document("alpha " * 40)
    batches = tuple(
        iter_packed_global_batches_v2(
            (document,),
            tokenizer=_tokenizer(),
            global_batch_sequences=1,
            sequence_length=8,
        )
    )
    assert len(batches) > 1
    assert all(batch.completed_raw_bytes == 0 for batch in batches[:-1])
    assert batches[-1].completed_raw_bytes == len(document.raw_bytes)
    # Every non-boundary token keeps its physical next-token target, including
    # the final token of a row whose successor begins the next row.
    flattened_targets = torch.cat(tuple(batch.target_ids.flatten() for batch in batches))
    flattened_inputs = torch.cat(tuple(batch.input_ids.flatten() for batch in batches))
    flattened_documents = torch.cat(
        tuple(batch.document_ids.flatten() for batch in batches)
    )
    assert set(flattened_documents[flattened_documents.ge(0)].tolist()) == {0}
    for left, right in zip(batches, batches[1:]):
        assert int(left.target_ids[0, -1]) == int(right.input_ids[0, 0])
    boundary_id = _tokenizer().token_to_id("<|doc_boundary|>")
    assert int(flattened_targets.ne(-100).sum()) == sum(
        batch.valid_prediction_count for batch in batches
    )
    assert boundary_id in flattened_inputs.tolist()
    assert boundary_id in flattened_targets.tolist()


def test_plan_is_replay_stable_and_rejects_wrong_frozen_denominator() -> None:
    tokenizer = _tokenizer()
    documents = (_document("alpha"), _document("beta"))
    factory = lambda: iter(documents)
    expected = sum(len(item.raw_bytes) for item in documents)
    first = plan_training_stream_v2(
        factory,
        tokenizer=tokenizer,
        expected_realized_raw_bytes=expected,
    )
    second = plan_training_stream_v2(
        factory,
        tokenizer=tokenizer,
        expected_realized_raw_bytes=expected,
    )
    assert first == second
    assert first.packing_binding_sha256 == PACKING_BINDING_SHA256_V2
    with pytest.raises(RuntimeError, match="differ from frozen"):
        plan_training_stream_v2(
            factory,
            tokenizer=tokenizer,
            expected_realized_raw_bytes=expected + 1,
        )


def test_v4_source_rejoins_current_d6_and_order_before_both_streams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    training_seed = next(iter(training.GTOK_DATA_ORDER_SEED_BY_TRAINING_SEED_V4))
    data_seed = training.GTOK_DATA_ORDER_SEED_BY_TRAINING_SEED_V4[training_seed]
    physical_sha = "1" * 64
    order_sha = "2" * 64
    calls: list[tuple[object, ...]] = []

    def training_iterator(root: Path, **kwargs: object):
        calls.append(("T", root, kwargs))
        return iter(("alpha",))

    def d6_assertion(root: Path, *, expected_physical_sha256: str):
        calls.append(("H", root, expected_physical_sha256))
        return {}

    monkeypatch.setattr(training, "iter_materialized_training_texts_v4", training_iterator)
    monkeypatch.setattr(training, "assert_current_physical_d6_identity_v4", d6_assertion)
    monkeypatch.setattr(training, "_load_screen_shard_manifest_v4", lambda _root: ("3" * 64, ()))
    source = training.V4CorpusSourceV2(
        root=tmp_path,
        physical_d6_evidence_sha256=physical_sha,
        training_order_receipts=((training_seed, data_seed, order_sha),),
        training_raw_bytes=5,
        heldout_raw_bytes_by_stratum=(("general", 0),),
    )

    assert tuple(source.training_documents(training_seed))[0].text == "alpha"
    assert tuple(source.heldout_documents("general")) == ()
    assert calls == [
        (
            "T",
            tmp_path,
            {
                "training_seed": training_seed,
                "expected_physical_d6_evidence_sha256": physical_sha,
                "expected_consumer_order_receipt": (
                    training_seed,
                    data_seed,
                    order_sha,
                ),
            },
        ),
        ("H", tmp_path, physical_sha),
    ]


def test_calibration_prefix_receipt_rejects_same_size_physical_substitution() -> None:
    full_slots = 256 * 2_048
    plan = training.TrainingPlanV2(
        optimizer_steps=101,
        compute_token_slots=100 * full_slots + 2_048,
        valid_prediction_count=101,
        realized_raw_bytes=101,
        document_count=101,
        packed_stream_sha256="a" * 64,
        calibration_prefix_steps=100,
        calibration_prefix_compute_token_slots=100 * full_slots,
        calibration_prefix_valid_prediction_count=100,
        calibration_prefix_realized_raw_bytes=100,
        calibration_prefix_document_count=100,
        calibration_prefix_packed_stream_sha256="b" * 64,
    )
    measurement = training.CalibrationMeasurementV2(
        steps=100,
        warmup_steps=20,
        measured_steps=80,
        measured_tokens=80 * full_slots,
        measured_a100_microseconds=80,
        charged_a100_microseconds=100,
        measured_heldout_evaluation_a100_microseconds=10,
        heldout_evaluations_per_full_run=3,
        measured_output_surface_a100_microseconds=10,
        output_surface_benchmarks_per_full_run=1,
        planned_tokens_per_run=plan.compute_token_slots,
        shared_initial_state_sha256="c" * 64,
        training_plan_sha256=plan.receipt_sha256,
        physical_prefix_packed_stream_sha256="b" * 64,
        physical_prefix_compute_token_slots=100 * full_slots,
        physical_prefix_valid_prediction_count=100,
        physical_prefix_realized_raw_bytes=100,
        physical_prefix_document_count=100,
    )
    training.validate_calibration_prefix_v2(measurement, plan=plan)
    with pytest.raises(training.GTokTrainingV2Error, match="physical prefix"):
        training.validate_calibration_prefix_v2(
            replace(measurement, physical_prefix_packed_stream_sha256="d" * 64),
            plan=plan,
        )


def test_physical_full_run_rejects_same_size_packed_stream_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _profile_batch(1, sequence=2_048)
    digest = hashlib.sha256()
    training._update_training_plan_digest_v2(digest, expected)
    plan = training.TrainingPlanV2(
        optimizer_steps=1,
        compute_token_slots=expected.input_ids.numel(),
        valid_prediction_count=expected.valid_prediction_count,
        realized_raw_bytes=expected.completed_raw_bytes,
        document_count=expected.completed_document_count,
        packed_stream_sha256=digest.hexdigest(),
    )
    mutated_ids = expected.input_ids.clone()
    mutated_ids[0, 0] = (mutated_ids[0, 0] + 1) % 10
    mutated = PackedBatchV2(
        input_ids=mutated_ids,
        target_ids=expected.target_ids,
        document_ids=expected.document_ids,
        attention_mask=expected.attention_mask,
        completed_raw_bytes=expected.completed_raw_bytes,
        completed_document_count=expected.completed_document_count,
        valid_prediction_count=expected.valid_prediction_count,
    )

    class Model:
        def to(self, _device):
            return self

        def train(self):
            return self

    class Accountant:
        def __init__(self, *_args, **_kwargs):
            pass

        def execute(self, *, operation, **_kwargs):
            operation()

    monkeypatch.setattr(training, "build_flat_a1_adamw_v2", lambda _model: object())
    monkeypatch.setattr(training, "_PhysicalFlopAccountantV2", Accountant)
    monkeypatch.setattr(training, "_execute_optimizer_step_v2", lambda *_a, **_k: None)
    monkeypatch.setattr(
        training,
        "iter_packed_global_batches_v2",
        lambda *_a, **_k: iter((mutated,)),
    )
    monkeypatch.setattr(training, "_elapsed_microseconds", lambda *_a, **_k: 1)
    with pytest.raises(training.GTokTrainingV2Error, match="frozen training plan"):
        training._train_batches_v2(
            Model(),  # type: ignore[arg-type]
            tokenizer=_tokenizer(),
            document_factory=lambda: iter(()),
            plan=plan,
            device=torch.device("cpu"),
            microbatch_sequences=1,
            maximum_steps=None,
            watchdog_limit_a100_microseconds=None,
            prior_campaign_a100_microseconds=0,
            campaign_tripwire_a100_microseconds=None,
            heldout_factory=None,
            heldout_stream_sha256=None,
        )


def test_exact_proxy_config_and_scheduler_endpoints() -> None:
    config = gtok_proxy_config_v2(vocab_size=16_384, initialization_seed=1, run_seed=2)
    assert (config.n_prelude_layers, config.n_core_blocks, config.n_coda_layers) == (4, 2, 4)
    assert not config.use_recurrence
    assert learning_rate_for_step_v2(1, 1_000) == pytest.approx(3e-5)
    assert learning_rate_for_step_v2(10, 1_000) == pytest.approx(3e-4)
    assert learning_rate_for_step_v2(1_000, 1_000) == pytest.approx(3e-5)
    # Literal floor(1%) differs from ceil at these edge counts.
    assert learning_rate_for_step_v2(1, 101) == pytest.approx(3e-4)
    assert learning_rate_for_step_v2(1, 199) == pytest.approx(3e-4)
    assert learning_rate_for_step_v2(101, 101) == pytest.approx(3e-5)
    assert learning_rate_for_step_v2(199, 199) == pytest.approx(3e-5)


def test_flat_adamw_has_one_group_and_fp32_masters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    optimizer = build_flat_a1_adamw_v2(model)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(3e-4)
    assert group["betas"] == (0.9, 0.95)
    assert group["eps"] == pytest.approx(1e-8)
    assert group["weight_decay"] == pytest.approx(0.1)
    assert all(parameter.dtype == torch.float32 for parameter in group["params"])


def test_shared_state_hash_excludes_only_named_vocabulary_aliases() -> None:
    class Tiny(torch.nn.Module):
        def __init__(self, vocab: int) -> None:
            super().__init__()
            self.token_embedding = torch.nn.Embedding(vocab, 3)
            self.lm_head = torch.nn.Linear(3, vocab, bias=False)
            self.lm_head.weight = self.token_embedding.weight
            self.hidden = torch.nn.Linear(3, 3, bias=False)

    torch.manual_seed(11)
    first = Tiny(5)
    torch.manual_seed(11)
    second = Tiny(9)
    with torch.no_grad():
        second.hidden.weight.copy_(first.hidden.weight)
    assert shared_nonvocabulary_state_sha256_v2(first) == shared_nonvocabulary_state_sha256_v2(second)
    with torch.no_grad():
        second.hidden.weight.add_(1)
    assert shared_nonvocabulary_state_sha256_v2(first) != shared_nonvocabulary_state_sha256_v2(second)


class _TinyProfileLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            d_model=4,
            d_ff=8,
            n_heads=2,
            n_kv_heads=1,
            vocab_size=320,
            n_prelude_layers=4,
            n_core_blocks=2,
            n_coda_layers=4,
        )
        self.token_embedding = torch.nn.Embedding(320, 4)
        self.lm_head = torch.nn.Linear(4, 320, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.hidden = torch.nn.Linear(4, 4, bias=False)

    def forward(self, input_ids: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
        hidden = self.hidden(self.token_embedding(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))


def _profile_batch(rows: int, sequence: int = 4) -> PackedBatchV2:
    ids = torch.arange(rows * sequence, dtype=torch.int64).reshape(rows, sequence) % 10
    targets = ids.roll(-1, dims=1)
    documents = torch.zeros_like(ids)
    return PackedBatchV2(
        input_ids=ids,
        target_ids=targets,
        document_ids=documents,
        attention_mask=torch.ones_like(ids, dtype=torch.bool),
        completed_raw_bytes=rows,
        completed_document_count=rows,
        valid_prediction_count=rows * sequence,
    )


def test_physical_flop_ledger_profiles_actual_tail_and_prices_unsupported_ops() -> None:
    model = _TinyProfileLM()
    optimizer = build_flat_a1_adamw_v2(model)
    device = torch.device("cpu")
    accountant = _PhysicalFlopAccountantV2(model, device=device)
    plan = SimpleNamespace(optimizer_steps=3, compute_token_slots=20)
    for step, batch in enumerate((_profile_batch(2), _profile_batch(2), _profile_batch(1)), 1):
        accountant.execute(
            batch=batch,
            step=step,
            operation=lambda batch=batch, step=step: _execute_optimizer_step_v2(
                model,
                optimizer,
                batch=batch,
                step=step,
                plan=plan,  # type: ignore[arg-type]
                device=device,
                microbatch_sequences=1,
            ),
        )
    ledger = accountant.finalize(plan)  # type: ignore[arg-type]
    assert isinstance(ledger, CompleteFlopLedgerV2)
    assert ledger.compute_token_slots == 20
    assert {(row.batch_rows, row.optimizer_phase, row.occurrences) for row in ledger.shapes} == {
        (2, "initial", 1),
        (2, "steady", 1),
        (1, "steady", 1),
    }
    assert ledger.profiler_flops > 0
    assert ledger.unsupported_flops > 0
    assert ledger.measured_flops == sum(row.total_flops for row in ledger.shapes)
    assert all(row.zero_flop_profiler_operators for row in ledger.shapes)
    assert all("=" in item.derivation for row in ledger.shapes for item in row.unsupported_rows)


def test_section_6_8_compression_undertrained_and_roundtrip_metrics() -> None:
    tokenizer = _tokenizer()
    training = (
        _document("alpha beta", "general"),
        _document("delta", "code"),
    )
    heldout = {
        stratum: (_document(f"{stratum} alpha beta", stratum),)
        for stratum in ("general", "code", "mathematics", "science_technical")
    }
    metrics = measure_tokenizer_corpus_metrics_v2(
        tokenizer=tokenizer,
        training_document_factory=lambda: iter(training),
        heldout_factory=lambda stratum: iter(heldout[stratum]),
    )
    assert metrics.measurement_binding_sha256 == MEASUREMENT_BINDING_SHA256_V2
    assert metrics.exact_byte_round_trip_passed
    assert metrics.undertrained_threshold == 1_000
    assert metrics.undertrained_row_count == metrics.nonreserved_row_count
    assert metrics.training_encoded_tokens > 0
    assert tuple(row.stratum for row in metrics.strata) == (
        "general",
        "code",
        "mathematics",
        "science_technical",
    )
    for row in metrics.strata:
        assert row.bytes_per_token > 0
        assert row.coverage_2048_raw_bytes_p50 == row.raw_bytes
        assert row.coverage_2048_raw_bytes_p95 == row.raw_bytes


def test_2048_token_coverage_is_an_exact_replayable_raw_byte_prefix() -> None:
    tokenizer = _tokenizer()
    text = "alpha beta " * 3_000
    covered, ids = _exact_token_prefix_raw_bytes_v2(tokenizer, text)
    assert len(ids) > 2_048
    assert 0 < covered < len(text.encode())


class _TinyOutputSurface(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=8, d_model=4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.hidden = torch.nn.Linear(4, 4, bias=False)


def test_section_6_8_output_surface_trials_and_vocab_fractions_are_literal() -> None:
    model = _TinyOutputSurface()
    full, decode = measure_output_surface_performance_v2(
        model,
        device=torch.device("cpu"),
    )
    assert (full.warmup_trials, full.timed_trials, full.timed_token_count) == (
        20,
        100,
        204_800,
    )
    assert (
        decode.batch_size,
        decode.context_tokens,
        decode.decode_tokens,
        decode.warmup_trials,
        decode.timed_trials,
        decode.timed_token_count,
    ) == (1, 2_048, 128, 20, 100, 12_800)
    assert full.tokens_per_second > 0
    assert decode.microseconds_per_decoded_token is not None
    assert (
        decode.scope
        == "output_projection_full_softmax_decode_surface_latency"
    )
    fractions = measure_vocabulary_fractions_v2(model)
    assert tuple(row.rung for row in fractions) == ("proxy", "target_a", "target_b")
    assert fractions[0].vocabulary_parameters == 32
    assert fractions[1].vocabulary_parameters == 8 * 1_024
    assert fractions[1].total_unique_parameters == 302_900_000 + (8 - 32_768) * 1_024
    assert fractions[2].total_unique_parameters == 305_800_000 + (8 - 32_768) * 1_024
    assert all(0 < row.fraction < 1 for row in fractions)
