from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import training.weft1_gtok_training_v2 as training
from training.weft1_corpus_a2 import A2_CAMPAIGN_ROOT_SEED
from training.weft1_corpus_materialize_a3 import ConfirmationConsumerOrderV4
from training.weft1_gtok_training_v2 import (
    ConfirmationTrainingPlanV2,
    GTokTrainingV2Error,
    TrainingDocumentV2,
    V4CorpusSourceV2,
    plan_confirmation_training_prefix_v2,
)
from training.weft1_seed import derive_module_seed


class _OneBytePerTokenTokenizer:
    _specials = {
        "<|pad|>": 0,
        "<|bos|>": 1,
        "<|eos|>": 2,
        "<|doc_boundary|>": 3,
    }

    def token_to_id(self, value: str) -> int | None:
        return self._specials.get(value)

    def encode(self, text: str, *, add_special_tokens: bool) -> SimpleNamespace:
        assert not add_special_tokens
        raw = text.encode("ascii")
        return SimpleNamespace(
            ids=[64 + value for value in raw],
            tokens=[chr(value) for value in raw],
        )


def _document(text: str) -> TrainingDocumentV2:
    raw = text.encode("utf-8")
    return TrainingDocumentV2(
        raw_content_id=hashlib.sha1(raw).hexdigest(),  # noqa: S324
        text=text,
        stratum="general",
    )


def _order_receipt(
    *,
    texts: tuple[str, ...],
    physical_sha256: str = "a" * 64,
) -> ConfirmationConsumerOrderV4:
    run_seed = 17
    return ConfirmationConsumerOrderV4(
        confirmation_run_seed=run_seed,
        data_order_seed=derive_module_seed(
            A2_CAMPAIGN_ROOT_SEED,
            f"gtok.data.shared.{run_seed}",
        ),
        physical_d6_evidence_sha256=physical_sha256,
        document_multiset_sha256="b" * 64,
        ordered_raw_content_ids_sha256="c" * 64,
        framed_payload_sha256="d" * 64,
        document_count=len(texts),
        retained_text_bytes=sum(len(text.encode("utf-8")) for text in texts),
    )


def _source(*, texts: tuple[str, ...], physical_sha256: str = "a" * 64) -> V4CorpusSourceV2:
    return V4CorpusSourceV2(
        root=Path("unused-by-isolated-test"),
        physical_d6_evidence_sha256=physical_sha256,
        training_order_receipts=(),
        training_raw_bytes=sum(len(text.encode("utf-8")) for text in texts),
        heldout_raw_bytes_by_stratum=(),
    )


def test_confirmation_source_preserves_q3_physical_joins_and_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = ("alpha", "beta")
    receipt = _order_receipt(texts=texts)
    source = _source(texts=texts)

    def fake_iterator(
        root: Path,
        *,
        order_receipt: ConfirmationConsumerOrderV4,
    ):
        assert root == source.root
        assert order_receipt is receipt
        yield from texts

    monkeypatch.setattr(
        training,
        "iter_materialized_confirmation_training_texts_v4",
        fake_iterator,
    )
    documents = tuple(source.confirmation_training_documents(receipt))
    assert tuple(document.text for document in documents) == texts
    assert all(document.stratum == "general" for document in documents)

    with pytest.raises(GTokTrainingV2Error, match="physical D6"):
        tuple(
            source.confirmation_training_documents(
                replace(receipt, physical_d6_evidence_sha256="e" * 64)
            )
        )
    with pytest.raises(GTokTrainingV2Error, match="frozen T byte"):
        tuple(
            source.confirmation_training_documents(
                replace(receipt, retained_text_bytes=receipt.retained_text_bytes + 1)
            )
        )

    monkeypatch.setattr(
        training,
        "iter_materialized_confirmation_training_texts_v4",
        lambda *_args, **_kwargs: iter(texts[:1]),
    )
    with pytest.raises(GTokTrainingV2Error, match="adapter exhaustion"):
        tuple(source.confirmation_training_documents(receipt))


def test_confirmation_prefix_exhausts_order_and_keeps_large_suffix_separate() -> None:
    texts = ("a" * (130 * 8), "tail")
    documents = tuple(_document(text) for text in texts)
    receipt = _order_receipt(texts=texts)
    plan = plan_confirmation_training_prefix_v2(
        lambda: iter(documents),
        tokenizer=_OneBytePerTokenTokenizer(),
        optimizer_steps=104,
        confirmation_order_receipt=receipt,
        global_batch_sequences=1,
        sequence_length=8,
    )
    replay = plan_confirmation_training_prefix_v2(
        lambda: iter(documents),
        tokenizer=_OneBytePerTokenTokenizer(),
        optimizer_steps=104,
        confirmation_order_receipt=receipt,
        global_batch_sequences=1,
        sequence_length=8,
    )

    assert isinstance(plan, ConfirmationTrainingPlanV2)
    assert replay == plan
    assert plan.confirmation_order_receipt_sha256 == receipt.receipt_sha256
    assert plan.optimizer_steps == 104
    assert plan.compute_token_slots == plan.trained_tokens == 104 * 8
    assert plan.stream_bytes == 130 * 8 + len("tail")
    assert plan.stream_tokens == (130 * 8 + 3) + (len("tail") + 3)
    assert plan.trained_bytes == 104 * 8 - 1
    assert plan.dropped_bytes == plan.stream_bytes - plan.trained_bytes
    assert plan.dropped_tokens == plan.stream_tokens - plan.trained_tokens
    assert plan.dropped_tokens > 8
    assert plan.stream_docs == 2
    assert plan.trained_docs_full == 0
    assert plan.boundary_doc_id == documents[0].raw_content_id
    assert plan.boundary_doc_consumed_tokens == 104 * 8
    assert plan.dropped_docs == 1
    assert plan.calibration_prefix_compute_token_slots == 100 * 8
    assert plan.calibration_prefix_realized_raw_bytes == 100 * 8 - 1
    assert plan.bpb_checkpoint_steps[2] == 104
    assert plan.bpb_checkpoint_steps[0] < plan.bpb_checkpoint_steps[1] < 104
    assert plan.realized_raw_bytes == plan.trained_bytes
    assert plan.document_count == plan.trained_docs_full

    with pytest.raises(GTokTrainingV2Error, match="Q3 order receipt"):
        plan_confirmation_training_prefix_v2(
            lambda: iter(documents),
            tokenizer=_OneBytePerTokenTokenizer(),
            optimizer_steps=104,
            confirmation_order_receipt=replace(
                receipt,
                retained_text_bytes=receipt.retained_text_bytes + 1,
            ),
            global_batch_sequences=1,
            sequence_length=8,
        )
    with pytest.raises(GTokTrainingV2Error, match="fewer complete batches"):
        plan_confirmation_training_prefix_v2(
            lambda: iter(documents),
            tokenizer=_OneBytePerTokenTokenizer(),
            optimizer_steps=132,
            confirmation_order_receipt=receipt,
            global_batch_sequences=1,
            sequence_length=8,
        )


def test_base_plan_still_rejects_more_than_one_dropped_global_batch() -> None:
    full_batch_slots = 256 * 2_048
    with pytest.raises(ValueError, match="strict partial global-batch suffix"):
        training.TrainingPlanV2(
            optimizer_steps=3,
            compute_token_slots=3 * full_batch_slots,
            valid_prediction_count=3 * full_batch_slots,
            realized_raw_bytes=1,
            document_count=0,
            packed_stream_sha256="a" * 64,
            stream_bytes=2,
            stream_docs=1,
            stream_tokens=4 * full_batch_slots,
            trained_tokens=3 * full_batch_slots,
            dropped_tokens=full_batch_slots,
            trained_bytes=1,
            dropped_bytes=1,
            trained_docs_full=0,
            boundary_doc_id=None,
            boundary_doc_consumed_tokens=None,
            dropped_docs=1,
            bpb_checkpoint_steps=(1, 2, 3),
        )
