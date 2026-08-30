"""Throwaway 4/2/4 proxy training primitives for the WEFT-1 G-TOK screen.

This is the execution layer for the already-reviewed v2 evidence contracts.  It
contains no vocabulary selector and writes no model state.  Production model
construction is fixed to the structural-OFF ten-block graph, FP32 master
parameters under bf16 autocast, and one literal AdamW parameter group.

Packing is a literal A2-R7 binding:

* each document is ``BOS + exact tokenizer encoding + EOS + DOC_BOUNDARY``;
* documents may share a sequence but never a document ID;
* a long document may continue in a later sequence under one causal document
  identity; its cross-row next-token target is retained even though attention
  never crosses a sequence row;
* all full global batches contain 256x2048 compute slots and the terminal batch
  executes its actual row count without row duplication or empty-row padding;
* raw bytes are credited only in the optimizer step containing that document's
  EOS, so milestone receipts never credit untrained suffix tokens;
* no global batch is dropped and no checkpoint is retained.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Callable, Iterable, Iterator, Mapping, Protocol

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from models.ablation_lm import AblationLM, AblationLMConfig
from training.weft1_corpus_a2 import JsonlZstdShardIdentityV3
from training.weft1_corpus_materialize_a3 import (
    GTOK_DATA_ORDER_SEED_BY_TRAINING_SEED_V4,
    GTOK_TRAINING_SEEDS,
    _load_screen_shard_manifest_v4,
    assert_current_physical_d6_identity_v4,
    iter_materialized_training_texts_v4,
    validate_physical_d6_evidence_v4,
)
from training.weft1_gtok_contract import (
    GTOK_PROXY_TOPOLOGY,
    GTOK_PROXY_TOPOLOGY_SHA256,
    GTOK_STRATA,
    StratumNllReceipt,
    a1_flat_adamw_recipe,
    canonical_json_bytes,
    canonical_sha256,
)
from training.weft1_gtok_tokenizer_a2 import iter_a2_shard_texts, special_token_strings
from training.weft1_gtok_v2_contract import (
    BpbMilestoneReceiptV2,
    FrozenScreenCorpusV2,
    GTOK_FIRST_BOUNDARY_BYTES,
    GTOK_MILESTONE_LABELS,
    GTOK_SECOND_BOUNDARY_BYTES,
    GTokRunReceiptV2,
    TokenizerArmReceiptV2,
)
from training.weft1_strict_io import assert_no_symlink_ancestors


PACKING_BINDING_V2 = {
    "batch_sequences": 256,
    "bos": "one_at_document_start",
    "document_boundary_attention": "forbidden_by_document_ids",
    "document_credit": "raw_bytes_at_batch_containing_eos",
    "eos": "one_at_document_end",
    "doc_boundary": "append_after_every_document",
    "final_batch": "execute_partial_sequence_batch_without_duplication_or_drop",
    "long_document": "lossless_token_chunks_one_document_identity_cross_row_target_retained",
    "packing": "greedy_in_registered_raw_content_id_order",
    "sequence_length": 2_048,
}
PACKING_BINDING_SHA256_V2 = canonical_sha256(PACKING_BINDING_V2)
SCHEDULE_BINDING_V2 = {
    "base_learning_rate": 3e-4,
    "final_fraction": (1, 10),
    "step_timing": "set_before_optimizer_step_1_indexed",
    "warmup_steps": "max(1,floor(total_steps*0.01))",
    "warmup": "linear_from_zero_to_peak_inclusive",
    "decay": "cosine_peak_to_0.1_peak_inclusive_at_terminal_step",
}
SCHEDULE_BINDING_SHA256_V2 = canonical_sha256(SCHEDULE_BINDING_V2)
FLOP_BINDING_V2 = {
    "definition": "torch_profiler_with_flops_plus_explicit_analytic_unsupported_operator_ledger",
    "compute_token_slots": "sum_of_physically_executed_batch_rows_times_2048",
    "profile_granularity": "first_actual_optimizer_step_for_each_rows_x_optimizer_phase_shape",
    "profiler": "torch.profiler.profile(with_flops=True,record_shapes=True)",
    "unsupported": "explicit_formula_rows_never_silent_zero_flop_events",
    "purpose": "same-run accounting_and_equal-flop_confirmation_budget",
}
FLOP_BINDING_SHA256_V2 = canonical_sha256(FLOP_BINDING_V2)
MEASUREMENT_BINDING_V2 = {
    "compression_stream": "frozen_H_per_stratum",
    "coverage": "nearest_rank_p50_p95_raw_utf8_bytes_in_first_2048_content_tokens",
    "decode": {
        "batch": 1,
        "context_tokens": 2_048,
        "decode_tokens": 128,
        "scope": "tied_output_projection_plus_exact_full_log_softmax",
        "timed_trials": 100,
        "warmup_trials": 20,
    },
    "full_softmax": {
        "batch": 1,
        "sequence_tokens": 2_048,
        "scope": "tied_output_projection_plus_exact_full_log_softmax",
        "timed_trials": 100,
        "warmup_trials": 20,
    },
    "round_trip": "UTF8_bytes_equal_decode_encode_for_every_scanned_T_and_H_document",
    "target_unique_parameter_denominators_at_vocab_32768": {
        "rung_a": 302_900_000,
        "rung_b": 305_800_000,
    },
    "target_width": 1_024,
    "undertrained": "nonreserved_tokenizer_rows_with_T_count_below_1000",
}
MEASUREMENT_BINDING_SHA256_V2 = canonical_sha256(MEASUREMENT_BINDING_V2)
INITIALIZATION_RECIPE_SHA256_V2 = canonical_sha256(
    {
        "algorithm": "AblationLM.reset_parameters_module_isolated_rng_v1",
        "packing_binding_sha256": PACKING_BINDING_SHA256_V2,
        "schedule_binding_sha256": SCHEDULE_BINDING_SHA256_V2,
        "topology_sha256": GTOK_PROXY_TOPOLOGY_SHA256,
        "vocabulary_rows": "arm_specific_same_initialization_seed",
    }
)


class GTokTrainingV2Error(RuntimeError):
    """A training or physical-data invariant failed."""


class GTokRunWatchdogV2(GTokTrainingV2Error):
    def __init__(self, consumed_a100_microseconds: int) -> None:
        self.consumed_a100_microseconds = consumed_a100_microseconds
        super().__init__("run crossed its strict >2x projected A100-time watchdog")


class GTokCampaignTripwireV2(GTokTrainingV2Error):
    def __init__(self, consumed_a100_microseconds: int) -> None:
        self.consumed_a100_microseconds = consumed_a100_microseconds
        super().__init__("campaign crossed the strict >12 A100-hour cumulative tripwire")


@dataclass(frozen=True)
class TrainingDocumentV2:
    raw_content_id: str
    text: str
    stratum: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.raw_content_id, str)
            or len(self.raw_content_id) != 40
            or any(character not in "0123456789abcdef" for character in self.raw_content_id)
        ):
            raise ValueError("training document requires a lowercase SHA-1 raw-content ID")
        if not isinstance(self.text, str):
            raise TypeError("training document text must be a string")
        if hashlib.sha1(self.raw_bytes).hexdigest() != self.raw_content_id:  # noqa: S324
            raise ValueError("training document raw-content ID differs from UTF-8 bytes")
        if self.stratum not in GTOK_STRATA:
            raise ValueError("training document uses an unknown stratum")

    @property
    def raw_bytes(self) -> bytes:
        return self.text.encode("utf-8", errors="strict")


@dataclass(frozen=True)
class PackedBatchV2:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    document_ids: torch.Tensor
    attention_mask: torch.Tensor
    completed_raw_bytes: int
    completed_document_count: int
    valid_prediction_count: int

    def __post_init__(self) -> None:
        expected = self.input_ids.shape
        if (
            len(expected) != 2
            or self.target_ids.shape != expected
            or self.document_ids.shape != expected
            or self.attention_mask.shape != expected
        ):
            raise ValueError("packed batch tensors must align as [batch, sequence]")
        if (
            self.input_ids.dtype != torch.int64
            or self.target_ids.dtype != torch.int64
            or self.document_ids.dtype != torch.int64
        ):
            raise TypeError("packed token and document IDs must be int64")
        if self.attention_mask.dtype != torch.bool:
            raise TypeError("packed attention mask must be bool")
        if type(self.completed_raw_bytes) is not int or self.completed_raw_bytes < 0:
            raise ValueError("completed raw bytes must be a non-negative exact integer")
        if type(self.completed_document_count) is not int or self.completed_document_count < 0:
            raise ValueError("completed document count must be non-negative")
        if type(self.valid_prediction_count) is not int or self.valid_prediction_count < 1:
            raise ValueError("each emitted packed batch needs a trainable next-token pair")
        if int(self.target_ids.ne(-100).sum().item()) != self.valid_prediction_count:
            raise ValueError("packed batch target count differs from its receipt")


@dataclass(frozen=True)
class TrainingPlanV2:
    optimizer_steps: int
    compute_token_slots: int
    valid_prediction_count: int
    realized_raw_bytes: int
    document_count: int
    packed_stream_sha256: str
    packing_binding_sha256: str = PACKING_BINDING_SHA256_V2
    calibration_prefix_steps: int | None = None
    calibration_prefix_compute_token_slots: int | None = None
    calibration_prefix_valid_prediction_count: int | None = None
    calibration_prefix_realized_raw_bytes: int | None = None
    calibration_prefix_document_count: int | None = None
    calibration_prefix_packed_stream_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "optimizer_steps",
            "compute_token_slots",
            "valid_prediction_count",
            "realized_raw_bytes",
            "document_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        row_slots = int(PACKING_BINDING_V2["sequence_length"])
        full_batch_slots = int(PACKING_BINDING_V2["batch_sequences"]) * row_slots
        minimum = (self.optimizer_steps - 1) * full_batch_slots + row_slots
        maximum = self.optimizer_steps * full_batch_slots
        if (
            self.compute_token_slots < minimum
            or self.compute_token_slots > maximum
            or self.compute_token_slots % row_slots
        ):
            raise ValueError("planned compute slots do not encode one actual partial final batch")
        if self.packing_binding_sha256 != PACKING_BINDING_SHA256_V2:
            raise ValueError("training plan packing binding drifted")
        for value in (self.packed_stream_sha256, self.packing_binding_sha256):
            if len(value) != 64:
                raise ValueError("training plan hashes must be SHA-256 values")
        prefix_names = (
            "calibration_prefix_steps",
            "calibration_prefix_compute_token_slots",
            "calibration_prefix_valid_prediction_count",
            "calibration_prefix_realized_raw_bytes",
            "calibration_prefix_document_count",
            "calibration_prefix_packed_stream_sha256",
        )
        prefix_values = tuple(getattr(self, name) for name in prefix_names)
        if any(value is not None for value in prefix_values):
            if any(value is None for value in prefix_values):
                raise ValueError("training plan calibration-prefix evidence is incomplete")
            if self.calibration_prefix_steps != 100:
                raise ValueError("training plan calibration prefix must be exactly 100 steps")
            if self.optimizer_steps < self.calibration_prefix_steps:
                raise ValueError("training plan is shorter than its calibration prefix")
            for name in (
                "calibration_prefix_compute_token_slots",
                "calibration_prefix_valid_prediction_count",
            ):
                value = getattr(self, name)
                if type(value) is not int or value < 1:
                    raise ValueError(f"{name} must be a positive exact integer")
            for name in (
                "calibration_prefix_realized_raw_bytes",
                "calibration_prefix_document_count",
            ):
                value = getattr(self, name)
                if type(value) is not int or value < 0:
                    raise ValueError(f"{name} must be a non-negative exact integer")
            if (
                self.calibration_prefix_compute_token_slots
                > self.compute_token_slots
                or self.calibration_prefix_valid_prediction_count
                > self.valid_prediction_count
                or self.calibration_prefix_realized_raw_bytes > self.realized_raw_bytes
                or self.calibration_prefix_document_count > self.document_count
            ):
                raise ValueError("training plan calibration prefix exceeds the full stream")
            if (
                not isinstance(self.calibration_prefix_packed_stream_sha256, str)
                or len(self.calibration_prefix_packed_stream_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.calibration_prefix_packed_stream_sha256
                )
            ):
                raise ValueError("training plan calibration prefix requires SHA-256")

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self)


class ModelOutput(Protocol):
    logits: torch.Tensor


def _tokenizer_ids(tokenizer: Tokenizer) -> tuple[int, int, int, int]:
    pad = tokenizer.token_to_id("<|pad|>")
    bos = tokenizer.token_to_id("<|bos|>")
    eos = tokenizer.token_to_id("<|eos|>")
    boundary = tokenizer.token_to_id("<|doc_boundary|>")
    if (pad, bos, eos, boundary) != (0, 1, 2, 3):
        raise GTokTrainingV2Error("registered pad/BOS/EOS/document-boundary token IDs drifted")
    return 0, 1, 2, 3


def iter_packed_global_batches_v2(
    documents: Iterable[TrainingDocumentV2],
    *,
    tokenizer: Tokenizer,
    global_batch_sequences: int = 256,
    sequence_length: int = 2_048,
) -> Iterator[PackedBatchV2]:
    """Greedily pack one registered document order into fixed global batches."""

    if type(global_batch_sequences) is not int or global_batch_sequences < 1:
        raise ValueError("global_batch_sequences must be positive")
    if type(sequence_length) is not int or sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    pad_id, bos_id, eos_id, boundary_id = _tokenizer_ids(tokenizer)
    rows_tokens: list[list[int]] = []
    rows_targets: list[list[int]] = []
    rows_documents: list[list[int]] = []
    current_tokens: list[int] = []
    current_targets: list[int] = []
    current_documents: list[int] = []
    batch_raw_bytes = 0
    batch_documents = 0
    document_ordinal = 0

    def close_row() -> None:
        nonlocal current_tokens, current_targets, current_documents
        if not current_tokens:
            return
        pad_count = sequence_length - len(current_tokens)
        rows_tokens.append(current_tokens + [pad_id] * pad_count)
        rows_targets.append(current_targets + [-100] * pad_count)
        rows_documents.append(current_documents + [-1] * pad_count)
        current_tokens = []
        current_targets = []
        current_documents = []

    def maybe_emit(*, force: bool = False) -> PackedBatchV2 | None:
        nonlocal rows_tokens, rows_targets, rows_documents, batch_raw_bytes, batch_documents
        if not rows_tokens or (not force and len(rows_tokens) != global_batch_sequences):
            return None
        if len(rows_tokens) > global_batch_sequences:
            raise GTokTrainingV2Error("packer accumulated more than one global batch")
        tokens = torch.tensor(rows_tokens, dtype=torch.int64)
        targets = torch.tensor(rows_targets, dtype=torch.int64)
        doc_ids = torch.tensor(rows_documents, dtype=torch.int64)
        mask = doc_ids.ge(0)
        count = int(targets.ne(-100).sum().item())
        if count < 1:
            raise GTokTrainingV2Error("packed global batch has no next-token target")
        result = PackedBatchV2(
            input_ids=tokens,
            target_ids=targets,
            document_ids=doc_ids,
            attention_mask=mask,
            completed_raw_bytes=batch_raw_bytes,
            completed_document_count=batch_documents,
            valid_prediction_count=count,
        )
        rows_tokens = []
        rows_targets = []
        rows_documents = []
        batch_raw_bytes = 0
        batch_documents = 0
        return result

    for document in documents:
        if not isinstance(document, TrainingDocumentV2):
            raise TypeError("packed stream requires TrainingDocumentV2 values")
        content_ids = tokenizer.encode(document.text, add_special_tokens=False).ids
        token_ids = [bos_id, *content_ids, eos_id, boundary_id]
        target_ids = token_ids[1:] + [-100]
        for token_index, token_id in enumerate(token_ids):
            if len(current_tokens) == sequence_length:
                close_row()
                emitted = maybe_emit()
                if emitted is not None:
                    yield emitted
            current_tokens.append(int(token_id))
            current_targets.append(int(target_ids[token_index]))
            current_documents.append(document_ordinal)
            if token_id == eos_id and token_index == len(token_ids) - 2:
                batch_raw_bytes += len(document.raw_bytes)
                batch_documents += 1
        document_ordinal += 1
    close_row()
    if rows_tokens:
        emitted = maybe_emit(force=True)
        if emitted is not None:
            yield emitted


def _update_training_plan_digest_v2(
    digest: Any,
    batch: PackedBatchV2,
) -> None:
    """Hash exactly the physical packed-batch fields used by planning."""

    digest.update(batch.completed_raw_bytes.to_bytes(8, "big"))
    digest.update(batch.completed_document_count.to_bytes(8, "big"))
    digest.update(batch.valid_prediction_count.to_bytes(8, "big"))
    digest.update(hashlib.sha256(batch.input_ids.numpy().tobytes()).digest())
    digest.update(hashlib.sha256(batch.target_ids.numpy().tobytes()).digest())
    digest.update(hashlib.sha256(batch.document_ids.numpy().tobytes()).digest())


def plan_training_stream_v2(
    document_factory: Callable[[], Iterable[TrainingDocumentV2]],
    *,
    tokenizer: Tokenizer,
    expected_realized_raw_bytes: int,
    global_batch_sequences: int = 256,
    sequence_length: int = 2_048,
) -> TrainingPlanV2:
    digest = hashlib.sha256()
    prefix_digest = hashlib.sha256()
    steps = 0
    slots = 0
    predictions = 0
    raw_bytes = 0
    documents = 0
    prefix_slots = 0
    prefix_predictions = 0
    prefix_raw_bytes = 0
    prefix_documents = 0
    for batch in iter_packed_global_batches_v2(
        document_factory(),
        tokenizer=tokenizer,
        global_batch_sequences=global_batch_sequences,
        sequence_length=sequence_length,
    ):
        _update_training_plan_digest_v2(digest, batch)
        steps += 1
        slots += batch.input_ids.numel()
        predictions += batch.valid_prediction_count
        raw_bytes += batch.completed_raw_bytes
        documents += batch.completed_document_count
        if steps <= 100:
            _update_training_plan_digest_v2(prefix_digest, batch)
            prefix_slots += batch.input_ids.numel()
            prefix_predictions += batch.valid_prediction_count
            prefix_raw_bytes += batch.completed_raw_bytes
            prefix_documents += batch.completed_document_count
    if raw_bytes != expected_realized_raw_bytes:
        raise GTokTrainingV2Error(
            f"planned T bytes {raw_bytes} differ from frozen realized T {expected_realized_raw_bytes}"
        )
    return TrainingPlanV2(
        optimizer_steps=steps,
        compute_token_slots=slots,
        valid_prediction_count=predictions,
        realized_raw_bytes=raw_bytes,
        document_count=documents,
        packed_stream_sha256=digest.hexdigest(),
        calibration_prefix_steps=100 if steps >= 100 else None,
        calibration_prefix_compute_token_slots=prefix_slots if steps >= 100 else None,
        calibration_prefix_valid_prediction_count=(
            prefix_predictions if steps >= 100 else None
        ),
        calibration_prefix_realized_raw_bytes=(prefix_raw_bytes if steps >= 100 else None),
        calibration_prefix_document_count=(prefix_documents if steps >= 100 else None),
        calibration_prefix_packed_stream_sha256=(
            prefix_digest.hexdigest() if steps >= 100 else None
        ),
    )


def gtok_proxy_config_v2(*, vocab_size: int, initialization_seed: int, run_seed: int) -> AblationLMConfig:
    config = AblationLMConfig(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        n_kv_heads=4,
        d_ff=1_408,
        n_prelude_layers=4,
        n_core_blocks=2,
        n_coda_layers=4,
        max_sequence_length=2_048,
        rope_theta=500_000.0,
        norm_eps=1e-5,
        attention_dropout=0.0,
        tie_embeddings=True,
        recurrent_steps=1,
        max_recurrent_steps=8,
        recurrence_coefficient=1.0,
        recurrence_exponent=1.0,
        use_recurrence=False,
        use_static_kv_core=False,
        static_kv_midpoint_refresh=False,
        use_front_hadamard_experts=False,
        use_reentry_bridge=False,
        use_scratch=False,
        use_lane_carrier=False,
        use_engram=False,
        use_long_term_memory=False,
        z_loss_coefficient=0.0,
        initialization_seed=initialization_seed,
        run_seed=run_seed,
    )
    GTOK_PROXY_TOPOLOGY.from_config(config)
    return config


def build_gtok_proxy_model_v2(*, vocab_size: int, initialization_seed: int, run_seed: int) -> AblationLM:
    model = AblationLM(
        gtok_proxy_config_v2(
            vocab_size=vocab_size,
            initialization_seed=initialization_seed,
            run_seed=run_seed,
        )
    )
    if model.config.use_recurrence or any(
        (
            model.config.use_front_hadamard_experts,
            model.config.use_reentry_bridge,
            model.config.use_scratch,
            model.config.use_lane_carrier,
            model.config.use_engram,
            model.config.use_long_term_memory,
        )
    ):
        raise GTokTrainingV2Error("G-TOK proxy contains a non-S0 module")
    return model


def shared_nonvocabulary_state_sha256_v2(model: torch.nn.Module) -> str:
    """Hash identical-shape state while excluding only the tied vocabulary rows."""

    digest = hashlib.sha256()
    excluded = {"token_embedding.weight", "lm_head.weight"}
    observed = 0
    for name, tensor in sorted(model.state_dict().items()):
        if name in excluded:
            continue
        value = tensor.detach().cpu().contiguous()
        header = canonical_json_bytes(
            {"dtype": str(value.dtype), "name": name, "shape": tuple(value.shape)}
        )
        raw = value.view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        observed += 1
    if observed < 1:
        raise GTokTrainingV2Error("shared state hash observed no non-vocabulary state")
    return digest.hexdigest()


def build_flat_a1_adamw_v2(model: torch.nn.Module) -> torch.optim.AdamW:
    recipe = a1_flat_adamw_recipe()
    values = dict(recipe.hyperparameters)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters or len({id(parameter) for parameter in parameters}) != len(parameters):
        raise GTokTrainingV2Error("flat AdamW input is empty or repeats a tied tensor")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise GTokTrainingV2Error("G-TOK requires FP32 master parameters")
    optimizer = torch.optim.AdamW(
        [{"params": parameters}],
        lr=float(values["learning_rate"]),
        betas=tuple(values["betas"]),
        eps=float(values["eps"]),
        weight_decay=float(values["weight_decay"]),
    )
    if len(optimizer.param_groups) != 1:
        raise GTokTrainingV2Error("G-TOK AdamW must have exactly one flat group")
    return optimizer


def learning_rate_for_step_v2(step: int, total_steps: int) -> float:
    if type(step) is not int or type(total_steps) is not int or not 1 <= step <= total_steps:
        raise ValueError("scheduler step must lie in [1,total_steps]")
    peak = 3e-4
    warmup_steps = max(1, math.floor(total_steps * 0.01))
    if step <= warmup_steps:
        return peak * step / warmup_steps
    if total_steps == warmup_steps:
        return peak
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * multiplier


def _prediction_targets(batch: PackedBatchV2, device: torch.device) -> tuple[torch.Tensor, int]:
    targets = batch.target_ids.to(device, non_blocking=True)
    count = int(targets.ne(-100).sum().item())
    return targets, count


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _require_production_a100(device: torch.device, *, allow_nonproduction_cpu: bool) -> None:
    if allow_nonproduction_cpu:
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        raise GTokTrainingV2Error("production G-TOK requires an allocated NVIDIA A100")
    name = torch.cuda.get_device_name(device)
    if "A100" not in name.upper():
        raise GTokTrainingV2Error(
            f"A100-hour receipts cannot be minted on unbound accelerator {name!r}"
        )
    if not torch.cuda.is_bf16_supported():
        raise GTokTrainingV2Error("allocated A100 does not report bf16 support")


def require_production_a100_v2(device: torch.device) -> None:
    """Fail before corpus planning unless the selected device is an A100."""

    _require_production_a100(device, allow_nonproduction_cpu=False)


def _elapsed_microseconds(start_ns: int, device: torch.device) -> int:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return max(1, math.ceil((time.perf_counter_ns() - start_ns) / 1_000))


def evaluate_heldout_v2(
    model: torch.nn.Module,
    *,
    tokenizer: Tokenizer,
    heldout_factory: Callable[[str], Iterable[TrainingDocumentV2]],
    device: torch.device,
    microbatch_sequences: int,
    sequence_length: int = 2_048,
) -> tuple[StratumNllReceipt, ...]:
    model.eval()
    receipts: list[StratumNllReceipt] = []
    with torch.no_grad():
        for stratum in GTOK_STRATA:
            nll = 0.0
            raw_bytes = 0
            for batch in iter_packed_global_batches_v2(
                heldout_factory(stratum),
                tokenizer=tokenizer,
                global_batch_sequences=microbatch_sequences,
                sequence_length=sequence_length,
            ):
                raw_bytes += batch.completed_raw_bytes
                for offset in range(0, batch.input_ids.shape[0], microbatch_sequences):
                    target_view = batch.target_ids[offset : offset + microbatch_sequences]
                    local_count = int(target_view.ne(-100).sum().item())
                    if local_count < 1:
                        continue
                    item = PackedBatchV2(
                        input_ids=batch.input_ids[offset : offset + microbatch_sequences],
                        target_ids=batch.target_ids[offset : offset + microbatch_sequences],
                        document_ids=batch.document_ids[offset : offset + microbatch_sequences],
                        attention_mask=batch.attention_mask[offset : offset + microbatch_sequences],
                        completed_raw_bytes=0,
                        completed_document_count=0,
                        valid_prediction_count=local_count,
                    )
                    ids = item.input_ids.to(device)
                    documents = item.document_ids.to(device)
                    mask = item.attention_mask.to(device)
                    targets, count = _prediction_targets(item, device)
                    with _autocast(device):
                        output = model(
                            ids,
                            attention_mask=mask,
                            document_ids=documents,
                            labels=None,
                        )
                    logits = output.logits.float()
                    nll += float(
                        F.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]),
                            targets.reshape(-1),
                            ignore_index=-100,
                            reduction="sum",
                        ).item()
                    )
                    if count < 1:
                        raise GTokTrainingV2Error("held-out microbatch has no target")
            receipts.append(
                StratumNllReceipt(
                    stratum=stratum,
                    nll_nats=nll,
                    raw_byte_count=raw_bytes,
                )
            )
    model.train()
    return tuple(receipts)


@dataclass(frozen=True)
class CalibrationMeasurementV2:
    steps: int
    warmup_steps: int
    measured_steps: int
    measured_tokens: int
    measured_a100_microseconds: int
    charged_a100_microseconds: int
    measured_heldout_evaluation_a100_microseconds: int
    heldout_evaluations_per_full_run: int
    measured_output_surface_a100_microseconds: int
    output_surface_benchmarks_per_full_run: int
    planned_tokens_per_run: int
    shared_initial_state_sha256: str
    training_plan_sha256: str | None = None
    physical_prefix_packed_stream_sha256: str | None = None
    physical_prefix_compute_token_slots: int | None = None
    physical_prefix_valid_prediction_count: int | None = None
    physical_prefix_realized_raw_bytes: int | None = None
    physical_prefix_document_count: int | None = None

    def __post_init__(self) -> None:
        if (self.steps, self.warmup_steps, self.measured_steps) != (100, 20, 80):
            raise ValueError("A2 calibration must be exactly 20 warmup + 80 measured steps")
        for name in (
            "measured_tokens",
            "measured_a100_microseconds",
            "charged_a100_microseconds",
            "measured_heldout_evaluation_a100_microseconds",
            "measured_output_surface_a100_microseconds",
            "planned_tokens_per_run",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.charged_a100_microseconds < (
            self.measured_a100_microseconds
            + self.measured_heldout_evaluation_a100_microseconds
            + self.measured_output_surface_a100_microseconds
        ):
            raise ValueError(
                "charged calibration time cannot omit warmup, H evaluation, or output benchmark"
            )
        if self.heldout_evaluations_per_full_run != 3:
            raise ValueError("each full run has exactly three held-out evaluations")
        if self.output_surface_benchmarks_per_full_run != 1:
            raise ValueError("each full run has exactly one output-surface benchmark")
        if (
            len(self.shared_initial_state_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.shared_initial_state_sha256
            )
        ):
            raise ValueError("calibration requires an exact shared-state SHA-256")
        prefix_names = (
            "training_plan_sha256",
            "physical_prefix_packed_stream_sha256",
            "physical_prefix_compute_token_slots",
            "physical_prefix_valid_prediction_count",
            "physical_prefix_realized_raw_bytes",
            "physical_prefix_document_count",
        )
        prefix_values = tuple(getattr(self, name) for name in prefix_names)
        if any(value is not None for value in prefix_values):
            if any(value is None for value in prefix_values):
                raise ValueError("calibration physical-prefix evidence is incomplete")
            for name in prefix_names[:2]:
                value = getattr(self, name)
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"{name} must be SHA-256")
            for name in prefix_names[2:4]:
                value = getattr(self, name)
                if type(value) is not int or value < 1:
                    raise ValueError(f"{name} must be a positive exact integer")
            for name in prefix_names[4:]:
                value = getattr(self, name)
                if type(value) is not int or value < 0:
                    raise ValueError(f"{name} must be a non-negative exact integer")


def validate_calibration_prefix_v2(
    measurement: CalibrationMeasurementV2,
    *,
    plan: TrainingPlanV2,
) -> None:
    """Join the 100 physically executed calibration batches to the CPU plan."""

    if not isinstance(measurement, CalibrationMeasurementV2) or not isinstance(
        plan, TrainingPlanV2
    ):
        raise TypeError("calibration-prefix validation requires typed evidence")
    expected = (
        plan.receipt_sha256,
        plan.calibration_prefix_packed_stream_sha256,
        plan.calibration_prefix_compute_token_slots,
        plan.calibration_prefix_valid_prediction_count,
        plan.calibration_prefix_realized_raw_bytes,
        plan.calibration_prefix_document_count,
    )
    observed = (
        measurement.training_plan_sha256,
        measurement.physical_prefix_packed_stream_sha256,
        measurement.physical_prefix_compute_token_slots,
        measurement.physical_prefix_valid_prediction_count,
        measurement.physical_prefix_realized_raw_bytes,
        measurement.physical_prefix_document_count,
    )
    if any(value is None for value in expected) or observed != expected:
        raise GTokTrainingV2Error(
            "calibration physical prefix differs from the frozen 100-step plan"
        )


@dataclass(frozen=True)
class ProfilerOperatorFlopRowV2:
    """One positive FLOP row emitted by ``torch.profiler``."""

    operator: str
    flops_per_occurrence: int

    def __post_init__(self) -> None:
        if not isinstance(self.operator, str) or not self.operator:
            raise ValueError("profiler operator name must be nonempty")
        if type(self.flops_per_occurrence) is not int or self.flops_per_occurrence < 1:
            raise ValueError("profiler FLOP rows must be positive exact integers")


@dataclass(frozen=True)
class AnalyticUnsupportedFlopRowV2:
    """An explicit formula for arithmetic that profiler does not price.

    PyTorch deliberately leaves many fused and elementwise operators at zero
    FLOPs.  A zero is never silently interpreted as free: every such arithmetic
    family used by the fixed S0 graph is priced here under a named convention.
    """

    family: str
    flops_per_occurrence: int
    derivation: str

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("unsupported FLOP family must be named")
        if type(self.flops_per_occurrence) is not int or self.flops_per_occurrence < 1:
            raise ValueError("unsupported FLOP rows must be positive exact integers")
        if not isinstance(self.derivation, str) or "=" not in self.derivation:
            raise ValueError("unsupported FLOP row requires its literal derivation")


@dataclass(frozen=True)
class PhysicalShapeFlopReceiptV2:
    """A profiled physical optimizer-step shape and its run multiplicity."""

    batch_rows: int
    sequence_length: int
    optimizer_phase: str
    occurrences: int
    profiler_rows: tuple[ProfilerOperatorFlopRowV2, ...]
    unsupported_rows: tuple[AnalyticUnsupportedFlopRowV2, ...]
    zero_flop_profiler_operators: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("batch_rows", "sequence_length", "occurrences"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.optimizer_phase not in {"initial", "steady"}:
            raise ValueError("optimizer phase must be initial or steady")
        if not self.profiler_rows or not all(
            isinstance(row, ProfilerOperatorFlopRowV2) for row in self.profiler_rows
        ):
            raise ValueError("each physical shape needs positive profiler FLOP evidence")
        if not self.unsupported_rows or not all(
            isinstance(row, AnalyticUnsupportedFlopRowV2) for row in self.unsupported_rows
        ):
            raise ValueError("each physical shape needs an explicit unsupported-op ledger")
        if tuple(sorted(set(self.zero_flop_profiler_operators))) != (
            self.zero_flop_profiler_operators
        ):
            raise ValueError("zero-FLOP profiler inventory must be sorted and unique")

    @property
    def profiler_flops_per_occurrence(self) -> int:
        return sum(row.flops_per_occurrence for row in self.profiler_rows)

    @property
    def unsupported_flops_per_occurrence(self) -> int:
        return sum(row.flops_per_occurrence for row in self.unsupported_rows)

    @property
    def total_flops(self) -> int:
        return self.occurrences * (
            self.profiler_flops_per_occurrence + self.unsupported_flops_per_occurrence
        )


@dataclass(frozen=True)
class CompleteFlopLedgerV2:
    """Complete physical run FLOPs, including the actual terminal row count."""

    shapes: tuple[PhysicalShapeFlopReceiptV2, ...]
    optimizer_steps: int
    compute_token_slots: int
    profiler_with_flops: bool = True
    flop_binding_sha256: str = FLOP_BINDING_SHA256_V2

    def __post_init__(self) -> None:
        if not self.shapes or not all(
            isinstance(row, PhysicalShapeFlopReceiptV2) for row in self.shapes
        ):
            raise ValueError("complete FLOP ledger requires physical shape rows")
        if type(self.optimizer_steps) is not int or self.optimizer_steps < 1:
            raise ValueError("complete FLOP ledger requires positive optimizer steps")
        if type(self.compute_token_slots) is not int or self.compute_token_slots < 1:
            raise ValueError("complete FLOP ledger requires positive physical token slots")
        if type(self.profiler_with_flops) is not bool or not self.profiler_with_flops:
            raise ValueError("complete FLOP ledger requires profiler with_flops=True")
        if self.flop_binding_sha256 != FLOP_BINDING_SHA256_V2:
            raise ValueError("complete FLOP ledger binding drifted")
        if sum(row.occurrences for row in self.shapes) != self.optimizer_steps:
            raise ValueError("physical FLOP shape counts differ from optimizer steps")
        if sum(
            row.batch_rows * row.sequence_length * row.occurrences for row in self.shapes
        ) != self.compute_token_slots:
            raise ValueError("physical FLOP shape rows differ from executed token slots")
        initial = sum(
            row.occurrences for row in self.shapes if row.optimizer_phase == "initial"
        )
        if initial != 1:
            raise ValueError("complete FLOP ledger requires exactly one initial AdamW step")

    @property
    def profiler_flops(self) -> int:
        return sum(
            row.occurrences * row.profiler_flops_per_occurrence for row in self.shapes
        )

    @property
    def unsupported_flops(self) -> int:
        return sum(
            row.occurrences * row.unsupported_flops_per_occurrence for row in self.shapes
        )

    @property
    def measured_flops(self) -> int:
        return self.profiler_flops + self.unsupported_flops

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self)


def _unique_trainable_parameters_v2(model: torch.nn.Module) -> int:
    unique = {id(parameter): parameter for parameter in model.parameters() if parameter.requires_grad}
    if not unique:
        raise GTokTrainingV2Error("FLOP accounting observed no trainable parameters")
    return sum(parameter.numel() for parameter in unique.values())


def _unsupported_flop_rows_v2(
    model: torch.nn.Module,
    *,
    batch_rows: int,
    sequence_length: int,
    attention_matmuls_profiled: bool,
    profiler_rows: tuple[ProfilerOperatorFlopRowV2, ...],
) -> tuple[AnalyticUnsupportedFlopRowV2, ...]:
    """Price the fixed graph's profiler-unsupported arithmetic conventions.

    The profiler remains authoritative for supported GEMM/MM/BMM kernels.  The
    formulas below cover the known zero-FLOP families in the exact structural-
    OFF graph.  Comparisons, indexing, allocation, copies, and views are
    intentionally not called floating-point operations.
    """

    config = getattr(model, "config", None)
    required = ("d_model", "d_ff", "n_heads", "n_kv_heads", "vocab_size")
    if config is None or any(type(getattr(config, name, None)) is not int for name in required):
        raise GTokTrainingV2Error(
            "complete analytic FLOP ledger requires the bound model config"
        )
    blocks = int(getattr(config, "n_prelude_layers", 0)) + int(
        getattr(config, "n_core_blocks", 0)
    ) + int(getattr(config, "n_coda_layers", 0))
    if blocks != 10:
        raise GTokTrainingV2Error("G-TOK FLOP ledger requires the ten-block S0 graph")
    d_model = int(config.d_model)
    d_ff = int(config.d_ff)
    n_heads = int(config.n_heads)
    n_kv_heads = int(config.n_kv_heads)
    vocab = int(config.vocab_size)
    tokens = batch_rows * sequence_length
    parameters = _unique_trainable_parameters_v2(model)
    rows: list[AnalyticUnsupportedFlopRowV2] = []
    if not attention_matmuls_profiled:
        # QK^T and P@V cost 4*B*L^2*D forward. Their two reverse-mode
        # products cost another 8*B*L^2*D.
        value = 12 * blocks * batch_rows * sequence_length * sequence_length * d_model
        rows.append(
            AnalyticUnsupportedFlopRowV2(
                family="fused_scaled_dot_product_attention_forward_backward",
                flops_per_occurrence=value,
                derivation=(
                    f"12*blocks({blocks})*B({batch_rows})*L({sequence_length})^2*D({d_model})={value}"
                ),
            )
        )
    # Two stream RMSNorms plus Q/K head norms per block, and one final RMSNorm.
    # Ten is the registered forward+backward FLOP convention per normalized
    # scalar (square, reductions, rsqrt, scale and their reverse operations).
    norm_widths = blocks * (2 * d_model + d_model + d_model * n_kv_heads // n_heads) + d_model
    norm_value = 10 * tokens * norm_widths
    rows.append(
        AnalyticUnsupportedFlopRowV2(
            family="rmsnorm_forward_backward",
            flops_per_occurrence=norm_value,
            derivation=f"10*tokens({tokens})*normalized_widths({norm_widths})={norm_value}",
        )
    )
    rope_width = blocks * (d_model + d_model * n_kv_heads // n_heads)
    rope_value = 12 * tokens * rope_width
    rows.append(
        AnalyticUnsupportedFlopRowV2(
            family="rope_trigonometric_mix_forward_backward",
            flops_per_occurrence=rope_value,
            derivation=f"12*tokens({tokens})*rotated_widths({rope_width})={rope_value}",
        )
    )
    swiglu_value = 12 * blocks * tokens * d_ff
    rows.append(
        AnalyticUnsupportedFlopRowV2(
            family="swiglu_silu_gate_forward_backward",
            flops_per_occurrence=swiglu_value,
            derivation=f"12*blocks({blocks})*tokens({tokens})*Dff({d_ff})={swiglu_value}",
        )
    )
    residual_value = 4 * blocks * tokens * d_model
    rows.append(
        AnalyticUnsupportedFlopRowV2(
            family="residual_add_scale_forward_backward",
            flops_per_occurrence=residual_value,
            derivation=f"4*blocks({blocks})*tokens({tokens})*D({d_model})={residual_value}",
        )
    )
    cross_entropy_value = 8 * tokens * vocab
    rows.append(
        AnalyticUnsupportedFlopRowV2(
            family="full_softmax_cross_entropy_forward_backward",
            flops_per_occurrence=cross_entropy_value,
            derivation=f"8*tokens({tokens})*V({vocab})={cross_entropy_value}",
        )
    )
    embedding_value = tokens * d_model
    rows.append(
        AnalyticUnsupportedFlopRowV2(
            family="embedding_gradient_scatter_add",
            flops_per_occurrence=embedding_value,
            derivation=f"tokens({tokens})*D({d_model})={embedding_value}",
        )
    )
    clip_value = 4 * parameters
    rows.append(
        AnalyticUnsupportedFlopRowV2(
            family="global_gradient_norm_and_clip",
            flops_per_occurrence=clip_value,
            derivation=f"4*unique_trainable_parameters({parameters})={clip_value}",
        )
    )
    matrix_markers = ("mm", "bmm", "addmm", "matmul", "conv")
    supported_nonmatrix = sum(
        row.flops_per_occurrence
        for row in profiler_rows
        if not any(marker in row.operator for marker in matrix_markers)
    )
    # Profiler versions that price aten::add/aten::mul have already counted
    # those arithmetic operations.  Apply the positive credit once against the
    # aggregate elementwise/optimizer convention instead of double counting it.
    adamw_gross = 12 * parameters
    if supported_nonmatrix >= adamw_gross:
        raise GTokTrainingV2Error(
            "profiler-supported nonmatrix FLOPs exceed the analytic elementwise budget"
        )
    adamw_value = adamw_gross - supported_nonmatrix
    rows.append(
        AnalyticUnsupportedFlopRowV2(
            family="adamw_and_elementwise_remainder_net_of_profiler_support",
            flops_per_occurrence=adamw_value,
            derivation=(
                f"12*unique_trainable_parameters({parameters})"
                f"-profiler_supported_nonmatrix({supported_nonmatrix})={adamw_value}"
            ),
        )
    )
    return tuple(rows)


class _PhysicalFlopAccountantV2:
    """Profile actual optimizer steps without inserting a synthetic step."""

    def __init__(self, model: torch.nn.Module, *, device: torch.device) -> None:
        self.model = model
        self.device = device
        self._occurrences: dict[tuple[int, int, str], int] = {}
        self._prototypes: dict[
            tuple[int, int, str],
            tuple[
                tuple[ProfilerOperatorFlopRowV2, ...],
                tuple[AnalyticUnsupportedFlopRowV2, ...],
                tuple[str, ...],
            ],
        ] = {}

    def execute(
        self,
        *,
        batch: PackedBatchV2,
        step: int,
        operation: Callable[[], None],
    ) -> None:
        phase = "initial" if step == 1 else "steady"
        key = (int(batch.input_ids.shape[0]), int(batch.input_ids.shape[1]), phase)
        if key not in self._prototypes:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            try:
                with torch.profiler.profile(
                    activities=activities,
                    with_flops=True,
                    record_shapes=True,
                ) as profiler:
                    operation()
            except Exception as exc:
                raise GTokTrainingV2Error("torch profiler FLOP capture failed closed") from exc
            positive: list[ProfilerOperatorFlopRowV2] = []
            zero_names: set[str] = set()
            try:
                averages = profiler.key_averages()
            except Exception as exc:
                raise GTokTrainingV2Error("torch profiler emitted no readable FLOP table") from exc
            for event in averages:
                name = str(event.key)
                raw_flops = getattr(event, "flops", 0)
                if raw_flops is None:
                    raw_flops = 0
                if isinstance(raw_flops, bool) or not isinstance(raw_flops, (int, float)):
                    raise GTokTrainingV2Error("torch profiler emitted a nonnumeric FLOP value")
                if not math.isfinite(float(raw_flops)) or float(raw_flops) < 0:
                    raise GTokTrainingV2Error("torch profiler emitted an invalid FLOP value")
                rounded = int(raw_flops)
                if float(raw_flops) != float(rounded):
                    raise GTokTrainingV2Error("torch profiler emitted a fractional FLOP count")
                if rounded:
                    positive.append(
                        ProfilerOperatorFlopRowV2(
                            operator=name,
                            flops_per_occurrence=rounded,
                        )
                    )
                else:
                    zero_names.add(name)
            if not positive:
                raise GTokTrainingV2Error(
                    "torch profiler priced no operator; complete FLOP receipt is unavailable"
                )
            positive_tuple = tuple(sorted(positive, key=lambda row: row.operator))
            attention_profiled = any(
                ("bmm" in row.operator or "scaled_dot_product" in row.operator)
                for row in positive_tuple
            )
            unsupported = _unsupported_flop_rows_v2(
                self.model,
                batch_rows=key[0],
                sequence_length=key[1],
                attention_matmuls_profiled=attention_profiled,
                profiler_rows=positive_tuple,
            )
            self._prototypes[key] = (
                positive_tuple,
                unsupported,
                tuple(sorted(zero_names)),
            )
        else:
            operation()
        self._occurrences[key] = self._occurrences.get(key, 0) + 1

    def finalize(self, plan: TrainingPlanV2) -> CompleteFlopLedgerV2:
        if set(self._occurrences) != set(self._prototypes):
            raise GTokTrainingV2Error("FLOP profiler shape inventory is incomplete")
        shapes = tuple(
            PhysicalShapeFlopReceiptV2(
                batch_rows=key[0],
                sequence_length=key[1],
                optimizer_phase=key[2],
                occurrences=self._occurrences[key],
                profiler_rows=self._prototypes[key][0],
                unsupported_rows=self._prototypes[key][1],
                zero_flop_profiler_operators=self._prototypes[key][2],
            )
            for key in sorted(self._occurrences, key=lambda item: (item[2], item[0], item[1]))
        )
        return CompleteFlopLedgerV2(
            shapes=shapes,
            optimizer_steps=plan.optimizer_steps,
            compute_token_slots=plan.compute_token_slots,
        )


@dataclass(frozen=True)
class StratumCompressionMetricsV2:
    stratum: str
    raw_bytes: int
    encoded_tokens: int
    document_count: int
    coverage_2048_raw_bytes_p50: int
    coverage_2048_raw_bytes_p95: int

    def __post_init__(self) -> None:
        if self.stratum not in GTOK_STRATA:
            raise ValueError("compression metric uses an unknown stratum")
        for name in ("raw_bytes", "encoded_tokens", "document_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        for name in ("coverage_2048_raw_bytes_p50", "coverage_2048_raw_bytes_p95"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.coverage_2048_raw_bytes_p50 > self.coverage_2048_raw_bytes_p95:
            raise ValueError("P50 coverage cannot exceed P95 coverage")

    @property
    def bytes_per_token(self) -> float:
        return self.raw_bytes / self.encoded_tokens


@dataclass(frozen=True)
class TokenizerCorpusMetricsV2:
    strata: tuple[StratumCompressionMetricsV2, ...]
    nonreserved_row_count: int
    undertrained_row_count: int
    undertrained_threshold: int
    training_encoded_tokens: int
    scanned_training_documents: int
    scanned_heldout_documents: int
    exact_byte_round_trip_passed: bool
    measurement_binding_sha256: str = MEASUREMENT_BINDING_SHA256_V2

    def __post_init__(self) -> None:
        if tuple(row.stratum for row in self.strata) != GTOK_STRATA:
            raise ValueError("compression receipt must contain the registered strata in order")
        for name in (
            "nonreserved_row_count",
            "training_encoded_tokens",
            "scanned_training_documents",
            "scanned_heldout_documents",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        if (
            type(self.undertrained_row_count) is not int
            or not 0 <= self.undertrained_row_count <= self.nonreserved_row_count
        ):
            raise ValueError("undertrained row count lies outside the nonreserved vocabulary")
        if self.undertrained_threshold != 1_000:
            raise ValueError("undertrained-row threshold is fixed at fewer than 1000 T hits")
        if type(self.exact_byte_round_trip_passed) is not bool:
            raise TypeError("exact byte round-trip result must be boolean")
        if not self.exact_byte_round_trip_passed:
            raise ValueError("G-TOK metrics cannot mint after a byte round-trip failure")
        if self.measurement_binding_sha256 != MEASUREMENT_BINDING_SHA256_V2:
            raise ValueError("tokenizer corpus measurement binding drifted")

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self)


@dataclass(frozen=True)
class OutputSurfacePerformanceV2:
    scope: str
    batch_size: int
    context_tokens: int
    decode_tokens: int
    warmup_trials: int
    timed_trials: int
    timed_token_count: int
    measured_microseconds: int

    def __post_init__(self) -> None:
        if self.scope not in {
            "full_softmax_throughput",
            "output_projection_full_softmax_decode_surface_latency",
        }:
            raise ValueError("unknown output-surface performance scope")
        expected_decode = 0 if self.scope == "full_softmax_throughput" else 128
        if (
            self.batch_size != 1
            or self.context_tokens != 2_048
            or self.decode_tokens != expected_decode
            or self.warmup_trials != 20
            or self.timed_trials != 100
        ):
            raise ValueError("output-surface benchmark shape or trial count drifted")
        expected_tokens = (
            self.timed_trials * self.context_tokens
            if self.scope == "full_softmax_throughput"
            else self.timed_trials * self.decode_tokens
        )
        if self.timed_token_count != expected_tokens:
            raise ValueError("output-surface timed token count drifted")
        if type(self.measured_microseconds) is not int or self.measured_microseconds < 1:
            raise ValueError("output-surface time must be a positive exact integer")

    @property
    def tokens_per_second(self) -> float:
        return self.timed_token_count * 1_000_000 / self.measured_microseconds

    @property
    def microseconds_per_decoded_token(self) -> float | None:
        if self.scope != "output_projection_full_softmax_decode_surface_latency":
            return None
        return self.measured_microseconds / self.timed_token_count


@dataclass(frozen=True)
class VocabularyFractionRowV2:
    rung: str
    vocabulary_parameters: int
    total_unique_parameters: int

    def __post_init__(self) -> None:
        if self.rung not in {"proxy", "target_a", "target_b"}:
            raise ValueError("unknown vocabulary-fraction rung")
        if (
            type(self.vocabulary_parameters) is not int
            or type(self.total_unique_parameters) is not int
            or not 0 < self.vocabulary_parameters < self.total_unique_parameters
        ):
            raise ValueError("vocabulary fraction requires positive exact parameter counts")

    @property
    def fraction(self) -> float:
        return self.vocabulary_parameters / self.total_unique_parameters


@dataclass(frozen=True)
class ArmMeasurementPanelV2:
    tokenizer_corpus: TokenizerCorpusMetricsV2
    full_softmax: OutputSurfacePerformanceV2
    decode: OutputSurfacePerformanceV2
    vocabulary_fractions: tuple[VocabularyFractionRowV2, ...]
    complete_measured_flops: int
    flop_ledger_sha256: str
    measurement_binding_sha256: str = MEASUREMENT_BINDING_SHA256_V2

    def __post_init__(self) -> None:
        if self.full_softmax.scope != "full_softmax_throughput":
            raise ValueError("arm panel lacks full-softmax throughput")
        if (
            self.decode.scope
            != "output_projection_full_softmax_decode_surface_latency"
        ):
            raise ValueError(
                "arm panel lacks output-projection/full-softmax decode-surface latency"
            )
        if tuple(row.rung for row in self.vocabulary_fractions) != (
            "proxy",
            "target_a",
            "target_b",
        ):
            raise ValueError("arm panel requires proxy and both target vocabulary fractions")
        vocab = self.tokenizer_corpus.nonreserved_row_count + 64
        expected_vocabulary_parameters = (vocab * 512, vocab * 1_024, vocab * 1_024)
        if (
            tuple(row.vocabulary_parameters for row in self.vocabulary_fractions)
            != expected_vocabulary_parameters
        ):
            raise ValueError("arm panel vocabulary fractions use a different tokenizer width")
        target_delta = (vocab - 32_768) * 1_024
        if tuple(row.total_unique_parameters for row in self.vocabulary_fractions[1:]) != (
            302_900_000 + target_delta,
            305_800_000 + target_delta,
        ):
            raise ValueError("arm panel target-rung denominators drifted")
        if type(self.complete_measured_flops) is not int or self.complete_measured_flops < 1:
            raise ValueError("arm panel requires positive complete measured FLOPs")
        if (
            len(self.flop_ledger_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.flop_ledger_sha256)
        ):
            raise ValueError("arm panel requires an exact FLOP ledger SHA-256")
        if self.measurement_binding_sha256 != MEASUREMENT_BINDING_SHA256_V2:
            raise ValueError("arm measurement binding drifted")

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self)


def _nearest_rank_percentile_v2(values: list[int], percentile: int) -> int:
    if not values or percentile not in {50, 95}:
        raise ValueError("coverage percentile requires values and P50 or P95")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered) / 100) - 1)
    return ordered[index]


def _exact_token_prefix_raw_bytes_v2(
    tokenizer: Tokenizer,
    text: str,
    *,
    token_limit: int = 2_048,
) -> tuple[int, tuple[int, ...]]:
    encoding = tokenizer.encode(text, add_special_tokens=False)
    ids = tuple(int(value) for value in encoding.ids)
    decoded = tokenizer.decode(list(ids), skip_special_tokens=False)
    if decoded.encode("utf-8", errors="strict") != text.encode("utf-8", errors="strict"):
        raise GTokTrainingV2Error("tokenizer failed exact UTF-8 byte round-trip")
    if len(ids) <= token_limit:
        return len(text.encode("utf-8", errors="strict")), ids
    prefix_ids = ids[:token_limit]
    prefix = tokenizer.decode(list(prefix_ids), skip_special_tokens=False)
    if not text.startswith(prefix):
        # Offset fallback is permitted only when it re-encodes to precisely the
        # first registered token IDs; otherwise coverage is ambiguous.
        offsets = encoding.offsets
        if len(offsets) < token_limit:
            raise GTokTrainingV2Error("tokenizer coverage offsets are incomplete")
        endpoint = int(offsets[token_limit - 1][1])
        if not 0 <= endpoint <= len(text):
            raise GTokTrainingV2Error("tokenizer coverage offset lies outside the document")
        prefix = text[:endpoint]
    replay = tuple(
        int(value) for value in tokenizer.encode(prefix, add_special_tokens=False).ids
    )
    if replay != prefix_ids or not text.startswith(prefix):
        raise GTokTrainingV2Error("2048-token raw-byte coverage is not replay-exact")
    return len(prefix.encode("utf-8", errors="strict")), ids


def measure_tokenizer_corpus_metrics_v2(
    *,
    tokenizer: Tokenizer,
    training_document_factory: Callable[[], Iterable[TrainingDocumentV2]],
    heldout_factory: Callable[[str], Iterable[TrainingDocumentV2]],
) -> TokenizerCorpusMetricsV2:
    """Measure literal §6.8 compression and row-training statistics."""

    reserved = len(special_token_strings())
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if reserved != 64 or vocab_size <= reserved:
        raise GTokTrainingV2Error("tokenizer reserved/nonreserved row partition drifted")
    row_counts = [0] * vocab_size
    training_tokens = 0
    training_documents = 0
    for document in training_document_factory():
        _coverage, ids = _exact_token_prefix_raw_bytes_v2(tokenizer, document.text)
        training_documents += 1
        training_tokens += len(ids)
        for token_id in ids:
            if not 0 <= token_id < vocab_size:
                raise GTokTrainingV2Error("tokenizer emitted an ID outside its vocabulary")
            row_counts[token_id] += 1
    if training_documents < 1 or training_tokens < 1:
        raise GTokTrainingV2Error("training corpus metrics scanned no encoded content")
    strata: list[StratumCompressionMetricsV2] = []
    heldout_documents = 0
    for stratum in GTOK_STRATA:
        raw_bytes = 0
        encoded_tokens = 0
        coverage: list[int] = []
        documents = 0
        for document in heldout_factory(stratum):
            if document.stratum != stratum:
                raise GTokTrainingV2Error("held-out metric factory crossed strata")
            covered, ids = _exact_token_prefix_raw_bytes_v2(tokenizer, document.text)
            raw_bytes += len(document.raw_bytes)
            encoded_tokens += len(ids)
            coverage.append(covered)
            documents += 1
        if raw_bytes < 1 or encoded_tokens < 1 or documents < 1:
            raise GTokTrainingV2Error(f"held-out {stratum} metric stream is empty")
        heldout_documents += documents
        strata.append(
            StratumCompressionMetricsV2(
                stratum=stratum,
                raw_bytes=raw_bytes,
                encoded_tokens=encoded_tokens,
                document_count=documents,
                coverage_2048_raw_bytes_p50=_nearest_rank_percentile_v2(coverage, 50),
                coverage_2048_raw_bytes_p95=_nearest_rank_percentile_v2(coverage, 95),
            )
        )
    undertrained = sum(count < 1_000 for count in row_counts[reserved:])
    return TokenizerCorpusMetricsV2(
        strata=tuple(strata),
        nonreserved_row_count=vocab_size - reserved,
        undertrained_row_count=undertrained,
        undertrained_threshold=1_000,
        training_encoded_tokens=training_tokens,
        scanned_training_documents=training_documents,
        scanned_heldout_documents=heldout_documents,
        exact_byte_round_trip_passed=True,
    )


def _run_output_surface_trial_v2(
    head: torch.nn.Module,
    hidden: torch.Tensor,
    *,
    device: torch.device,
) -> None:
    with _autocast(device):
        logits = head(hidden)
    # Full means every vocabulary logit participates; selecting an argmax alone
    # would not measure the softmax cost that changes with V.
    normalized = torch.log_softmax(logits.float(), dim=-1)
    if normalized.numel() < 1:
        raise GTokTrainingV2Error("full-softmax benchmark emitted no logits")


def measure_output_surface_performance_v2(
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> tuple[OutputSurfacePerformanceV2, OutputSurfacePerformanceV2]:
    """Benchmark the V-dependent output surface under the fixed §6.8 shapes."""

    head = getattr(model, "lm_head", None)
    width = getattr(head, "in_features", None)
    if not isinstance(head, torch.nn.Module) or type(width) is not int or width < 1:
        raise GTokTrainingV2Error("output-surface benchmark requires the bound LM head")
    was_training = model.training
    model.to(device)
    model.eval()
    context_hidden = torch.zeros((1, 2_048, width), dtype=torch.float32, device=device)
    decode_hidden = torch.zeros((1, 1, width), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for _ in range(20):
            _run_output_surface_trial_v2(head, context_hidden, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        full_start = time.perf_counter_ns()
        for _ in range(100):
            _run_output_surface_trial_v2(head, context_hidden, device=device)
        full_elapsed = _elapsed_microseconds(full_start, device)

        for _ in range(20):
            for _token in range(128):
                _run_output_surface_trial_v2(head, decode_hidden, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        decode_start = time.perf_counter_ns()
        for _ in range(100):
            for _token in range(128):
                _run_output_surface_trial_v2(head, decode_hidden, device=device)
        decode_elapsed = _elapsed_microseconds(decode_start, device)
    model.train(was_training)
    return (
        OutputSurfacePerformanceV2(
            scope="full_softmax_throughput",
            batch_size=1,
            context_tokens=2_048,
            decode_tokens=0,
            warmup_trials=20,
            timed_trials=100,
            timed_token_count=204_800,
            measured_microseconds=full_elapsed,
        ),
        OutputSurfacePerformanceV2(
            scope="output_projection_full_softmax_decode_surface_latency",
            batch_size=1,
            context_tokens=2_048,
            decode_tokens=128,
            warmup_trials=20,
            timed_trials=100,
            timed_token_count=12_800,
            measured_microseconds=decode_elapsed,
        ),
    )


def measure_vocabulary_fractions_v2(
    model: torch.nn.Module,
) -> tuple[VocabularyFractionRowV2, ...]:
    config = getattr(model, "config", None)
    vocab = getattr(config, "vocab_size", None)
    proxy_width = getattr(config, "d_model", None)
    if type(vocab) is not int or type(proxy_width) is not int:
        raise GTokTrainingV2Error("vocabulary-fraction receipt requires the bound model config")
    proxy_total = _unique_trainable_parameters_v2(model)
    target_vocab_delta = (vocab - 32_768) * 1_024
    target_a_total = 302_900_000 + target_vocab_delta
    target_b_total = 305_800_000 + target_vocab_delta
    return (
        VocabularyFractionRowV2(
            rung="proxy",
            vocabulary_parameters=vocab * proxy_width,
            total_unique_parameters=proxy_total,
        ),
        VocabularyFractionRowV2(
            rung="target_a",
            vocabulary_parameters=vocab * 1_024,
            total_unique_parameters=target_a_total,
        ),
        VocabularyFractionRowV2(
            rung="target_b",
            vocabulary_parameters=vocab * 1_024,
            total_unique_parameters=target_b_total,
        ),
    )


def _execute_optimizer_step_v2(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    *,
    batch: PackedBatchV2,
    step: int,
    plan: TrainingPlanV2,
    device: torch.device,
    microbatch_sequences: int,
) -> None:
    """Execute one literal global step, including an actual-size terminal batch."""

    optimizer.zero_grad(set_to_none=True)
    global_valid = batch.valid_prediction_count
    for offset in range(0, batch.input_ids.shape[0], microbatch_sequences):
        target_view = batch.target_ids[offset : offset + microbatch_sequences]
        local_count = int(target_view.ne(-100).sum().item())
        if local_count < 1:
            continue
        ids = batch.input_ids[offset : offset + microbatch_sequences].to(device)
        documents = batch.document_ids[offset : offset + microbatch_sequences].to(device)
        mask = batch.attention_mask[offset : offset + microbatch_sequences].to(device)
        targets = target_view.to(device)
        with _autocast(device):
            output = model(
                ids,
                attention_mask=mask,
                document_ids=documents,
                labels=None,
            )
        logits = output.logits.float()
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        (loss_sum / global_valid).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    learning_rate = learning_rate_for_step_v2(step, plan.optimizer_steps)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.step()


@dataclass(frozen=True)
class FullRunMeasurementV2:
    run: GTokRunReceiptV2
    flop_ledger: CompleteFlopLedgerV2
    measurement_panel: ArmMeasurementPanelV2
    training_runtime_receipt_sha256: str
    code_closure_receipt_sha256: str
    training_plan_sha256: str
    packing_binding_sha256: str
    schedule_binding_sha256: str
    flop_binding_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.flop_ledger, CompleteFlopLedgerV2):
            raise TypeError("full run requires its complete physical FLOP ledger")
        if not isinstance(self.measurement_panel, ArmMeasurementPanelV2):
            raise TypeError("full run requires its complete section 6.8 measurement panel")
        if self.run.measured_flops != self.flop_ledger.measured_flops:
            raise ValueError("run FLOPs differ from the complete physical ledger")
        if self.measurement_panel.complete_measured_flops != self.run.measured_flops:
            raise ValueError("section 6.8 panel FLOPs differ from the run receipt")
        if self.measurement_panel.flop_ledger_sha256 != self.flop_ledger.receipt_sha256:
            raise ValueError("section 6.8 panel points to a different FLOP ledger")
        if (
            self.measurement_panel.tokenizer_corpus.nonreserved_row_count + 64
            != self.run.vocab_size
        ):
            raise ValueError("section 6.8 panel tokenizer differs from the run vocabulary")
        for name in (
            "training_runtime_receipt_sha256",
            "code_closure_receipt_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"full run requires one exact {name}")


def _train_batches_v2(
    model: torch.nn.Module,
    *,
    tokenizer: Tokenizer,
    document_factory: Callable[[], Iterable[TrainingDocumentV2]],
    plan: TrainingPlanV2,
    device: torch.device,
    microbatch_sequences: int,
    maximum_steps: int | None,
    watchdog_limit_a100_microseconds: int | None,
    prior_campaign_a100_microseconds: int,
    campaign_tripwire_a100_microseconds: int | None,
    heldout_factory: Callable[[str], Iterable[TrainingDocumentV2]] | None,
    heldout_stream_sha256: str | None,
) -> tuple[
    int,
    int,
    int,
    tuple[BpbMilestoneReceiptV2, ...],
    CompleteFlopLedgerV2,
]:
    if 256 % microbatch_sequences:
        raise ValueError("microbatch_sequences must be a positive divisor of 256")
    optimizer = build_flat_a1_adamw_v2(model)
    model.to(device)
    model.train()
    flop_accountant = _PhysicalFlopAccountantV2(model, device=device)
    start_ns = time.perf_counter_ns()
    raw_total = 0
    documents_total = 0
    observations: list[BpbMilestoneReceiptV2] = []
    steps_executed = 0
    slots_total = 0
    predictions_total = 0
    packed_digest = hashlib.sha256()
    milestone_thresholds = (
        (GTOK_MILESTONE_LABELS[0], GTOK_FIRST_BOUNDARY_BYTES),
        (GTOK_MILESTONE_LABELS[1], GTOK_SECOND_BOUNDARY_BYTES),
    )

    def checked_elapsed() -> int:
        elapsed_value = _elapsed_microseconds(start_ns, device)
        if (
            watchdog_limit_a100_microseconds is not None
            and elapsed_value > watchdog_limit_a100_microseconds
        ):
            raise GTokRunWatchdogV2(elapsed_value)
        if (
            campaign_tripwire_a100_microseconds is not None
            and prior_campaign_a100_microseconds + elapsed_value
            > campaign_tripwire_a100_microseconds
        ):
            raise GTokCampaignTripwireV2(
                prior_campaign_a100_microseconds + elapsed_value
            )
        return elapsed_value

    for step, batch in enumerate(
        iter_packed_global_batches_v2(document_factory(), tokenizer=tokenizer), start=1
    ):
        if maximum_steps is not None and step > maximum_steps:
            break
        previous_raw = raw_total
        _update_training_plan_digest_v2(packed_digest, batch)
        slots_total += batch.input_ids.numel()
        predictions_total += batch.valid_prediction_count
        flop_accountant.execute(
            batch=batch,
            step=step,
            operation=lambda batch=batch, step=step: _execute_optimizer_step_v2(
                model,
                optimizer,
                batch=batch,
                step=step,
                plan=plan,
                device=device,
                microbatch_sequences=microbatch_sequences,
            ),
        )
        steps_executed = step
        raw_total += batch.completed_raw_bytes
        documents_total += batch.completed_document_count
        checked_elapsed()
        if heldout_factory is not None and heldout_stream_sha256 is not None:
            for label, threshold in milestone_thresholds:
                if (
                    label not in {item.label for item in observations}
                    and previous_raw < threshold <= raw_total
                ):
                    observations.append(
                        BpbMilestoneReceiptV2(
                            label=label,
                            optimizer_step=step,
                            previous_training_raw_bytes=previous_raw,
                            training_raw_bytes=raw_total,
                            heldout_stream_sha256=heldout_stream_sha256,
                            strata=evaluate_heldout_v2(
                                model,
                                tokenizer=tokenizer,
                                heldout_factory=heldout_factory,
                                device=device,
                                microbatch_sequences=microbatch_sequences,
                            ),
                        )
                    )
                    # Evaluation is part of the same watchdog and cumulative
                    # A100-hour meter, so re-read after it completes.
                    checked_elapsed()
    if maximum_steps is None:
        if (
            steps_executed != plan.optimizer_steps
            or slots_total != plan.compute_token_slots
            or predictions_total != plan.valid_prediction_count
            or raw_total != plan.realized_raw_bytes
            or documents_total != plan.document_count
            or packed_digest.hexdigest() != plan.packed_stream_sha256
        ):
            raise GTokTrainingV2Error("full run did not exhaust its frozen training plan")
        if heldout_factory is None or heldout_stream_sha256 is None:
            raise GTokTrainingV2Error("full run requires frozen held-out evaluation")
        if tuple(item.label for item in observations) != GTOK_MILESTONE_LABELS[:2]:
            raise GTokTrainingV2Error("full run failed to cross both preregistered byte milestones")
        observations.append(
            BpbMilestoneReceiptV2(
                label=GTOK_MILESTONE_LABELS[2],
                optimizer_step=steps_executed,
                previous_training_raw_bytes=observations[-1].training_raw_bytes,
                training_raw_bytes=raw_total,
                heldout_stream_sha256=heldout_stream_sha256,
                strata=evaluate_heldout_v2(
                    model,
                    tokenizer=tokenizer,
                    heldout_factory=heldout_factory,
                    device=device,
                    microbatch_sequences=microbatch_sequences,
                ),
            )
        )
        # The terminal held-out pass is charged before the receipt snapshots
        # elapsed time.
        checked_elapsed()
    elapsed = checked_elapsed()
    flop_ledger = flop_accountant.finalize(plan)
    if steps_executed != flop_ledger.optimizer_steps:
        raise GTokTrainingV2Error("complete FLOP ledger differs from executed steps")
    return steps_executed, elapsed, documents_total, tuple(observations), flop_ledger


def calibrate_arm_v2(
    *,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    document_factory: Callable[[], Iterable[TrainingDocumentV2]],
    plan: TrainingPlanV2,
    device: torch.device,
    microbatch_sequences: int,
    heldout_factory: Callable[[str], Iterable[TrainingDocumentV2]] | None = None,
    calibration_steps: int = 100,
    allow_nonproduction_cpu: bool = False,
) -> CalibrationMeasurementV2:
    _require_production_a100(device, allow_nonproduction_cpu=allow_nonproduction_cpu)
    shared = shared_nonvocabulary_state_sha256_v2(model)
    if calibration_steps != 100:
        raise GTokTrainingV2Error("A2 calibration step count is fixed at exactly 100")
    if plan.optimizer_steps < 100:
        raise GTokTrainingV2Error("training plan is too short for the literal 100-step calibration")
    if 256 % microbatch_sequences:
        raise ValueError("microbatch_sequences must be a positive divisor of 256")
    optimizer = build_flat_a1_adamw_v2(model)
    model.to(device)
    model.train()
    iterator = iter(iter_packed_global_batches_v2(document_factory(), tokenizer=tokenizer))
    prefix_digest = hashlib.sha256()
    prefix_slots = 0
    prefix_predictions = 0
    prefix_raw_bytes = 0
    prefix_documents = 0

    def record_prefix(batch: PackedBatchV2) -> None:
        nonlocal prefix_slots, prefix_predictions, prefix_raw_bytes, prefix_documents
        _update_training_plan_digest_v2(prefix_digest, batch)
        prefix_slots += batch.input_ids.numel()
        prefix_predictions += batch.valid_prediction_count
        prefix_raw_bytes += batch.completed_raw_bytes
        prefix_documents += batch.completed_document_count

    charged_start_ns = time.perf_counter_ns()
    for step in range(1, 21):
        batch = next(iterator)
        record_prefix(batch)
        _execute_optimizer_step_v2(
            model,
            optimizer,
            batch=batch,
            step=step,
            plan=plan,
            device=device,
            microbatch_sequences=microbatch_sequences,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measured_start_ns = time.perf_counter_ns()
    measured_tokens = 0
    for step in range(21, 101):
        batch = next(iterator)
        record_prefix(batch)
        measured_tokens += batch.input_ids.numel()
        _execute_optimizer_step_v2(
            model,
            optimizer,
            batch=batch,
            step=step,
            plan=plan,
            device=device,
            microbatch_sequences=microbatch_sequences,
        )
    prefix_observed = (
        plan.receipt_sha256,
        prefix_digest.hexdigest(),
        prefix_slots,
        prefix_predictions,
        prefix_raw_bytes,
        prefix_documents,
    )
    prefix_expected = (
        plan.receipt_sha256,
        plan.calibration_prefix_packed_stream_sha256,
        plan.calibration_prefix_compute_token_slots,
        plan.calibration_prefix_valid_prediction_count,
        plan.calibration_prefix_realized_raw_bytes,
        plan.calibration_prefix_document_count,
    )
    if any(value is None for value in prefix_expected) or prefix_observed != prefix_expected:
        raise GTokTrainingV2Error(
            "calibration physical prefix differs from the frozen 100-step plan"
        )
    measured = _elapsed_microseconds(measured_start_ns, device)
    if heldout_factory is None:
        raise GTokTrainingV2Error("production calibration must benchmark one frozen-H evaluation")
    evaluation_start_ns = time.perf_counter_ns()
    evaluate_heldout_v2(
        model,
        tokenizer=tokenizer,
        heldout_factory=heldout_factory,
        device=device,
        microbatch_sequences=microbatch_sequences,
    )
    evaluation = _elapsed_microseconds(evaluation_start_ns, device)
    output_surface_start_ns = time.perf_counter_ns()
    measure_output_surface_performance_v2(model, device=device)
    output_surface = _elapsed_microseconds(output_surface_start_ns, device)
    charged = _elapsed_microseconds(charged_start_ns, device)
    measurement = CalibrationMeasurementV2(
        steps=100,
        warmup_steps=20,
        measured_steps=80,
        measured_tokens=measured_tokens,
        measured_a100_microseconds=measured,
        charged_a100_microseconds=charged,
        measured_heldout_evaluation_a100_microseconds=evaluation,
        heldout_evaluations_per_full_run=3,
        measured_output_surface_a100_microseconds=output_surface,
        output_surface_benchmarks_per_full_run=1,
        planned_tokens_per_run=plan.compute_token_slots,
        shared_initial_state_sha256=shared,
        training_plan_sha256=plan.receipt_sha256,
        physical_prefix_packed_stream_sha256=prefix_digest.hexdigest(),
        physical_prefix_compute_token_slots=prefix_slots,
        physical_prefix_valid_prediction_count=prefix_predictions,
        physical_prefix_realized_raw_bytes=prefix_raw_bytes,
        physical_prefix_document_count=prefix_documents,
    )
    validate_calibration_prefix_v2(measurement, plan=plan)
    return measurement


def execute_full_run_v2(
    *,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    tokenizer_receipt: TokenizerArmReceiptV2,
    corpus: FrozenScreenCorpusV2,
    document_factory: Callable[[], Iterable[TrainingDocumentV2]],
    heldout_factory: Callable[[str], Iterable[TrainingDocumentV2]],
    plan: TrainingPlanV2,
    seed: int,
    initialization_seed: int,
    data_order_seed: int,
    data_order_sha256: str,
    compute_attempt_id: str,
    training_runtime_receipt_sha256: str,
    code_closure_receipt_sha256: str,
    gpu_uuid_provenance: str | None = None,
    watchdog_limit_a100_microseconds: int,
    prior_campaign_a100_microseconds: int = 0,
    campaign_tripwire_a100_microseconds: int | None = None,
    device: torch.device,
    microbatch_sequences: int,
    tokenizer_corpus_metrics: TokenizerCorpusMetricsV2 | None = None,
    allow_nonproduction_cpu: bool = False,
) -> FullRunMeasurementV2:
    _require_production_a100(device, allow_nonproduction_cpu=allow_nonproduction_cpu)
    if tokenizer_receipt.vocab_size != model.config.vocab_size:  # type: ignore[attr-defined]
        raise GTokTrainingV2Error("tokenizer arm differs from model vocabulary")
    shared = shared_nonvocabulary_state_sha256_v2(model)
    steps, elapsed, _documents, observations, flop_ledger = _train_batches_v2(
        model,
        tokenizer=tokenizer,
        document_factory=document_factory,
        plan=plan,
        device=device,
        microbatch_sequences=microbatch_sequences,
        maximum_steps=None,
        watchdog_limit_a100_microseconds=watchdog_limit_a100_microseconds,
        prior_campaign_a100_microseconds=prior_campaign_a100_microseconds,
        campaign_tripwire_a100_microseconds=campaign_tripwire_a100_microseconds,
        heldout_factory=heldout_factory,
        heldout_stream_sha256=corpus.heldout_stream_sha256,
    )
    if steps != plan.optimizer_steps:
        raise GTokTrainingV2Error("full-run optimizer steps differ from the frozen plan")
    measurement_start_ns = time.perf_counter_ns()
    if tokenizer_corpus_metrics is None:
        tokenizer_corpus_metrics = measure_tokenizer_corpus_metrics_v2(
            tokenizer=tokenizer,
            training_document_factory=document_factory,
            heldout_factory=heldout_factory,
        )
    full_softmax, decode = measure_output_surface_performance_v2(model, device=device)
    vocabulary_fractions = measure_vocabulary_fractions_v2(model)
    elapsed += _elapsed_microseconds(measurement_start_ns, device)
    if elapsed > watchdog_limit_a100_microseconds:
        raise GTokRunWatchdogV2(elapsed)
    if (
        campaign_tripwire_a100_microseconds is not None
        and prior_campaign_a100_microseconds + elapsed
        > campaign_tripwire_a100_microseconds
    ):
        raise GTokCampaignTripwireV2(prior_campaign_a100_microseconds + elapsed)
    measurement_panel = ArmMeasurementPanelV2(
        tokenizer_corpus=tokenizer_corpus_metrics,
        full_softmax=full_softmax,
        decode=decode,
        vocabulary_fractions=vocabulary_fractions,
        complete_measured_flops=flop_ledger.measured_flops,
        flop_ledger_sha256=flop_ledger.receipt_sha256,
    )
    receipt = GTokRunReceiptV2(
        vocab_size=tokenizer_receipt.vocab_size,
        seed=seed,
        frozen_screen_corpus_sha256=corpus.receipt_sha256,
        tokenizer_receipt_sha256=tokenizer_receipt.receipt_sha256,
        initialization_recipe_sha256=INITIALIZATION_RECIPE_SHA256_V2,
        initialization_seed=initialization_seed,
        shared_initial_state_sha256=shared,
        data_order_seed=data_order_seed,
        data_order_sha256=data_order_sha256,
        training_runtime_receipt_sha256=training_runtime_receipt_sha256,
        code_closure_receipt_sha256=code_closure_receipt_sha256,
        gpu_uuid_provenance=gpu_uuid_provenance,
        compute_attempt_id=compute_attempt_id,
        measured_a100_microseconds=elapsed,
        measured_flops=flop_ledger.measured_flops,
        optimizer=a1_flat_adamw_recipe(),
        observations=observations,
        byte_round_trip_passed=tokenizer_corpus_metrics.exact_byte_round_trip_passed,
    )
    return FullRunMeasurementV2(
        run=receipt,
        flop_ledger=flop_ledger,
        measurement_panel=measurement_panel,
        training_runtime_receipt_sha256=training_runtime_receipt_sha256,
        code_closure_receipt_sha256=code_closure_receipt_sha256,
        training_plan_sha256=plan.receipt_sha256,
        packing_binding_sha256=PACKING_BINDING_SHA256_V2,
        schedule_binding_sha256=SCHEDULE_BINDING_SHA256_V2,
        flop_binding_sha256=FLOP_BINDING_SHA256_V2,
    )


@dataclass(frozen=True)
class V4CorpusSourceV2:
    """Physical V4 T/H adapter with registered raw-content identities."""

    root: Path
    physical_d6_evidence_sha256: str
    training_order_receipts: tuple[tuple[int, int, str], ...]
    training_raw_bytes: int
    heldout_raw_bytes_by_stratum: tuple[tuple[str, int], ...]

    def training_documents(self, seed: int) -> Iterator[TrainingDocumentV2]:
        by_training_seed = {
            training_seed: (data_order_seed, order_sha256)
            for training_seed, data_order_seed, order_sha256 in self.training_order_receipts
        }
        if seed not in by_training_seed:
            raise GTokTrainingV2Error("unregistered V4 training seed")
        expected_data_seed = GTOK_DATA_ORDER_SEED_BY_TRAINING_SEED_V4.get(seed)
        if by_training_seed[seed][0] != expected_data_seed:
            raise GTokTrainingV2Error(
                "V4 row key is not joined to its governed data-order seed"
            )
        # V4's public iterator validates the raw-content-ID ordering receipt
        # before yielding, then validates its exhaustion after yielding.
        data_order_seed, order_sha256 = by_training_seed[seed]
        for text in iter_materialized_training_texts_v4(
            self.root,
            training_seed=seed,
            expected_physical_d6_evidence_sha256=self.physical_d6_evidence_sha256,
            expected_consumer_order_receipt=(
                seed,
                data_order_seed,
                order_sha256,
            ),
        ):
            raw = text.encode("utf-8", errors="strict")
            yield TrainingDocumentV2(
                raw_content_id=hashlib.sha1(raw).hexdigest(),  # noqa: S324
                text=text,
                stratum="general",  # Training loss is pooled; physical order proves T.
            )

    def heldout_documents(self, stratum: str) -> Iterator[TrainingDocumentV2]:
        if stratum not in GTOK_STRATA:
            raise ValueError("unknown held-out stratum")
        assert_current_physical_d6_identity_v4(
            self.root,
            expected_physical_sha256=self.physical_d6_evidence_sha256,
        )
        _manifest_sha, rows = _load_screen_shard_manifest_v4(self.root)
        for row in rows:
            if row["stream"] != "H" or row["stratum"] != stratum:
                continue
            identity = JsonlZstdShardIdentityV3(
                relative_path=str(row["identity_relative_path"]),
                record_count=int(row["record_count"]),
                retained_text_bytes=int(row["retained_text_bytes"]),
                logical_jsonl_sha256=str(row["logical_jsonl_sha256"]),
                logical_jsonl_bytes=int(row["logical_jsonl_bytes"]),
                zstd_sha256=str(row["zstd_sha256"]),
                zstd_bytes=int(row["zstd_bytes"]),
                codec_binding_sha256=str(row["codec_binding_sha256"]),
            )
            for text in iter_a2_shard_texts(self.root / "shards", (identity,)):
                raw = text.encode("utf-8", errors="strict")
                yield TrainingDocumentV2(
                    raw_content_id=hashlib.sha1(raw).hexdigest(),  # noqa: S324
                    text=text,
                    stratum=stratum,
                )


def load_v4_corpus_source_v2(root: Path, *, sqlite_path: Path) -> V4CorpusSourceV2:
    resolved = assert_no_symlink_ancestors(root).resolve(strict=True)
    evidence, physical_sha = validate_physical_d6_evidence_v4(
        root=resolved,
        sqlite_path=sqlite_path,
    )
    orders = evidence.get("consumer_order_receipts")
    streams = evidence.get("stream_identities")
    groups = evidence.get("split_groups")
    if not isinstance(orders, list) or not isinstance(streams, list) or not isinstance(groups, list):
        raise GTokTrainingV2Error("physical V4 evidence lacks consumer bindings")
    order_rows = tuple(
        (
            int(row["training_seed"]),
            int(row["data_order_seed"]),
            str(row["ordered_raw_content_ids_sha256"]),
        )
        for row in orders
        if isinstance(row, Mapping)
    )
    if tuple(seed for seed, _, _ in order_rows) != tuple(GTOK_TRAINING_SEEDS):
        raise GTokTrainingV2Error("physical V4 evidence training seed order drifted")
    if any(
        data_order_seed != GTOK_DATA_ORDER_SEED_BY_TRAINING_SEED_V4[training_seed]
        for training_seed, data_order_seed, _ in order_rows
    ):
        raise GTokTrainingV2Error(
            "physical V4 evidence does not bind the governed data-order seed"
        )
    training = next(
        (row for row in streams if isinstance(row, Mapping) and row.get("stream") == "T"),
        None,
    )
    if training is None:
        raise GTokTrainingV2Error("physical V4 evidence has no T identity")
    heldout = tuple(
        (stratum, int(next(
            row["retained_text_bytes"]
            for row in groups
            if isinstance(row, Mapping) and row.get("stream") == "H" and row.get("stratum") == stratum
        )))
        for stratum in GTOK_STRATA
    )
    return V4CorpusSourceV2(
        root=resolved,
        physical_d6_evidence_sha256=physical_sha,
        training_order_receipts=order_rows,
        training_raw_bytes=int(training["retained_text_bytes"]),
        heldout_raw_bytes_by_stratum=heldout,
    )


__all__ = [
    "AnalyticUnsupportedFlopRowV2",
    "ArmMeasurementPanelV2",
    "CalibrationMeasurementV2",
    "CompleteFlopLedgerV2",
    "FLOP_BINDING_SHA256_V2",
    "FullRunMeasurementV2",
    "GTokRunWatchdogV2",
    "GTokCampaignTripwireV2",
    "GTokTrainingV2Error",
    "INITIALIZATION_RECIPE_SHA256_V2",
    "MEASUREMENT_BINDING_SHA256_V2",
    "OutputSurfacePerformanceV2",
    "PACKING_BINDING_SHA256_V2",
    "PackedBatchV2",
    "PhysicalShapeFlopReceiptV2",
    "ProfilerOperatorFlopRowV2",
    "SCHEDULE_BINDING_SHA256_V2",
    "TrainingDocumentV2",
    "TrainingPlanV2",
    "StratumCompressionMetricsV2",
    "TokenizerCorpusMetricsV2",
    "V4CorpusSourceV2",
    "build_flat_a1_adamw_v2",
    "build_gtok_proxy_model_v2",
    "calibrate_arm_v2",
    "evaluate_heldout_v2",
    "execute_full_run_v2",
    "gtok_proxy_config_v2",
    "iter_packed_global_batches_v2",
    "learning_rate_for_step_v2",
    "load_v4_corpus_source_v2",
    "measure_output_surface_performance_v2",
    "measure_tokenizer_corpus_metrics_v2",
    "measure_vocabulary_fractions_v2",
    "plan_training_stream_v2",
    "require_production_a100_v2",
    "shared_nonvocabulary_state_sha256_v2",
    "validate_calibration_prefix_v2",
]
