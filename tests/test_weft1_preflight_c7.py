from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import math

import pytest

from analysis.weft1_preflight_c7 import (
    BOUNDARY_FIELDS,
    CONSUMPTION_FIELDS,
    FULL_GLOBAL_BATCH_TOKENS,
    C7BoundaryTokenRecord,
    C7SchemaIncomplete,
    C7_STAGE1_TOY_SHA256,
    GTOK_SEMANTICS_AUTHORITIES,
    PF2_AUTHORITY_SHA256,
    STAGE1_FAMILIES,
    audit_preflight_c7_schema,
    emit_c7_stage1_toy,
)
from training.weft1_gtok_confirmation_v2 import GTokConfirmationV2Error
from training.weft1_gtok_contract import GTOK_VOCABULARY_ARMS
from training.weft1_gtok_training_v2 import TrainingDocumentV2


def _independent_half_even_micros(value: float) -> int:
    with localcontext() as context:
        context.prec = 50
        rounded = Decimal.from_float(value).quantize(
            Decimal("0.000001"),
            rounding=ROUND_HALF_EVEN,
        )
    return int(rounded * Decimal(1_000_000))


def test_c7_pf2_authority_and_stage1_emission_are_pinned() -> None:
    receipt = audit_preflight_c7_schema()
    repeated = audit_preflight_c7_schema()
    by_name = {line.name: line for line in receipt.lines}

    assert receipt.authority_byte_verified
    assert receipt.pf2_authority_sha256 == PF2_AUTHORITY_SHA256
    assert receipt.gtok_semantics_authority_sha256s == tuple(
        authority[2] for authority in GTOK_SEMANTICS_AUTHORITIES
    )
    assert receipt.stage == "stage_1_emitted_and_verified_stage_2_pending"
    assert receipt.stage1_emission.validated_families == STAGE1_FAMILIES
    assert receipt.stage1_emission.synthetic_only is True
    assert receipt.stage1_emission_sha256 == receipt.stage1_emission.receipt_sha256
    assert receipt.stage1_emission_sha256 == C7_STAGE1_TOY_SHA256
    assert repeated.stage1_emission_sha256 == receipt.stage1_emission_sha256
    assert len(receipt.stage1_emission_sha256) == 64
    assert all(by_name[name].status == "emitted_and_verified" for name in STAGE1_FAMILIES)
    emission = receipt.stage1_emission
    assert emission.source_selection.matrix_receipt_sha256 == (
        emission.source_matrix.receipt_sha256
    )
    assert emission.confirmation_budget.matrix_receipt_sha256 == (
        emission.source_matrix.receipt_sha256
    )
    assert emission.confirmation_budget.selection_receipt_sha256 == (
        emission.source_selection.receipt_sha256
    )
    assert emission.consumption.source_run.receipt_sha256 == (
        emission.checkpoints.base_run.receipt_sha256
    )
    assert emission.checkpoints.base_plan.bpb_checkpoint_steps == (
        emission.checkpoints.base.checkpoint_steps
    )
    assert emission.checkpoints.confirmation_plan.bpb_checkpoint_steps == (
        emission.checkpoints.confirmation.checkpoint_steps
    )
    assert emission.checkpoints.base.n == 400
    assert emission.checkpoints.confirmation.n == 399
    assert emission.checkpoints.confirmation_budget_row.vocab_size == 32_768
    assert emission.checkpoints.confirmation_budget_row in (
        emission.confirmation_budget.rows
    )
    assert emission.checkpoints.confirmation_budget_row.planned_optimizer_steps == (
        emission.checkpoints.confirmation_plan.optimizer_steps
    )


def test_c7_rho_rows_use_ordered_binary64_half_even_six_decimal_values() -> None:
    rows = emit_c7_stage1_toy().rho_rows

    assert tuple(row.vocab_size for row in rows) == GTOK_VOCABULARY_ARMS
    assert tuple(row.seeds for row in rows) == ((101, 202),) * 4
    assert tuple(row.rho_bpb_micros for row in rows) == (
        1_400_000,
        1_300_000,
        1_000_001,
        1_000_001,
    )
    for row in rows:
        mean = math.fsum(row.seed_bpbs) / 2.0
        assert type(mean) is float
        assert row.rho_bpb_micros == _independent_half_even_micros(mean)

    # These two binary64 values straddle opposite decimal half-even decisions.
    decimal_string_results = tuple(
        int(
            Decimal(str(math.fsum(row.seed_bpbs) / 2.0)).quantize(
                Decimal("0.000001"),
                rounding=ROUND_HALF_EVEN,
            )
            * Decimal(1_000_000)
        )
        for row in rows[2:]
    )
    assert decimal_string_results == (1_000_002, 1_000_000)
    assert decimal_string_results != tuple(row.rho_bpb_micros for row in rows[2:])

    with pytest.raises(ValueError, match="rounded float64"):
        replace(rows[0], rho_bpb_micros=rows[0].rho_bpb_micros + 1)
    with pytest.raises(ValueError, match="registered vocabulary arms"):
        replace(emit_c7_stage1_toy(), rho_rows=tuple(reversed(rows)))
    swapped_first = replace(rows[0], seed_bpbs=tuple(reversed(rows[0].seed_bpbs)))
    with pytest.raises(ValueError, match="production selection join"):
        replace(emit_c7_stage1_toy(), rho_rows=(swapped_first, *rows[1:]))


def test_c7_consumption_closes_and_boundary_is_a_strict_partial_document() -> None:
    consumption = emit_c7_stage1_toy().consumption

    assert consumption.stream_tokens == (
        consumption.trained_tokens + consumption.dropped_tokens
    )
    assert consumption.stream_bytes == consumption.trained_bytes + consumption.dropped_bytes
    assert consumption.trained_tokens == (
        consumption.optimizer_steps * FULL_GLOBAL_BATCH_TOKENS
    )
    assert consumption.boundary_doc_id is not None
    assert consumption.boundary_doc_consumed_tokens is not None
    assert consumption.boundary_doc_token_length is not None
    assert consumption.boundary_source is not None
    assert consumption.boundary_doc_token_length == len(
        consumption.boundary_source.token_ids
    )
    assert consumption.boundary_doc_id == (
        consumption.boundary_source.document.raw_content_id
    )
    assert (
        consumption.boundary_doc_consumed_tokens
        < consumption.boundary_doc_token_length
    )
    assert consumption.stream_docs == (
        consumption.trained_docs_full + consumption.dropped_docs + 1
    )

    no_boundary_run = replace(
        consumption.source_run,
        stream_docs=consumption.stream_docs - 1,
        boundary_doc_id=None,
        boundary_doc_consumed_tokens=None,
    )
    no_boundary = replace(
        consumption,
        source_run=no_boundary_run,
        stream_docs=consumption.stream_docs - 1,
        boundary_doc_id=None,
        boundary_doc_consumed_tokens=None,
        boundary_source=None,
    )
    assert no_boundary.boundary_doc_id is None
    with pytest.raises(ValueError, match="production source run"):
        replace(consumption, stream_tokens=consumption.stream_tokens + 1)
    with pytest.raises(ValueError, match="production source run"):
        replace(consumption, trained_tokens=consumption.trained_tokens - 1)
    with pytest.raises(ValueError, match="travel together"):
        replace(consumption, boundary_source=None)
    with pytest.raises(ValueError, match="init=False"):
        replace(consumption, boundary_doc_token_length=999)
    with pytest.raises(ValueError, match="toy byte tokenizer"):
        replace(
            consumption.boundary_source,
            token_ids=(*consumption.boundary_source.token_ids, 4),
        )
    short_text = "12345678"
    short_document = TrainingDocumentV2(
        raw_content_id=hashlib.sha1(  # noqa: S324 - governed raw-content ID
            short_text.encode("utf-8")
        ).hexdigest(),
        text=short_text,
        stratum="general",
    )
    short_source = C7BoundaryTokenRecord(
        document=short_document,
        token_ids=(
            1,
            *(4 + value for value in short_document.raw_bytes),
            2,
            3,
        ),
    )
    short_run = replace(
        consumption.source_run,
        boundary_doc_id=short_document.raw_content_id,
    )
    with pytest.raises(ValueError, match="strictly below"):
        replace(
            consumption,
            source_run=short_run,
            boundary_doc_id=short_document.raw_content_id,
            boundary_source=short_source,
        )
    with pytest.raises(ValueError, match="partial global-batch suffix"):
        replace(
            consumption.source_run,
            stream_tokens=(
                consumption.trained_tokens + FULL_GLOBAL_BATCH_TOKENS
            ),
            dropped_tokens=FULL_GLOBAL_BATCH_TOKENS,
        )


def test_c7_consumption_schema_includes_all_nine_fields_and_boundary_length() -> None:
    receipt = audit_preflight_c7_schema()
    line = next(line for line in receipt.lines if line.name == "consumption_fields")

    assert line.source_fields == (
        CONSUMPTION_FIELDS
        + BOUNDARY_FIELDS
        + (
            "source_run",
            "boundary_source",
            "boundary_doc_token_length",
            "optimizer_steps",
        )
    )


def test_c7_f_star_is_exact_integer_min_of_pairwise_floor_means() -> None:
    budget = emit_c7_stage1_toy().confirmation_budget
    means = tuple((row.base_flops[0] + row.base_flops[1]) // 2 for row in budget.rows)

    assert budget.pair == (32_768, 49_152)
    assert all(value > 2**53 for row in budget.rows for value in row.base_flops)
    assert tuple(row.arm_mean_flops for row in budget.rows) == means
    assert budget.target_flops == min(means)
    assert type(budget.target_flops) is int
    assert tuple(row.planned_optimizer_steps for row in budget.rows) == (399, 400)

    with pytest.raises(ValueError, match="min floor arm mean"):
        replace(budget, target_flops=budget.target_flops + 1)
    emission = emit_c7_stage1_toy()
    mismatched_evidence = replace(
        emission.base_flop_evidence[0],
        base_run_receipt_sha256="a" * 64,
    )
    with pytest.raises(GTokConfirmationV2Error, match="differs from base matrix"):
        replace(
            emission,
            base_flop_evidence=(
                mismatched_evidence,
                *emission.base_flop_evidence[1:],
            ),
        )


def test_c7_checkpoint_indices_are_exact_first_crossings_and_third_is_n() -> None:
    emission = emit_c7_stage1_toy()
    checkpoints = emission.checkpoints
    base = checkpoints.base
    confirmation = checkpoints.confirmation

    assert base.checkpoint_steps == (100, 200, 400)
    assert confirmation.checkpoint_steps == (100, 200, 399)
    assert checkpoints.base_plan.bpb_checkpoint_steps == base.checkpoint_steps
    assert checkpoints.confirmation_plan.bpb_checkpoint_steps == (
        confirmation.checkpoint_steps
    )
    assert checkpoints.confirmation_budget_row.planned_optimizer_steps == (
        checkpoints.confirmation_plan.optimizer_steps
    )
    assert checkpoints.confirmation_plan.optimizer_steps == confirmation.n == 399
    assert confirmation.total_bytes < base.total_bytes
    for series in (base, confirmation):
        assert series.checkpoint_steps[2] == series.n
        assert series.checkpoint_steps == tuple(
            sorted(set(series.checkpoint_steps))
        )
        for step, denominator in zip(series.checkpoint_steps, (4, 2, 1)):
            assert denominator * series.cumulative_consumed_bytes[step - 1] >= (
                series.total_bytes
            )
            previous = (
                0
                if step == 1
                else series.cumulative_consumed_bytes[step - 2]
            )
            assert denominator * previous < series.total_bytes

    with pytest.raises(ValueError, match="governed first crossings"):
        replace(base, checkpoint_steps=(99, 200, 400))
    with pytest.raises(ValueError, match="third checkpoint index"):
        replace(base, checkpoint_steps=(100, 200, 399))
    with pytest.raises(ValueError, match="distinct and end at the horizon"):
        replace(
            checkpoints.confirmation_plan,
            bpb_checkpoint_steps=(100, 100, 399),
        )
    shifted_confirmation_plan = replace(
        checkpoints.confirmation_plan,
        bpb_checkpoint_steps=(99, 200, 399),
    )
    with pytest.raises(ValueError, match="confirmation indices"):
        replace(
            checkpoints,
            confirmation_plan=shifted_confirmation_plan,
        )
    shifted_base_bytes = list(base.cumulative_consumed_bytes)
    shifted_base_bytes[base.checkpoint_steps[0] - 1] += 1
    shifted_base = replace(
        base,
        cumulative_consumed_bytes=tuple(shifted_base_bytes),
    )
    with pytest.raises(ValueError, match="production source run"):
        replace(
            checkpoints,
            base=shifted_base,
        )
    shifted_confirmation_bytes = list(confirmation.cumulative_consumed_bytes)
    shifted_confirmation_bytes[-1] += 1
    shifted_confirmation = replace(
        confirmation,
        cumulative_consumed_bytes=tuple(shifted_confirmation_bytes),
    )
    with pytest.raises(ValueError, match="confirmation bytes"):
        replace(checkpoints, confirmation=shifted_confirmation)

    other_budget_row = next(
        row
        for row in emission.confirmation_budget.rows
        if row != checkpoints.confirmation_budget_row
    )
    with pytest.raises(ValueError, match="budget row n differs"):
        replace(checkpoints, confirmation_budget_row=other_budget_row)
    with pytest.raises(ValueError, match="budget row n differs"):
        replace(checkpoints, confirmation=base)


def test_c7_eta_is_struck_and_stage2_lines_keep_complete_gate_incomplete() -> None:
    receipt = audit_preflight_c7_schema()
    by_name = {line.name: line for line in receipt.lines}

    assert by_name["gate_rate_by_k"].status == "stage_2_pending_sidecar"
    assert by_name["realized_eta_lambda"].status == "struck_by_pf2_4"
    assert by_name["lambda_adapters"].status == (
        "stage_2_pending_catch_26_c_jac_1"
    )
    assert by_name["lambda_adapters"].source_fields == ("lambda_adapters",)
    assert by_name["lambda_hat_core"].status == (
        "stage_2_pending_catch_26_c_jac_1"
    )
    assert by_name["lambda_hat_core"].source_fields == ("lambda_hat_core",)
    assert "loop_lipschitz" not in by_name
    assert receipt.complete is False
    assert receipt.a100_hours == 0.0
    assert receipt.disposition == (
        "complete_c7_gate_preserved_incomplete_stage_2_pending"
    )

    with pytest.raises(C7SchemaIncomplete, match="fail-closed") as error:
        receipt.require_complete()
    message = str(error.value)
    assert "gate_rate_by_k" in message
    assert "lambda_adapters" in message
    assert "lambda_hat_core" in message
    assert "realized_eta_lambda" not in message
    with pytest.raises(ValueError, match="complete flag"):
        replace(receipt, complete=True)
    object.__setattr__(receipt, "complete", True)
    with pytest.raises(C7SchemaIncomplete, match="fail-closed"):
        receipt.require_complete()
