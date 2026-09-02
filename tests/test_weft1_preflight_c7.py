from __future__ import annotations

import pytest

from analysis.weft1_preflight_c7 import (
    BOUNDARY_FIELDS,
    CONSUMPTION_FIELDS,
    C7SchemaIncomplete,
    GTOK_SEMANTICS_AUTHORITIES,
    audit_preflight_c7_schema,
)


def test_c7_bound_schema_lines_are_present_but_not_promoted_to_emission() -> None:
    receipt = audit_preflight_c7_schema()
    by_name = {line.name: line for line in receipt.lines}

    assert receipt.authority_byte_verified
    assert receipt.gtok_semantics_authority_sha256s == tuple(
        authority[2] for authority in GTOK_SEMANTICS_AUTHORITIES
    )
    assert by_name["rho_values"].status == "schema_present_emitter_unverified"
    assert by_name["consumption_fields"].status == "schema_present_emitter_unverified"
    assert by_name["integer_f_star"].status == "schema_present_emitter_unverified"
    assert (
        by_name["checkpoint_step_indices"].status
        == "schema_present_emitter_unverified"
    )
    assert by_name["consumption_fields"].source_fields == (
        CONSUMPTION_FIELDS + BOUNDARY_FIELDS
    )
    assert by_name["integer_f_star"].source_fields == (
        "pair",
        "target_flops",
        "rows",
    )
    assert by_name["checkpoint_step_indices"].source_fields == (
        "bpb_checkpoint_steps",
        "bpb_checkpoint_steps",
    )


def test_c7_never_promotes_struck_or_unbound_lines_to_a_pass() -> None:
    receipt = audit_preflight_c7_schema()
    by_name = {line.name: line for line in receipt.lines}

    assert by_name["gate_rate_by_k"].status == "schema_only_not_materialized"
    assert (
        by_name["realized_eta_lambda"].status
        == "conflict_pf1_6_pending_c7_ruling"
    )
    assert by_name["loop_lipschitz"].status == "blocked_catch_26"
    assert receipt.complete is False
    assert receipt.a100_hours == 0.0
    assert (
        receipt.disposition
        == "return_to_strategy_without_inventing_missing_receipt_lines"
    )
    with pytest.raises(C7SchemaIncomplete, match="fail-closed"):
        receipt.require_complete()
