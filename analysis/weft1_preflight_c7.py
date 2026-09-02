"""Fail-closed PRE-FLIGHT C7 receipt-schema-presence inventory.

C7 asks the toy run to emit every receipt line accumulated by the semantics
chain.  This module proves only which bound schema fields exist; it does not
construct the required toy run or promote field presence to semantic emission.
PF-1 explicitly struck the only ``eta * lambda`` producer, while catch #26
blocks a production loop certificate.  The inventory makes those boundaries
machine-readable and refuses to call C7 complete.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path

from models.ablation_lm.accounting import CompositionReceipt
from models.ablation_lm.certificates import LoopLipschitzReceipt
from training.weft1_gtok_confirmation_v2 import ConfirmationBudgetReceiptV2
from training.weft1_gtok_training_v2 import (
    ConfirmationTrainingPlanV2,
    TrainingPlanV2,
)
from training.weft1_gtok_v2_contract import (
    ArmTerminalStatisticsV2,
    GTokRunReceiptV2,
)


PREFLIGHT_PROGRAM_BYTES = 15_575
PREFLIGHT_PROGRAM_SHA256 = (
    "ceaa5338830307d3783296b8a4aef7bb87962eb35535d392f4c6d217dff88a5b"
)
PREFLIGHT_RATIFICATION_BYTES = 2_233
PREFLIGHT_RATIFICATION_SHA256 = (
    "4a13054d38c68e5e9476330528649d445ff845e639e0a36bb01641b54ef66965"
)
PF1_AUTHORITY_BYTES = 12_285
PF1_AUTHORITY_SHA256 = (
    "4e3186c432b57f71b9f32a444a269eec08557ca5181a6896b477078dbbb40861"
)
GTOK_SEMANTICS_AUTHORITIES = (
    (
        "STRATEGY_GTOK_CONFIRMATION_SEMANTICS_20260831.md",
        13_975,
        "2e42664d0062a119c9fadcb76bf227a91134914920116627f9244f650defe72d",
    ),
    (
        "STRATEGY_GTOK_SEMANTICS_AMENDMENT_S1_20260831.md",
        12_411,
        "c37c4be064fe447e01182acc11b1713239c761ddd50583a8299972b4b340bd2a",
    ),
    (
        "STRATEGY_GTOK_SEMANTICS_AMENDMENT_S2_20260831.md",
        6_638,
        "5420a4e57c080d09f5f924acc859a5579edd1ca1939c8bbdaf727e5afd55ac5e",
    ),
)

CONSUMPTION_FIELDS = (
    "stream_bytes",
    "stream_tokens",
    "stream_docs",
    "trained_bytes",
    "trained_tokens",
    "trained_docs_full",
    "dropped_bytes",
    "dropped_tokens",
    "dropped_docs",
)
BOUNDARY_FIELDS = ("boundary_doc_id", "boundary_doc_consumed_tokens")


class C7SchemaIncomplete(RuntimeError):
    """Raised when a caller attempts to promote the dry run to a pass."""


@dataclass(frozen=True)
class C7SchemaLine:
    name: str
    status: str
    source_type: str | None
    source_fields: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class C7SchemaAudit:
    program_authority_sha256: str
    ratification_authority_sha256: str
    pf1_authority_sha256: str
    gtok_semantics_authority_sha256s: tuple[str, ...]
    authority_byte_verified: bool
    lines: tuple[C7SchemaLine, ...]
    complete: bool
    disposition: str
    a100_hours: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def require_complete(self) -> None:
        if self.complete:
            return
        blocked = tuple(
            (line.name, line.status)
            for line in self.lines
            if line.status != "emitted_and_verified"
        )
        raise C7SchemaIncomplete(f"C7 schema remains fail-closed: {blocked}")


def _field_names(receipt_type: type[object]) -> tuple[str, ...]:
    return tuple(field.name for field in fields(receipt_type))


def _require_fields(
    receipt_type: type[object],
    required: tuple[str, ...],
) -> tuple[str, ...]:
    available = _field_names(receipt_type)
    missing = tuple(name for name in required if name not in available)
    if missing:
        raise C7SchemaIncomplete(
            f"{receipt_type.__name__} is missing required C7 fields {missing}"
        )
    return required


def _verify_authority_bytes() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = (
        (
            root / "docs" / "STRATEGY_PREFLIGHT_PROGRAM_20260902.md",
            PREFLIGHT_PROGRAM_BYTES,
            PREFLIGHT_PROGRAM_SHA256,
        ),
        (
            root / "docs" / "STRATEGY_PREFLIGHT_RATIFICATION_20260902.md",
            PREFLIGHT_RATIFICATION_BYTES,
            PREFLIGHT_RATIFICATION_SHA256,
        ),
        (
            root / "docs" / "STRATEGY_PREFLIGHT_AMENDMENT_PF1_20260902.md",
            PF1_AUTHORITY_BYTES,
            PF1_AUTHORITY_SHA256,
        ),
        *tuple(
            (root / "docs" / name, expected_bytes, expected_sha256)
            for name, expected_bytes, expected_sha256 in GTOK_SEMANTICS_AUTHORITIES
        ),
    )
    for path, expected_bytes, expected_sha256 in expected:
        payload = path.read_bytes()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if len(payload) != expected_bytes or actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"C7 authority drift at {path.name}: bytes={len(payload)}, "
                f"sha256={actual_sha256}"
            )


def audit_preflight_c7_schema() -> C7SchemaAudit:
    """Validate authority and field presence without inventing emitted values."""

    _verify_authority_bytes()

    lines = (
        C7SchemaLine(
            name="rho_values",
            status="schema_present_emitter_unverified",
            source_type=ArmTerminalStatisticsV2.__name__,
            source_fields=_require_fields(
                ArmTerminalStatisticsV2,
                ("vocab_size", "seeds", "seed_bpbs", "rho_bpb_micros"),
            ),
            reason=(
                "the field exists, but no toy emitter produced and validated four "
                "arm rows"
            ),
        ),
        C7SchemaLine(
            name="consumption_fields",
            status="schema_present_emitter_unverified",
            source_type=GTokRunReceiptV2.__name__,
            source_fields=_require_fields(
                GTokRunReceiptV2,
                CONSUMPTION_FIELDS + BOUNDARY_FIELDS,
            ),
            reason=(
                "nine stream/trained/dropped byte-token-document fields plus the "
                "single optional boundary-document identity and consumed-token count; "
                "presence does not prove populated accounting"
            ),
        ),
        C7SchemaLine(
            name="integer_f_star",
            status="schema_present_emitter_unverified",
            source_type=ConfirmationBudgetReceiptV2.__name__,
            source_fields=_require_fields(
                ConfirmationBudgetReceiptV2,
                ("pair", "target_flops", "rows"),
            ),
            reason=(
                "target_flops can hold the exact integer budget, but no toy emitter "
                "produced and validated it"
            ),
        ),
        C7SchemaLine(
            name="checkpoint_step_indices",
            status="schema_present_emitter_unverified",
            source_type=(
                f"{TrainingPlanV2.__name__}/{ConfirmationTrainingPlanV2.__name__}"
            ),
            source_fields=(
                *_require_fields(TrainingPlanV2, ("bpb_checkpoint_steps",)),
                *_require_fields(
                    ConfirmationTrainingPlanV2,
                    ("bpb_checkpoint_steps",),
                ),
            ),
            reason=(
                "the plan fields exist, but no toy emitter proved populated pre-launch "
                "byte checkpoints"
            ),
        ),
        C7SchemaLine(
            name="gate_rate_by_k",
            status="schema_only_not_materialized",
            source_type=CompositionReceipt.__name__,
            source_fields=_require_fields(
                CompositionReceipt,
                (
                    "requested_visits",
                    "executed_visits",
                    "sidecar_firing_fraction_by_step",
                ),
            ),
            reason=(
                "field exists, but the WEFT-1 sidecar is absent and C-S6-1/2 remain "
                "unbound; no realized rate may be emitted"
            ),
        ),
        C7SchemaLine(
            name="realized_eta_lambda",
            status="conflict_pf1_6_pending_c7_ruling",
            source_type=None,
            source_fields=(),
            reason=(
                "PF-1.6 strikes A8 and moves its certificate tests to MEM-SYN-FW, "
                "but does not literally dispose C7's independently listed realized "
                "eta-lambda receipt line; coding will not infer the resolution"
            ),
        ),
        C7SchemaLine(
            name="loop_lipschitz",
            status="blocked_catch_26",
            source_type=LoopLipschitzReceipt.__name__,
            source_fields=_require_fields(
                LoopLipschitzReceipt,
                (
                    "lambda_adapters",
                    "lambda_hat_core",
                    "alarm_threshold",
                    "alarm_fired",
                    "production_claim",
                ),
            ),
            reason=(
                "standalone schema exists, but nonlinear scratch and the open C-JAC-1 "
                "joint metric prevent a production certificate or alarm"
            ),
        ),
    )
    complete = all(line.status == "emitted_and_verified" for line in lines)
    return C7SchemaAudit(
        program_authority_sha256=PREFLIGHT_PROGRAM_SHA256,
        ratification_authority_sha256=PREFLIGHT_RATIFICATION_SHA256,
        pf1_authority_sha256=PF1_AUTHORITY_SHA256,
        gtok_semantics_authority_sha256s=tuple(
            authority[2] for authority in GTOK_SEMANTICS_AUTHORITIES
        ),
        authority_byte_verified=True,
        lines=lines,
        complete=complete,
        disposition=(
            "schema_complete"
            if complete
            else "return_to_strategy_without_inventing_missing_receipt_lines"
        ),
    )


def main() -> None:
    print(json.dumps(audit_preflight_c7_schema().to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "BOUNDARY_FIELDS",
    "C7SchemaAudit",
    "C7SchemaIncomplete",
    "C7SchemaLine",
    "CONSUMPTION_FIELDS",
    "GTOK_SEMANTICS_AUTHORITIES",
    "PF1_AUTHORITY_BYTES",
    "PF1_AUTHORITY_SHA256",
    "PREFLIGHT_PROGRAM_BYTES",
    "PREFLIGHT_PROGRAM_SHA256",
    "PREFLIGHT_RATIFICATION_BYTES",
    "PREFLIGHT_RATIFICATION_SHA256",
    "audit_preflight_c7_schema",
]
