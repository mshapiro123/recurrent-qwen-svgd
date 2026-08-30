"""Fail-closed P-B to G-TOK adapter for the ratified A2 first-fit screen.

This module never selects documents and never reconstructs A1 prefix-floor
fields.  It re-runs the P-B freeze verifier against the materialization root,
requires the stored freeze to equal that fresh result byte-for-byte at the
canonical object level, and then projects only validated V4 raw-content-ID
evidence into :class:`FrozenScreenCorpusV2`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from training.weft1_corpus_materialize_a3 import D6_PHYSICAL_EVIDENCE_SCHEMA_V4
from training.weft1_corpus_pb import (
    PBFreezeError,
    build_freeze_receipt,
    inspect_pa_v4,
)
from training.weft1_gtok_contract import (
    GTOK_SCREEN_HELDOUT_STRATUM_TARGETS,
    GTOK_SCREEN_TRAIN_STRATUM_TARGETS,
    GTOK_STRATA,
    canonical_json_bytes,
)
from training.weft1_gtok_v2_contract import (
    A2FirstFitGroupReceiptV2,
    A2FirstFitScreenReceiptV2,
    FrozenScreenCorpusV2,
    GTokV2Stop,
)
from training.weft1_strict_io import (
    StrictJsonError,
    assert_no_symlink_ancestors,
    load_canonical_json_snapshot,
)


class GTokPBAdapterError(GTokV2Stop):
    """The stored P-B freeze or its physical corpus evidence did not revalidate."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_canonical_mapping(path: Path, name: str) -> tuple[bytes, Mapping[str, Any]]:
    try:
        assert_no_symlink_ancestors(path)
        raw, value = load_canonical_json_snapshot(path)
    except (OSError, StrictJsonError, TypeError, ValueError) as error:
        raise GTokPBAdapterError(f"{name} is absent or invalid") from error
    if not isinstance(value, Mapping) or raw != canonical_json_bytes(value) + b"\n":
        raise GTokPBAdapterError(f"{name} must be canonical newline-terminated JSON")
    return raw, value


def _require_exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise GTokPBAdapterError(f"{name} must be an exact integer >= {minimum}")
    return value


def _project_first_fit(
    *,
    physical_d6: Mapping[str, Any],
    diagnostic_d6: Mapping[str, Any],
) -> A2FirstFitScreenReceiptV2:
    split_rows = diagnostic_d6.get("split_rows")
    if not isinstance(split_rows, list) or tuple(
        row.get("stratum") for row in split_rows if isinstance(row, Mapping)
    ) != GTOK_STRATA:
        raise GTokPBAdapterError("diagnostic D6 split accounting is noncanonical")
    split_accounting: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in split_rows:
        if not isinstance(row, Mapping):
            raise GTokPBAdapterError("diagnostic D6 split row is not an object")
        stratum = str(row["stratum"])
        for stream, field in (("T", "training"), ("H", "heldout")):
            accounting = row.get(field)
            if not isinstance(accounting, Mapping):
                raise GTokPBAdapterError("diagnostic D6 split accounting is absent")
            split_accounting[(stream, stratum)] = accounting

    physical_groups = physical_d6.get("split_groups")
    expected_keys = tuple(
        (stream, stratum) for stream in ("T", "H") for stratum in GTOK_STRATA
    )
    if not isinstance(physical_groups, list) or tuple(
        (row.get("stream"), row.get("stratum"))
        for row in physical_groups
        if isinstance(row, Mapping)
    ) != expected_keys:
        raise GTokPBAdapterError("physical D6 does not contain the canonical eight groups")

    groups: list[A2FirstFitGroupReceiptV2] = []
    for row in physical_groups:
        if not isinstance(row, Mapping):  # guarded above; keeps type narrowing explicit
            raise GTokPBAdapterError("physical D6 group is not an object")
        stream = str(row.get("stream"))
        stratum = str(row.get("stratum"))
        targets = dict(
            GTOK_SCREEN_TRAIN_STRATUM_TARGETS
            if stream == "T"
            else GTOK_SCREEN_HELDOUT_STRATUM_TARGETS
        )
        accounting = split_accounting[(stream, stratum)]
        target = _require_exact_int(
            accounting.get("target_bytes"),
            f"{stream}/{stratum} target bytes",
            minimum=1,
        )
        realized = _require_exact_int(
            accounting.get("realized_bytes"),
            f"{stream}/{stratum} realized bytes",
            minimum=1,
        )
        deficit = _require_exact_int(
            accounting.get("deficit_bytes"),
            f"{stream}/{stratum} deficit bytes",
        )
        document_count = _require_exact_int(
            accounting.get("document_count"),
            f"{stream}/{stratum} document count",
            minimum=1,
        )
        if (
            target != targets[stratum]
            or realized != row.get("retained_text_bytes")
            or document_count != row.get("document_count")
        ):
            raise GTokPBAdapterError(
                "diagnostic D6 first-fit accounting differs from physical D6"
            )
        ordered_raw_ids = row.get("ordered_raw_content_ids_sha256")
        if not isinstance(ordered_raw_ids, str):
            raise GTokPBAdapterError("physical D6 group omits raw-content-ID order")
        groups.append(
            A2FirstFitGroupReceiptV2(
                stream=stream,
                stratum=stratum,
                target_bytes=target,
                realized_bytes=realized,
                deficit_bytes=deficit,
                document_count=document_count,
                ordered_raw_content_ids_sha256=ordered_raw_ids,
            )
        )

    streams = physical_d6.get("stream_identities")
    if not isinstance(streams, list) or tuple(
        row.get("stream") for row in streams if isinstance(row, Mapping)
    ) != ("T", "H"):
        raise GTokPBAdapterError("physical D6 T/H stream identities are noncanonical")
    by_stream = {str(row["stream"]): row for row in streams}
    for stream in ("T", "H"):
        row = by_stream[stream]
        stream_groups = tuple(group for group in groups if group.stream == stream)
        if (
            _require_exact_int(
                row.get("document_count"), f"{stream} document count", minimum=1
            )
            != sum(group.document_count for group in stream_groups)
            or _require_exact_int(
                row.get("retained_text_bytes"), f"{stream} retained bytes", minimum=1
            )
            != sum(group.realized_bytes for group in stream_groups)
        ):
            raise GTokPBAdapterError("physical D6 stream totals differ from its groups")

    document_overlap = _require_exact_int(
        diagnostic_d6.get("document_overlap_count"), "D6 document overlap"
    )
    cluster_overlap = _require_exact_int(
        diagnostic_d6.get("cluster_overlap_count"), "D6 cluster overlap"
    )
    if document_overlap != 0 or cluster_overlap != 0:
        raise GTokPBAdapterError("D6 overlap evidence is not exact zero")
    if physical_d6.get("document_overlap_count") != document_overlap:
        raise GTokPBAdapterError("physical and diagnostic D6 document overlap differ")

    training_identity = by_stream["T"].get("framed_retained_text_sha256")
    heldout_identity = by_stream["H"].get("framed_retained_text_sha256")
    if not isinstance(training_identity, str) or not isinstance(heldout_identity, str):
        raise GTokPBAdapterError("physical D6 omits a framed stream identity")
    return A2FirstFitScreenReceiptV2(
        groups=tuple(groups),
        training_framed_stream_sha256=training_identity,
        heldout_framed_stream_sha256=heldout_identity,
        document_overlap_count=document_overlap,
        cluster_overlap_count=cluster_overlap,
    )


def load_frozen_screen_corpus_v2(
    *,
    materialization_root: Path,
    freeze_receipt_path: Path,
    gate_bundle_path: Path,
    c2_evidence_path: Path,
    decon_receipt_path: Path,
) -> FrozenScreenCorpusV2:
    """Revalidate P-A/P-B and construct the only production corpus contract.

    The expensive P-B verifier is intentionally called before any G-TOK object
    is returned.  A stored freeze is an input claim, not authority by itself.
    """

    for name, value in (
        ("materialization_root", materialization_root),
        ("freeze_receipt_path", freeze_receipt_path),
        ("gate_bundle_path", gate_bundle_path),
        ("c2_evidence_path", c2_evidence_path),
        ("decon_receipt_path", decon_receipt_path),
    ):
        if not isinstance(value, Path):
            raise TypeError(f"{name} must be pathlib.Path")

    try:
        pa = inspect_pa_v4(materialization_root)
        stored_raw, stored = _load_canonical_mapping(
            freeze_receipt_path, "stored P-B freeze receipt"
        )
        recomputed = build_freeze_receipt(
            materialization_root=materialization_root,
            gate_bundle_path=gate_bundle_path,
            c2_evidence_path=c2_evidence_path,
            decon_receipt_path=decon_receipt_path,
        )
    except GTokPBAdapterError:
        raise
    except (OSError, PBFreezeError, TypeError, ValueError) as error:
        raise GTokPBAdapterError("P-B freeze revalidation failed") from error

    if canonical_json_bytes(stored) != canonical_json_bytes(recomputed):
        raise GTokPBAdapterError("stored P-B freeze differs from fresh revalidation")

    expected_joins = {
        "full_shard_manifest_sha256": pa.full_shard_manifest_physical_sha256,
        "screen_submanifest_sha256": pa.screen_submanifest_physical_sha256,
        "d6_physical_evidence_sha256": pa.d6_physical_evidence_physical_sha256,
    }
    if any(recomputed.get(name) != expected for name, expected in expected_joins.items()):
        raise GTokPBAdapterError("fresh freeze substitutes a different P-A artifact")

    d6_path = pa.root.joinpath(
        *PurePosixPath(pa.d6_physical_evidence_relative_path).parts
    )
    d6_raw, physical_d6 = _load_canonical_mapping(d6_path, "physical D6 evidence")
    if (
        _sha256_bytes(d6_raw) != pa.d6_physical_evidence_physical_sha256
        or physical_d6.get("schema") != D6_PHYSICAL_EVIDENCE_SCHEMA_V4
        or physical_d6.get("evidence_identity_sha256")
        != pa.d6_physical_evidence_identity_sha256
    ):
        raise GTokPBAdapterError("physical D6 identity differs from P-A/P-B")

    diagnostic_raw, diagnostic_d6 = _load_canonical_mapping(
        pa.root / "diagnostics" / "d6.json", "D6 diagnostic"
    )
    diagnostic_hashes = dict(pa.diagnostic_sha256s)
    if _sha256_bytes(diagnostic_raw) != diagnostic_hashes.get("D6"):
        raise GTokPBAdapterError("D6 diagnostic differs from the P-A inventory")

    first_fit = _project_first_fit(
        physical_d6=physical_d6,
        diagnostic_d6=diagnostic_d6,
    )
    return FrozenScreenCorpusV2(
        full_corpus_manifest_sha256=pa.full_shard_manifest_physical_sha256,
        screen_submanifest_sha256=pa.screen_submanifest_physical_sha256,
        d6_physical_evidence_sha256=pa.d6_physical_evidence_physical_sha256,
        corpus_freeze_receipt_sha256=_sha256_bytes(stored_raw),
        d1_d6_gate_bundle_sha256=str(recomputed["d1_d6_gate_bundle_sha256"]),
        decontamination_receipt_sha256=str(recomputed["decon_receipt_sha256"]),
        first_fit=first_fit,
    )


__all__ = [
    "GTokPBAdapterError",
    "load_frozen_screen_corpus_v2",
]
