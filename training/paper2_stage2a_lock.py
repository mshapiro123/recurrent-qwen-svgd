"""Validation gates for the Paper Two Stage 2A executed lock.

The checked-in companion is intentionally a draft.  This module makes the
boundary executable: build and score-blind data preparation may proceed, but
no optimizer may be constructed until every signature field is bound and the
lock is explicitly ratified.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


LOCK_PATH = Path(__file__).with_name("paper2_stage2a_preregistration.draft.json")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _get(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def unresolved_signature_fields(lock: Mapping[str, Any]) -> list[str]:
    fields = lock.get("unresolved_before_ratification", [])
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
        return ["unresolved_before_ratification must be a list"]
    return [str(field) for field in fields]


def validate_stage2a_lock(lock: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if lock.get("kind") != "paper2_stage2a_executed_lock_v1":
        errors.append("kind must identify the Stage 2A executed lock")

    authority = lock.get("authority", {})
    if authority.get("drive_id") != "1GhTafphPuxmfq8hDn0Y-BrjWJCmafZKu":
        errors.append("strategy authority Drive ID changed")
    if authority.get("bytes") != 11_079:
        errors.append("strategy authority byte count changed")
    if authority.get("sha256") != (
        "57c41caa6d9bfe0174f295f6b7a56634ad26718dbb868f3cac9a3d857405ba2c"
    ):
        errors.append("strategy authority SHA-256 changed")

    sealed = lock.get("sealed_partitions", {})
    if sealed.get("confirm_scored") is not False:
        errors.append("CONFIRM must remain sealed")
    if sealed.get("eval_e_scored") is not False:
        errors.append("EVAL-E must remain sealed")
    if sealed.get("remain_sealed") is not True:
        errors.append("sealed-partition invariant is not armed")

    data = lock.get("data_separation", {})
    if data.get("dev_panel_rows") != 1_024:
        errors.append("Stage 2A requires the frozen 1,024-row DEV panel")
    if data.get("memory_slots") != 8_192:
        errors.append("Stage 2A memory slot count changed")
    if data.get("dev_rows_allowed_in_memory") != 0:
        errors.append("DEV rows may not contribute memory content")

    geometry = lock.get("geometry_fit", {})
    if geometry.get("fit_population") != "non_dev_reference_rows_only":
        errors.append("live PCA/transport must fit on non-DEV reference rows only")
    if geometry.get("dev_rows_used") != 0:
        errors.append("live geometry fit touched DEV")
    if geometry.get("diagnostic_dev_transform_reused") is not False:
        errors.append("diagnostic DEV transform may not be reused")
    if geometry.get("rank") != 128:
        errors.append("Stage 2A geometry rank changed")

    arms = lock.get("arms", {})
    required_arms = {"t3a_fingerprint", "t3b_ngram", "shuffled_values", "random_values"}
    if set(arms) != required_arms:
        errors.append("Stage 2A arm set changed")
    fingerprint = arms.get("t3a_fingerprint", {})
    if fingerprint.get("key_source") != "student_layer_6_pca128":
        errors.append("fingerprint key source changed")
    if fingerprint.get("keys_trainable") is not False:
        errors.append("fingerprint keys must be frozen")
    if fingerprint.get("query_transform_trainable") is not False:
        errors.append("fingerprint query transform must be frozen")

    trainable = set(lock.get("trainable_surface", []))
    allowed_trainable = {
        "memory.values",
        "memory.compatibility_projection.*",
        "injection.projection",
        "injection.gate",
    }
    if trainable != allowed_trainable:
        errors.append("trainable surface differs from the issued four-part allowlist")
    frozen = set(lock.get("frozen_surface", []))
    for required in ("substrate.*", "sidecar.*", "memory.keys", "memory.query_projection"):
        if required not in frozen:
            errors.append(f"missing frozen-surface assertion: {required}")

    identity = lock.get("identity_gate", {})
    if identity.get("zero_injection_gate_required") is not True:
        errors.append("zero-gate identity is not required")
    if identity.get("bit_exact_required") is not True:
        errors.append("bit-exact no-memory recovery is not required")

    for path in (
        "data_separation.source_manifest_sha256",
        "data_separation.panel_manifest_sha256",
        "data_separation.firm_knowledge_rule_sha256",
        "geometry_fit.fit_manifest_sha256",
        "geometry_fit.artifact_sha256",
        "initialization.seed_0.sha256",
        "initialization.seed_1.sha256",
    ):
        value = _get(lock, path)
        if value is not None and not _is_sha256(value):
            errors.append(f"{path} is not a SHA-256")

    unresolved = unresolved_signature_fields(lock)
    authorized = bool(lock.get("training_authorized"))
    locked = bool(lock.get("locked_before_training"))
    ratified = bool(lock.get("mark_ratified"))
    if authorized or locked or ratified:
        if not (authorized and locked and ratified):
            errors.append("authorization, lock, and ratification flags must flip together")
        if unresolved:
            errors.append("ratified lock retains unresolved signature fields")
        for path in lock.get("required_bound_paths", []):
            if _get(lock, str(path)) in (None, "", []):
                errors.append(f"ratified lock is missing required binding: {path}")
    else:
        if lock.get("status") != "draft_awaiting_mark_signature":
            errors.append("disabled companion must remain a signature-awaiting draft")
    return errors


def assert_stage2a_training_authorized(lock: Mapping[str, Any]) -> None:
    errors = validate_stage2a_lock(lock)
    if errors:
        raise RuntimeError("Stage 2A lock invalid: " + "; ".join(errors))
    if not (
        lock.get("training_authorized") is True
        and lock.get("locked_before_training") is True
        and lock.get("mark_ratified") is True
    ):
        raise RuntimeError("Stage 2A training remains structurally disabled")


def load_stage2a_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

