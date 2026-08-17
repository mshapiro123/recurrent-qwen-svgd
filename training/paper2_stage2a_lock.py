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
    rulings = lock.get("lock_rulings", {})
    if rulings.get("drive_id") != "1UOfWqVpV4ByPRVeTbsabgaID-6LohmkF":
        errors.append("Stage 2A lock-rulings Drive ID changed")
    if rulings.get("bytes") != 8_972:
        errors.append("Stage 2A lock-rulings byte count changed")
    if rulings.get("sha256") != (
        "445e77a089148d58910acabf1e277600e966068068300df26e11e02b12c98c73"
    ):
        errors.append("Stage 2A lock-rulings SHA-256 changed")
    objective_binding = lock.get("objective_binding", {})
    if objective_binding.get("drive_id") != "1-2iiv8aaTrBvUR2Zxs4V6BW1P8OLotb_":
        errors.append("T3a objective-binding Drive ID changed")
    if objective_binding.get("bytes") != 4_821:
        errors.append("T3a objective-binding byte count changed")
    if objective_binding.get("sha256") != (
        "78cbf2fb397cf2c6319636523a7feea44b1e21e8941ee32e898323e697f18a22"
    ):
        errors.append("T3a objective-binding SHA-256 changed")

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
    if data.get("heldout_nondev_validation_rows") != 512:
        errors.append("Stage 2A non-DEV validation split changed")
    if data.get("validation_rows_allowed_in_memory") != 0:
        errors.append("held-out non-DEV validation rows may not enter memory")
    if data.get("dev_rows_allowed_in_memory") != 0:
        errors.append("DEV rows may not contribute memory content")
    firm = data.get("firm_knowledge_rule", {})
    if firm.get("admission") != "teacher_correct AND family_concurrence":
        errors.append("V(x) admission rule changed")
    if firm.get("probability_thresholds") is not None:
        errors.append("V(x) may not use probability thresholds")
    if _get(firm, "teacher_correct.revision") != (
        "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    ):
        errors.append("V(x) 14B revision changed")
    if _get(firm, "family_concurrence.revision") != (
        "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
    ):
        errors.append("V(x) 32B revision changed")

    geometry = lock.get("geometry_fit", {})
    if geometry.get("fit_population") != "non_dev_reference_rows_only":
        errors.append("live PCA/transport must fit on non-DEV reference rows only")
    if geometry.get("dev_rows_used") != 0:
        errors.append("live geometry fit touched DEV")
    if geometry.get("diagnostic_dev_transform_reused") is not False:
        errors.append("diagnostic DEV transform may not be reused")
    if geometry.get("rank") != 128:
        errors.append("Stage 2A geometry rank changed")
    if geometry.get("fit_fraction") != 0.8 or geometry.get("fit_seed") != 20_260_817:
        errors.append("non-DEV fit split changed")
    if "teacher PCA coordinates" not in str(geometry.get("teacher_pca_contract")):
        errors.append("teacher values are not retained in teacher PCA coordinates")
    if "absent from live memory path" not in str(geometry.get("transport_direction")):
        errors.append("diagnostic Procrustes map leaked into the live memory path")

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
    if fingerprint.get("top_k") != 8 or fingerprint.get("temperature") != 0.07:
        errors.append("registered fingerprint retrieval constants changed")
    if fingerprint.get("training_leave_one_out") != (
        "exclude the current row-owned slot before top-k"
    ):
        errors.append("fingerprint leave-one-out retrieval changed")
    if arms.get("t3b_ngram", {}).get("training_leave_one_out") != (
        "vacuous: the hash tables contain no row-owned entry; no self slot exists"
    ):
        errors.append("n-gram no-self-entry contract changed")
    shuffled = arms.get("shuffled_values", {})
    random_values = arms.get("random_values", {})
    if shuffled.get("values_trainable_after_permutation") is not False:
        errors.append("shuffled values must remain frozen")
    if random_values.get("values_trainable") is not False:
        errors.append("random control values must remain frozen")
    if shuffled.get("seeds") != [0] or random_values.get("seeds") != [0]:
        errors.append("honesty-control seed matrix changed")

    trainable = set(lock.get("trainable_surface", []))
    allowed_trainable = {
        "memory.values",
        "memory.compatibility_projection.*",
        "injection.projection",
        "injection.slot_logits",
        "injection.gate",
    }
    if trainable != allowed_trainable:
        errors.append("trainable surface differs from the ratified allowlist")
    frozen = set(lock.get("frozen_surface", []))
    for required in ("substrate.*", "sidecar.*", "memory.keys", "memory.query_projection"):
        if required not in frozen:
            errors.append(f"missing frozen-surface assertion: {required}")

    identity = lock.get("identity_gate", {})
    if identity.get("zero_injection_gate_required") is not True:
        errors.append("zero-gate identity is not required")
    if identity.get("bit_exact_required") is not True:
        errors.append("bit-exact no-memory recovery is not required")

    injection = lock.get("injection", {})
    if injection.get("module") != "ScratchpadMemoryInjection":
        errors.append("post-initializer memory injection module changed")
    if (
        injection.get("memory_dim"),
        injection.get("scratch_dim"),
        injection.get("scratch_slots"),
    ) != (128, 128, 8):
        errors.append("registered scratch write dimensions changed")
    if injection.get("slot_weight_parameterization") != (
        "2 * sigmoid(slot_logits), nonnegative"
    ):
        errors.append("slot-write parameterization changed")

    training = lock.get("training", {})
    expected_training = {
        "steps": 1_200,
        "batch_size": 128,
        "optimizer": "AdamW",
        "learning_rate": 0.0005,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "warmup_steps": 50,
        "ema_decay": 0.999,
        "checkpoint_steps": [200, 400, 600, 800, 1000, 1200],
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            errors.append(f"registered training field changed: {key}")
    objective = training.get("objective", {})
    expected_objective = {
        "formula": "0.5 * L_CE + 0.5 * L_KL",
        "ce_weight": 0.5,
        "kl_weight": 0.5,
        "kl_direction": "teacher_to_student",
        "temperature": 1.0,
        "teacher_lattice_top_k": 128,
        "renormalize_over_teacher_lattice": True,
        "teacher_revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "teacher_forced_response": True,
        "reduction": "mean over unmasked positions within example, then mean over batch",
        "verifier_mask": False,
        "leave_one_out": True,
        "identical_across_arms": True,
    }
    for key, expected in expected_objective.items():
        if objective.get(key) != expected:
            errors.append(f"registered objective field changed: {key}")
    if "position zero excluded" not in str(objective.get("answer_region_mask")):
        errors.append("answer-region mask does not exclude position zero")
    if "top-k 8 at temperature 0.07" not in str(objective.get("full_inference_graph")):
        errors.append("objective inference graph changed")
    if training.get("objective_formula_source") != (
        "docs/STRATEGY_T3A_OBJECTIVE_BINDING_20260817.receipt.json"
    ):
        errors.append("objective formula source changed")

    evaluation = lock.get("evaluation", {})
    if evaluation.get("paired_sign_test_alpha") != 0.05:
        errors.append("paired sign-test alpha changed")
    if evaluation.get("shuffled_flat_equivalence_band") != [-3, 3]:
        errors.append("control equivalence band changed")
    if evaluation.get("chi_threshold") != 0.0005:
        errors.append("chi threshold changed")
    pricing = lock.get("confirm_pricing", {})
    if pricing.get("applies_to_this_screen") is not False:
        errors.append("CONFIRM campaign target was applied to the T3a screen")
    if pricing.get("selected_main_campaign_minimum_absolute_points") != 0.02:
        errors.append("successor campaign pricing target changed")

    for path in (
        "data_separation.source_manifest_sha256",
        "data_separation.panel_manifest_sha256",
        "data_separation.firm_knowledge_rule_sha256",
        "data_separation.validation_manifest_sha256",
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
