"""Locked contracts and adjudication for the Stage 2B-S depth study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


LOCK_KIND = "paper2_stage2bs_depth_study_cascade_lock_v3"
SCHEDULES = (
    "native_interleaved",
    "deferred_terminal_write_no_reentry",
    "per_loop_write_no_reentry",
    "partial_interleave_pairs",
)
EXPECTED_NATIVE_COUNTS = {0: [162, 10, 2, 2], 1: [162, 9, 5, 1]}
INITIALIZATION_SEED_BASE = 20260819
EXPECTED_INITIALIZATION_STATE_DIGESTS = {
    0: "f4c8bcc7497c5502e2ea321278e85c5ef5a812755b8fe413dcfc507c3b003b18",
    1: "3421b93d0f0a7ae6005a10c170e96f9ea7dfc1396a4c66180b28fd925ffa36c6",
}
ADDITIVITY_FLOOR_ROWS = 20
DIRECT_SCHEDULE = "deferred_terminal_write_no_reentry"
DIRECT_AMPLITUDE = 0.05
FINAL_SCHEDULE = "per_loop_write_no_reentry"
FINAL_AMPLITUDE = 0.05
FINAL_FLAT_BAND_ROWS = 9


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("kind") != LOCK_KIND:
        raise RuntimeError("wrong Stage 2B-S depth-study lock kind")
    if lock.get("status") != "LOCKED" or lock.get("locked_before_score") is not True:
        raise RuntimeError("Stage 2B-S depth study is not locked before scoring")
    if lock.get("training_authorized") is not False:
        raise RuntimeError("Stage 2B-S depth study must not authorize training")
    if int(lock.get("optimizer_steps_allowed", -1)) != 0:
        raise RuntimeError("Stage 2B-S depth-study optimizer allowance changed")
    authority = lock.get("authority", {})
    if authority != {
        "drive_id": "1BL-2x_mdRqJY56u55Tyf1kXHBT4JkHom",
        "bytes": 6525,
        "sha256": "868c2ba8a839c075d3fba14315e0242846b7c90557e673dad9eda3a24fa7017e",
    }:
        raise RuntimeError("Stage 2B-S depth-study authority changed")
    basis = lock.get("basis", {})
    if basis != {
        "drive_id": "1x8BTHXEJnhVHhtsI7mhRG_Wy_vf49iTi",
        "bytes": 12336,
        "sha256": "c28eca58e3b681b81196f6ff8f724533eca1aa5a184db82360c6e3bf020ba878",
    }:
        raise RuntimeError("Stage 2B-S depth-study basis changed")
    if lock.get("math_foundations") != {
        "drive_id": "1OfUuCvwTxlx4R1LEN7Ns3uCoGFy5oKa3",
        "bytes": 18018,
        "sha256": "6a52d1bc1e57fd403cfaa767b6029b5d7a8f206751bfeb03e4a80eb08b0ce7e7",
    }:
        raise RuntimeError("Stage 2B-S math-foundations basis changed")
    if lock.get("final_cell_authority") != {
        "drive_id": "1JB2gFt7cwthK4gyY4BgBCVNUfKoStDqF",
        "bytes": 4196,
        "sha256": "60b52390d2db1e898a88bffaba494211e700322154c08208edc462f684c20911",
    }:
        raise RuntimeError("Stage 2B-S final-cell authority changed")
    if lock.get("expected_native_counts") != {
        str(seed): values for seed, values in EXPECTED_NATIVE_COUNTS.items()
    }:
        raise RuntimeError("Stage 2B-S native preflight expectations changed")
    initialization = lock.get("initialization", {})
    if initialization.get("seed_base") != INITIALIZATION_SEED_BASE:
        raise RuntimeError("Stage 2B-S initialization seed convention changed")
    if initialization.get("state_digest_by_seed") != {
        str(seed): digest
        for seed, digest in EXPECTED_INITIALIZATION_STATE_DIGESTS.items()
    }:
        raise RuntimeError("Stage 2B-S initialization state identity changed")
    if initialization.get("estimator_source") != (
        "banked Stage 2B-S prelude initialization receipts"
    ):
        raise RuntimeError("Stage 2B-S initialization estimator source changed")
    if lock.get("schedules") != list(SCHEDULES):
        raise RuntimeError("Stage 2B-S schedule set changed")
    if lock.get("amplitude_cross") != [0.0, 0.02, 0.05]:
        raise RuntimeError("Stage 2B-S amplitude cross changed")
    if int(lock.get("additivity_floor_rows", -1)) != ADDITIVITY_FLOOR_ROWS:
        raise RuntimeError("Stage 2B-S additivity floor changed")
    if lock.get("endpoints") != ["initialization", "ema_step_1000"]:
        raise RuntimeError("Stage 2B-S endpoint set changed")
    panels = lock.get("panels", {})
    if panels.get("generative_rows") != 461 or panels.get("dev2_margin_rows") != 2048:
        raise RuntimeError("Stage 2B-S panel cardinality changed")
    if panels.get("confirm_scored") is not False or panels.get("eval_e_scored") is not False:
        raise RuntimeError("Stage 2B-S sealed partition contract changed")
    runtime = lock.get("runtime", {})
    if runtime.get("generation_batch_size") != 8 or runtime.get("margin_batch_size") != 2:
        raise RuntimeError("Stage 2B-S banked estimator batch sizes changed")
    semantics = lock.get("k_semantics", {})
    if semantics.get("native_interleaved") != "total_recurrent_passes_including_identity_pass":
        raise RuntimeError("native K semantics changed")
    for schedule in SCHEDULES[1:]:
        if semantics.get(schedule) != "sidecar_updates_after_one_identity_pass":
            raise RuntimeError(f"variant K semantics changed: {schedule}")
    if lock.get("partial_interleave_rule") != (
        "group_sidecar_updates_in_ordered_pairs; after each complete pair write once and "
        "reenter once; an odd terminal update forms a final one-update write-and-reentry group"
    ):
        raise RuntimeError("partial-interleave rule changed")
    cascade = lock.get("cascade", {})
    if cascade.get("direct_discriminator") != {
        "endpoint": "initialization",
        "schedule": DIRECT_SCHEDULE,
        "amplitude": DIRECT_AMPLITUDE,
        "k_values": [1, 2, 3, 4],
        "stop_after_both_seeds": True,
    }:
        raise RuntimeError("Stage 2B-S direct-discriminator contract changed")
    if cascade.get("clearance_rule") != "both_seeds_any_k_gt_1_at_or_above_native_k1_plus_20":
        raise RuntimeError("Stage 2B-S cascade clearance rule changed")
    if cascade.get("seed_split_rule") != "stop_and_relay_before_any_branch":
        raise RuntimeError("Stage 2B-S cascade seed-split rule changed")
    if cascade.get("fallback_order") != [
        "per_loop_write_no_reentry",
        "partial_interleave_pairs",
    ]:
        raise RuntimeError("Stage 2B-S cascade fallback order changed")
    if cascade.get("final_deciding_cell_requires_dev2_margins") is not True:
        raise RuntimeError("Stage 2B-S final margin requirement changed")
    if cascade.get("all_three_fail_action") != "bank_SUBTRACTIVE_and_close_implementation_line":
        raise RuntimeError("Stage 2B-S close-out clause changed")
    if cascade.get("final_cell") != {
        "endpoint": "initialization",
        "schedule": FINAL_SCHEDULE,
        "amplitude": FINAL_AMPLITUDE,
        "k_values": [1, 2, 3, 4],
        "both_seeds": True,
        "flat_band_rows": FINAL_FLAT_BAND_ROWS,
        "effect_floor_rows": ADDITIVITY_FLOOR_ROWS,
        "flat_requires_accumulated_write_growth": True,
        "flat_action": "bank_SCHEDULE_NEUTRALIZED_and_close_cascade",
        "improves_or_collapses_or_split_action": "stop_and_relay",
        "partial_interleave_authorized": False,
    }:
        raise RuntimeError("Stage 2B-S final-cell contract changed")
    if cascade.get("dual_write_telemetry") != {
        "accumulated_raw": (
            "sum across writes of active-token RMS for each incremental bridge delta "
            "before aggregation"
        ),
        "deployed_raw": (
            "active-token RMS of final post-write hidden minus the pre-write post-coda hidden"
        ),
        "normalized": (
            "each raw magnitude divided by active-token RMS of the pre-write post-coda hidden"
        ),
    }:
        raise RuntimeError("Stage 2B-S dual-write telemetry contract changed")
    if cascade.get("flat_followup_margins") != {
        "schedule": DIRECT_SCHEDULE,
        "endpoint": "initialization",
        "amplitude": DIRECT_AMPLITUDE,
        "k_values": [1, 4],
        "seeds": [0, 1],
        "rows_per_cell": 2048,
    }:
        raise RuntimeError("Stage 2B-S final margin contract changed")


def load_lock(path: str | Path) -> dict[str, Any]:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_lock(lock)
    return lock


def schedule_amplitudes(schedule: str) -> tuple[float, ...]:
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown Stage 2B-S schedule: {schedule}")
    return (0.0, 0.02, 0.05) if schedule in SCHEDULES[:2] else (0.05,)


def resolve_keys(
    rows: Sequence[Mapping[str, Any]], *, native_k1_by_seed: Mapping[int, int]
) -> dict[str, Any]:
    """Apply only the preregistered keys to primary initialization counts."""

    expected_seeds = set(native_k1_by_seed)
    seen_seeds = {int(row["seed"]) for row in rows}
    if seen_seeds != expected_seeds:
        raise RuntimeError("Stage 2B-S key resolution lacks both registered seeds")
    by_seed: dict[int, list[Mapping[str, Any]]] = {
        seed: [row for row in rows if int(row["seed"]) == seed] for seed in expected_seeds
    }
    additive_by_seed = {}
    deferred_by_seed = {}
    native_by_seed = {}
    qualifying = {}
    for seed, cells in by_seed.items():
        bar = int(native_k1_by_seed[seed])
        seed_qualifying = [
            dict(cell)
            for cell in cells
            if int(cell["k"]) > 1 and int(cell["correct"]) - bar >= ADDITIVITY_FLOOR_ROWS
        ]
        qualifying[str(seed)] = seed_qualifying
        additive_by_seed[seed] = bool(seed_qualifying)
        deferred_by_seed[seed] = any(
            cell["schedule"] == "deferred_terminal_write_no_reentry"
            for cell in seed_qualifying
        )
        native_by_seed[seed] = any(
            cell["schedule"] == "native_interleaved" for cell in seed_qualifying
        )
    additive = all(additive_by_seed.values())
    native_additive = all(native_by_seed.values())
    schedule_dependent = all(deferred_by_seed.values()) and not any(native_by_seed.values())
    subtractive = not any(additive_by_seed.values())
    seed_disagreement = len(set(additive_by_seed.values())) > 1 or len(
        set(deferred_by_seed.values())
    ) > 1
    return {
        "ADDITIVE": additive,
        "SUBTRACTIVE": subtractive,
        "SCHEDULE_DEPENDENT": schedule_dependent,
        "NATIVE_ADDITIVE": native_additive,
        "seed_disagreement": seed_disagreement,
        "qualifying_cells_by_seed": qualifying,
        "requires_strategy_escalation": bool(seed_disagreement or (additive and not schedule_dependent)),
    }


def resolve_direct_branch(
    rows: Sequence[Mapping[str, Any]], *, native_k1_by_seed: Mapping[int, int]
) -> dict[str, Any]:
    """Resolve only the ratified direct-discriminator branch gate."""

    expected_seeds = set(native_k1_by_seed)
    by_seed: dict[int, list[Mapping[str, Any]]] = {
        seed: [row for row in rows if int(row["seed"]) == seed]
        for seed in expected_seeds
    }
    if {seed for seed, cells in by_seed.items() if cells} != expected_seeds:
        raise RuntimeError("Stage 2B-S direct discriminator lacks both seeds")
    clears: dict[int, bool] = {}
    best: dict[int, dict[str, Any]] = {}
    for seed, cells in by_seed.items():
        if any(
            row.get("endpoint") != "initialization"
            or row.get("schedule") != DIRECT_SCHEDULE
            or float(row.get("amplitude", -1.0)) != DIRECT_AMPLITUDE
            for row in cells
        ):
            raise RuntimeError("Stage 2B-S direct discriminator contains an off-contract cell")
        if sorted(int(row["k"]) for row in cells) != [1, 2, 3, 4]:
            raise RuntimeError("Stage 2B-S direct discriminator K coverage changed")
        candidates = [row for row in cells if int(row["k"]) > 1]
        best_cell = max(candidates, key=lambda row: int(row["correct"]))
        delta = int(best_cell["correct"]) - int(native_k1_by_seed[seed])
        clears[seed] = delta >= ADDITIVITY_FLOOR_ROWS
        best[seed] = {
            "k": int(best_cell["k"]),
            "correct": int(best_cell["correct"]),
            "delta_vs_native_k1_rows": delta,
        }
    if len(set(clears.values())) > 1:
        branch = "STOP_SEED_SPLIT_REQUIRED_RELAY"
    elif all(clears.values()):
        branch = "RECOVERY_BRANCH_AUTHORIZED_AWAITING_RELAY"
    else:
        branch = "FALLBACK_BRANCH_AUTHORIZED_AWAITING_RELAY"
    return {
        "branch": branch,
        "clears_k1_plus_20_by_seed": {str(seed): clears[seed] for seed in sorted(clears)},
        "best_higher_k_by_seed": {str(seed): best[seed] for seed in sorted(best)},
        "requires_relay_before_branch": True,
    }


def resolve_final_cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the ratified final-cell map without opening partial interleave."""

    by_seed = {
        seed: [row for row in rows if int(row["seed"]) == seed]
        for seed in (0, 1)
    }
    if any(sorted(int(row["k"]) for row in cells) != [1, 2, 3, 4] for cells in by_seed.values()):
        raise RuntimeError("Stage 2B-S final cell lacks registered K coverage")
    classifications: dict[int, str] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    for seed, cells in by_seed.items():
        if any(
            row.get("endpoint") != "initialization"
            or row.get("schedule") != FINAL_SCHEDULE
            or float(row.get("amplitude", -1.0)) != FINAL_AMPLITUDE
            for row in cells
        ):
            raise RuntimeError("Stage 2B-S final cell contains an off-contract row")
        ordered = sorted(cells, key=lambda row: int(row["k"]))
        counts = [int(row["correct"]) for row in ordered]
        accumulated = [float(row["accumulated_write_magnitude_mean"]) for row in ordered]
        improves = max(counts[1:]) - 162 >= ADDITIVITY_FLOOR_ROWS
        collapses = 162 - min(counts[1:]) >= ADDITIVITY_FLOOR_ROWS
        flat = all(abs(value - counts[0]) <= FINAL_FLAT_BAND_ROWS for value in counts)
        grows = all(
            later > earlier
            for earlier, later in zip(accumulated, accumulated[1:])
        )
        if improves:
            classification = "IMPROVES_REQUIRED_RELAY"
        elif collapses:
            classification = "COLLAPSES_REQUIRED_RELAY"
        elif flat and grows:
            classification = "FLAT_ACCUMULATING"
        else:
            classification = "AMBIGUOUS_REQUIRED_RELAY"
        classifications[seed] = classification
        diagnostics[seed] = {
            "correct_by_k": counts,
            "delta_vs_native_k1_by_k": [value - 162 for value in counts],
            "accumulated_write_magnitude_mean_by_k": accumulated,
            "deployed_write_magnitude_mean_by_k": [
                float(row["deployed_write_magnitude_mean"]) for row in ordered
            ],
            "flat_within_rows": FINAL_FLAT_BAND_ROWS,
            "accumulated_write_strictly_grows": grows,
            "classification": classification,
        }
    values = set(classifications.values())
    if values == {"FLAT_ACCUMULATING"}:
        verdict = "SCHEDULE_NEUTRALIZED_AWAITING_MARGIN_BANK"
        score_margins = True
    elif len(values) > 1:
        verdict = "STOP_SEED_SPLIT_REQUIRED_RELAY"
        score_margins = False
    else:
        verdict = next(iter(values))
        score_margins = False
    return {
        "verdict": verdict,
        "classification_by_seed": {
            str(seed): classifications[seed] for seed in sorted(classifications)
        },
        "diagnostics_by_seed": {
            str(seed): diagnostics[seed] for seed in sorted(diagnostics)
        },
        "score_registered_deferred_margins": score_margins,
        "partial_interleave_authorized": False,
        "requires_relay": not score_margins,
    }
