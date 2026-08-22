"""Locked contracts and adjudication for the Stage 2B-S depth study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


LOCK_KIND = "paper2_stage2bs_depth_study_lock_v1"
SCHEDULES = (
    "native_interleaved",
    "deferred_terminal_write_no_reentry",
    "per_loop_write_no_reentry",
    "partial_interleave_pairs",
)
EXPECTED_NATIVE_COUNTS = {0: [162, 10, 2, 2], 1: [162, 9, 5, 2]}
INITIALIZATION_SEED_BASE = 20260819
EXPECTED_INITIALIZATION_STATE_DIGESTS = {
    0: "f4c8bcc7497c5502e2ea321278e85c5ef5a812755b8fe413dcfc507c3b003b18",
    1: "3421b93d0f0a7ae6005a10c170e96f9ea7dfc1396a4c66180b28fd925ffa36c6",
}
ADDITIVITY_FLOOR_ROWS = 20


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
        "drive_id": "1fQVbb8PrJPOHwLwpnv7mj73ON6uVMZBa",
        "bytes": 7423,
        "sha256": "1f9d889788a765631d2830153aa06508901f05930efbeb225567337ff048915e",
    }:
        raise RuntimeError("Stage 2B-S depth-study authority changed")
    basis = lock.get("basis", {})
    if basis != {
        "drive_id": "122c2W-ITzUlwLncl3DZRSonsYL3qg7Z6",
        "bytes": 13221,
        "sha256": "d9200a484160142a36b5579ba346aed19b7b6b6e5ef0c4f143ffcd85b6b087b4",
    }:
        raise RuntimeError("Stage 2B-S depth-study basis changed")
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
