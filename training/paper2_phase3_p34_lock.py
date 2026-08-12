"""Deterministic, score-blind construction for the P3.4 executed lock."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from training.paper2_phase3_p31 import canonical_sha256


P34_PANEL_SEED = 20260812
P34_PANEL_ROWS = 1_024
P34_PANEL_GROUP_ROWS = 512
P34_FLOOR_BATTERIES = ("mmlu", "tier1", "arc_easy")
P34_TARGET_BATTERIES = ("gsm8k", "mbpp", "arc_challenge")


def largest_remainder_quotas(
    counts: Mapping[str, int], *, total: int
) -> dict[str, int]:
    if total <= 0 or not counts or any(int(value) <= 0 for value in counts.values()):
        raise ValueError("P3.4 proportional quotas require positive counts and total")
    population = sum(int(value) for value in counts.values())
    if total > population:
        raise ValueError("P3.4 panel quota exceeds the available DEV population")
    exact = {name: total * int(value) / population for name, value in counts.items()}
    quotas = {name: math.floor(value) for name, value in exact.items()}
    remainder = total - sum(quotas.values())
    order = sorted(counts, key=lambda name: (-(exact[name] - quotas[name]), name))
    for name in order[:remainder]:
        quotas[name] += 1
    if sum(quotas.values()) != total:
        raise RuntimeError("P3.4 largest-remainder allocation did not close")
    return quotas


def _rank(row: Mapping[str, Any], *, seed: int) -> bytes:
    key = (
        f"{seed}:p34-tier-sw:{row['battery']}:{row['document_id']}:"
        f"{row['item_id']}"
    )
    return hashlib.sha256(key.encode("utf-8")).digest()


def _identity(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "battery": str(row["battery"]),
        "item_id": str(row["item_id"]),
        "document_id": str(row["document_id"]),
        "content_sha256": str(row["content_sha256"]),
        "partition": str(row["partition"]),
    }


def build_task_panel(
    rows: Iterable[Mapping[str, Any]], *, seed: int = P34_PANEL_SEED
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = [dict(row) for row in rows]
    dev = [row for row in records if str(row.get("partition")) == "dev"]
    if not dev or any(str(row.get("partition")) != "dev" for row in dev):
        raise RuntimeError("P3.4 task panel requires DEV-only source rows")
    if len({(str(row["battery"]), str(row["item_id"])) for row in dev}) != len(dev):
        raise RuntimeError("P3.4 DEV source contains duplicate battery/item identities")

    selected: list[dict[str, Any]] = []
    quotas_by_group: dict[str, dict[str, int]] = {}
    for group, batteries in (
        ("floor", P34_FLOOR_BATTERIES),
        ("target", P34_TARGET_BATTERIES),
    ):
        pools = {
            battery: [row for row in dev if str(row["battery"]) == battery]
            for battery in batteries
        }
        if any(not pool for pool in pools.values()):
            raise RuntimeError(f"P3.4 {group} panel group has an empty battery")
        quotas = largest_remainder_quotas(
            {battery: len(pool) for battery, pool in pools.items()},
            total=P34_PANEL_GROUP_ROWS,
        )
        quotas_by_group[group] = quotas
        for battery in batteries:
            ranked = sorted(pools[battery], key=lambda row: _rank(row, seed=seed))
            chosen = ranked[: quotas[battery]]
            if len({str(row["document_id"]) for row in chosen}) != len(chosen):
                raise RuntimeError(
                    f"P3.4 document-stratified sample repeated a document: {battery}"
                )
            selected.extend(
                {**row, "p34_panel_group": group, "p34_panel_seed": seed}
                for row in chosen
            )
    selected.sort(key=lambda row: (str(row["p34_panel_group"]), str(row["battery"]), _rank(row, seed=seed)))
    if len(selected) != P34_PANEL_ROWS:
        raise RuntimeError(f"P3.4 panel row count changed: {len(selected)}")
    identities = [_identity(row) for row in selected]
    receipt = {
        "kind": "paper2_phase3_p34_task_panel_v1",
        "status": "frozen_before_scoring",
        "seed": seed,
        "rows": len(selected),
        "group_counts": dict(Counter(str(row["p34_panel_group"]) for row in selected)),
        "battery_counts": dict(Counter(str(row["battery"]) for row in selected)),
        "quotas_by_group": quotas_by_group,
        "panel_sha256": canonical_sha256(identities),
        "source_partitions": sorted({str(row["partition"]) for row in selected}),
        "unique_documents": len({str(row["document_id"]) for row in selected}),
        "selection_rule": (
            "largest-remainder proportional allocation within ratified floor/target "
            "groups; SHA256(seed:battery:document_id:item_id) rank within battery"
        ),
        "scores_computed": False,
        "sealed_partitions_touched": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    if receipt["group_counts"] != {"floor": 512, "target": 512}:
        raise RuntimeError("P3.4 panel group balance changed")
    if receipt["unique_documents"] != P34_PANEL_ROWS:
        raise RuntimeError("P3.4 panel is not document-unique")
    return selected, receipt


def panel_identity_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256([_identity(row) for row in rows])
