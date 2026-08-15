"""Rebuild the P3 oracle cache under the exact BF16 serving reader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from eval.cache_paper2_phase3_agreement_oracle import (
    _lm_head,
    load_selected_anchor_hidden,
)
from training.paper2_phase3_p35 import (
    assert_source_anchor_identity,
    repaired_oracle_payload,
)
from training.run_paper2_phase3_p33 import atomic_torch_save, read_jsonl, write_json
from training.paper2_phase3_p33_prep import sha256_file


def selected_hidden_in_cache_order(
    *,
    records: list[dict[str, Any]],
    cache_record_ids: list[str],
    sources: Mapping[str, tuple[Path, Path]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    hidden, anchors, _lookup, receipt = load_selected_anchor_hidden(
        records=records, sources=sources
    )
    selected = hidden[
        anchors,
        torch.tensor([int(row["horizon"]) - 1 for row in records], dtype=torch.long),
    ]
    by_id = {str(row["record_id"]): selected[index] for index, row in enumerate(records)}
    if len(by_id) != len(records) or set(by_id) != set(cache_record_ids):
        raise RuntimeError("serving-cache repair record population changed")
    return torch.stack([by_id[record_id] for record_id in cache_record_ids]), receipt


def build_repaired_cache(
    *,
    old_summary: Path,
    old_private: Path,
    new_summary: Path,
    new_private: Path,
    positive_audit: Path,
    prior_cache: Path,
    output_cache: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    records = read_jsonl(positive_audit)
    if len(records) != 4096:
        raise RuntimeError("P3.5 serving-cache repair requires all 4096 positive rows")
    prior = torch.load(prior_cache, map_location="cpu", weights_only=False)
    if prior.get("kind") != "paper2_phase3_agreement_oracle_direction_cache_v1":
        raise RuntimeError("P3.5 serving-cache repair requires the registered v1 cache")
    record_ids = [str(value) for value in prior["record_ids"]]
    sources = {
        "old": (old_summary, old_private),
        "new": (new_summary, new_private),
    }
    selected, hidden_receipt = selected_hidden_in_cache_order(
        records=records, cache_record_ids=record_ids, sources=sources
    )
    lm_head, lm_head_receipt = _lm_head(sources)
    repaired = repaired_oracle_payload(
        prior=prior,
        selected_hidden=selected.to(device),
        lm_head_weight=lm_head.to(device),
    )
    identity = assert_source_anchor_identity(
        cache=repaired,
        selected_hidden=selected.to(device),
        lm_head_weight=lm_head.to(device),
    )
    repaired.update(
        {
            "prior_cache_sha256": sha256_file(prior_cache),
            "positive_audit_sha256": sha256_file(positive_audit),
            "hidden_receipt": hidden_receipt,
            "lm_head": lm_head_receipt,
            "source_anchor_identity": identity,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "confirm_scored": False,
            "eval_e_scored": False,
        }
    )
    atomic_torch_save(repaired, output_cache)
    summary = {
        "kind": "paper2_phase3_serving_oracle_cache_repair_summary_v1",
        "status": "complete_exact_identity_no_training",
        "cache": {
            "path": str(output_cache),
            "sha256": sha256_file(output_cache),
            "rows": len(record_ids),
        },
        "source_anchor_identity": identity,
        "prior_cache_sha256": repaired["prior_cache_sha256"],
        "positive_audit_sha256": repaired["positive_audit_sha256"],
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    write_json(output_cache.with_suffix(".summary.json"), summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "old_summary",
        "old_private",
        "new_summary",
        "new_private",
        "positive_audit",
        "prior_cache",
        "output_cache",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    result = build_repaired_cache(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
