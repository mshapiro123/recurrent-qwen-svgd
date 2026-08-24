"""Run registered W1 Phase-B granularity and residual-direction cells."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from transformers import AutoTokenizer

from eval.eval_paper2_bicameral_w1 import (
    EVALUATOR_TAG,
    SCHEDULE,
    atomic_json,
    load_dev2,
    score_arm,
)
from eval.eval_paper2_stage2b_autopsy import _extract_correction_field
from eval.eval_paper2_stage2b_autopsy import _state_digest
from training.paper2_bicameral_w1 import (
    POPULATION_TARGET,
    build_crossfitted_residual_directions,
    build_phase_b_granularity_targets,
    extend_frozen_centroids,
)
from training.paper2_stage2bs_depth_study import INITIALIZATION_SEED_BASE, sha256_file
from training.run_paper2_stage2b_depth import _build_model, _named_trainable_state


WINNER = "l0c"


def _sha256_json_rows(rows: list[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(dict(row), sort_keys=True) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _battery_composition(rows: list[Mapping[str, Any]], labels: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cluster in (0, 1):
        selected = [row for row, label in zip(rows, labels.tolist()) if label == cluster]
        result[str(cluster)] = {
            "rows": len(selected),
            "battery_counts": dict(sorted(Counter(str(row["battery"]) for row in selected).items())),
        }
    return result


def _load_target_cache(path: Path, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("item_ids") != [str(row["item_id"]) for row in rows]:
        raise RuntimeError("W1 Phase-B target-cache population changed")
    if WINNER not in payload.get("families", {}):
        raise RuntimeError("W1 Phase-B winner target is absent")
    return payload


def _centroids(initializers: Mapping[str, Any]) -> torch.Tensor:
    values = torch.stack(
        [
            torch.as_tensor(initializers[f"cluster_{cluster}_correction_mean_unit"]).float()
            for cluster in (0, 1)
        ]
    )
    if values.shape[0] != 2 or not torch.isfinite(values).all():
        raise RuntimeError("W1 frozen Stage-0 centroids changed")
    return values


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_dev2(args.dev2_manifest, args.reference_rows)
    target_cache = _load_target_cache(args.phase_a_targets, rows)
    original_correction = torch.load(
        args.stage0_correction, map_location="cpu", weights_only=False
    )
    initializers = torch.load(args.stage0_initializers, map_location="cpu", weights_only=False)
    if original_correction.get("item_ids") is None or len(original_correction["item_ids"]) != 256:
        raise RuntimeError("W1 Stage-0 correction population changed")
    frozen_labels = torch.as_tensor(initializers["labels"], dtype=torch.long)
    if frozen_labels.shape != (256,):
        raise RuntimeError("W1 Stage-0 frozen labels changed")

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        cache_dir=args.model_cache,
    )
    initialization_seed = INITIALIZATION_SEED_BASE + args.seed
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    torch.cuda.manual_seed_all(initialization_seed)
    wrapper, chain, _groups = _build_model(args)
    state_before = _state_digest(_named_trainable_state(wrapper))

    feature_path = args.private_dir / f"seed_{args.seed}_phase_b_extension_features.pt"
    if feature_path.is_file():
        feature_artifact = torch.load(feature_path, map_location="cpu", weights_only=False)
        if feature_artifact.get("item_ids") != [str(row["item_id"]) for row in rows]:
            raise RuntimeError("W1 resumed Phase-B feature population changed")
    else:
        extracted = _extract_correction_field(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=rows,
            batch_size=args.feature_batch_size,
        )
        feature_artifact = {
            "kind": "paper2_bicameral_w1_phase_b_extension_features_v1",
            "seed": args.seed,
            "rows": len(rows),
            "feature": "unit negative CE gradient at Stage2B loop-4 bridge hidden",
            "oracle_derived_assignment_feature": True,
            "item_ids": list(extracted["item_ids"]),
            "batteries": list(extracted["batteries"]),
            "features": extracted["corrections"][4].float(),
            "parameter_state_digest_before": extracted["parameter_state_digest_before"],
            "parameter_state_digest_after": extracted["parameter_state_digest_after"],
            "parameter_versions_unchanged": extracted["parameter_versions_unchanged"],
        }
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feature_artifact, feature_path)

    if _state_digest(_named_trainable_state(wrapper)) != state_before:
        raise RuntimeError("W1 Phase-B feature extraction mutated the model")
    assignments, extension = extend_frozen_centroids(
        feature_artifact["features"], _centroids(initializers)
    )
    extension["battery_composition"] = _battery_composition(rows, assignments)

    granularity = build_phase_b_granularity_targets(
        target_cache["families"][WINNER].float(), assignments
    )
    residual_directions, residual_receipt = build_crossfitted_residual_directions(
        original_correction["corrections"][4].float(), frozen_labels, directions=3
    )
    assignment_artifact = {
        "kind": "paper2_bicameral_w1_phase_b_assignment_v1",
        "seed": args.seed,
        "rows": len(rows),
        "winner": WINNER,
        "feature_path": feature_path.name,
        "feature_sha256": sha256_file(feature_path),
        "stage0_correction_sha256": sha256_file(args.stage0_correction),
        "stage0_initializers_sha256": sha256_file(args.stage0_initializers),
        "dev2_manifest_sha256": sha256_file(args.dev2_manifest),
        "row_identity_sha256": _sha256_json_rows(
            [{"item_id": str(row["item_id"]), "battery": str(row["battery"])} for row in rows]
        ),
        "assignments": assignments,
        "centroids": _centroids(initializers),
        "extension_receipt": extension,
        "oracle_derived_assignment_feature": True,
        "value_target_tag": POPULATION_TARGET,
    }
    assignment_path = args.private_dir / f"seed_{args.seed}_phase_b_assignments.pt"
    torch.save(assignment_artifact, assignment_path)
    residual_artifact = {
        "kind": "paper2_bicameral_w1_l6_directions_v1",
        "seed": args.seed,
        "directions": residual_directions,
        "receipt": residual_receipt,
        "source_correction_sha256": sha256_file(args.stage0_correction),
        "source_initializers_sha256": sha256_file(args.stage0_initializers),
        "target_tag": POPULATION_TARGET,
    }
    residual_path = args.private_dir / f"seed_{args.seed}_phase_b_l6_directions.pt"
    torch.save(residual_artifact, residual_path)
    pre_score = {
        "kind": "paper2_bicameral_w1_phase_b_pre_score_receipt_v1",
        "seed": args.seed,
        "status": "frozen_before_scoring",
        "assignment_artifact": {
            "path": assignment_path.name,
            "bytes": assignment_path.stat().st_size,
            "sha256": sha256_file(assignment_path),
        },
        "residual_artifact": {
            "path": residual_path.name,
            "bytes": residual_path.stat().st_size,
            "sha256": sha256_file(residual_path),
        },
        "extension": extension,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / f"seed_{args.seed}_phase_b_pre_score.json", pre_score)

    directions = {name: granularity[name].float() for name in ("l1", "l2", "l3")}
    for index, vector in enumerate(residual_directions, start=1):
        expanded = vector.unsqueeze(0).expand(len(rows), -1).contiguous()
        directions[f"l6_u{index}_pos"] = expanded
        directions[f"l6_u{index}_neg"] = -expanded

    cells = []
    for arm, values in directions.items():
        row_path = args.private_dir / f"seed_{args.seed}_phase_b_{arm}.jsonl"
        summary_path = args.output_dir / f"seed_{args.seed}_phase_b_{arm}.json"
        if row_path.is_file() and summary_path.is_file():
            cells.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        scored, summary = score_arm(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=rows,
            baseline_margins=target_cache["baseline_margins"],
            directions=values,
            arm=arm,
            seed=args.seed,
            batch_size=args.margin_batch_size,
        )
        for row in scored:
            row["target_tag"] = POPULATION_TARGET
            row["oracle_routed"] = arm in {"l1", "l3"}
        summary["target_tag"] = POPULATION_TARGET
        summary["oracle_routed"] = arm in {"l1", "l3"}
        from eval.eval_paper2_bicameral_w1 import write_jsonl

        write_jsonl(row_path, scored)
        atomic_json(summary_path, summary)
        cells.append(summary)

    result = {
        "kind": "paper2_bicameral_w1_phase_b_seed_v1",
        "status": "complete_score_only",
        "seed": args.seed,
        "rows": len(rows),
        "winner": WINNER,
        "evaluator": EVALUATOR_TAG,
        "schedule": SCHEDULE,
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": "bfloat16",
            "attention": "sdpa",
        },
        "checkpoint_chain": chain,
        "initialization_state_digest": state_before,
        "pre_score_receipt_sha256": sha256_file(
            args.output_dir / f"seed_{args.seed}_phase_b_pre_score.json"
        ),
        "assignment_artifact_sha256": sha256_file(assignment_path),
        "residual_artifact_sha256": sha256_file(residual_path),
        "extension": extension,
        "cells": cells,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / f"seed_{args.seed}_phase_b_summary.json", result)
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--seed", type=int, choices=(0, 1), required=True)
    result.add_argument("--dev2_manifest", type=Path, required=True)
    result.add_argument("--reference_rows", type=Path, required=True)
    result.add_argument("--phase_a_targets", type=Path, required=True)
    result.add_argument("--stage0_correction", type=Path, required=True)
    result.add_argument("--stage0_initializers", type=Path, required=True)
    result.add_argument("--output_dir", type=Path, required=True)
    result.add_argument("--private_dir", type=Path, required=True)
    result.add_argument("--model_cache", type=Path, required=True)
    result.add_argument("--feature_batch_size", type=int, default=2)
    result.add_argument("--margin_batch_size", type=int, default=4)
    result.add_argument("--device", default="cuda")
    for name in ("migrated", "p33", "i1", "p34", "p35"):
        result.add_argument(f"--{name}", type=Path, required=True)
        result.add_argument(f"--{name}_sha256", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
