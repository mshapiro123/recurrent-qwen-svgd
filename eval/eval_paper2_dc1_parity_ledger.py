"""Population-honest pre/post-D0 forced-depth parity ledger."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import load_drafter, read_jsonl
from eval.eval_paper2_dc0_depth_by_append import parameter_fingerprint
from eval.eval_paper2_dc1_preflight import select_probe_indices
from eval.eval_speculative_depth_d0_floor import load_partition_cache
from training.paper2_dc1_followups import (
    POST_D0_CHECKPOINT_SHA256,
    PRE_D0_CHECKPOINT_SHA256,
    floor_payload_has_all_positions,
    transition_ledger,
)
from training.paper2_dc1 import PREFLIGHT_POSITION_BUDGET
from training.speculative_depth_d0_corpus import sha256_file
from training.speculative_depth_d0_spec import deterministic_argmax_fp32


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


@torch.inference_mode()
def forced_depth_grid(
    checkpoint: Path,
    rows: list[dict[str, Any]],
    *,
    expected_sha256: str,
    resume_dir: Path,
    device: str,
    dtype: str,
    attn_implementation: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if sha256_file(checkpoint) != expected_sha256:
        raise RuntimeError(f"forced-depth parity checkpoint SHA mismatch: {checkpoint}")
    _tokenizer, wrapper, resize, _original_vocab = load_drafter(
        checkpoint=checkpoint,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    wrapper.eval()
    before = parameter_fingerprint(wrapper)
    outputs: list[torch.Tensor] = []
    tie_cells = argmax_cells = 0
    resume_dir.mkdir(parents=True, exist_ok=True)
    for row_index, row in enumerate(rows):
        cache_path = resume_dir / f"row_{row_index:06d}.pt"
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (
                payload.get("checkpoint_sha256") != expected_sha256
                or payload.get("row_index") != row_index
                or payload.get("forced_depths") != [1, 2, 3]
            ):
                raise RuntimeError(f"parity resume cache mismatch: {cache_path}")
        else:
            input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            result = wrapper(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_loops=3,
                use_cache=False,
                return_dict=True,
                return_loop_logits=True,
            )
            if result.loop_logits is None:
                raise RuntimeError("parity ledger requires per-loop logits")
            logits = result.loop_logits[0, 0, :, :-1, : resize.original_tokenizer_size]
            selected, ties = deterministic_argmax_fp32(logits, dim=-1)
            payload = {
                "kind": "paper2_dc1_parity_row_cache",
                "checkpoint_sha256": expected_sha256,
                "row_index": row_index,
                "forced_depths": [1, 2, 3],
                "predictions": selected.transpose(0, 1).cpu(),
                "tie_cells": int(ties.sum()),
                "argmax_cells": int(ties.numel()),
            }
            temporary = cache_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, cache_path)
        outputs.append(payload["predictions"].long())
        tie_cells += int(payload.get("tie_cells", 0))
        argmax_cells += int(payload.get("argmax_cells", 0))
        if row_index == 0 or (row_index + 1) % 16 == 0 or row_index + 1 == len(rows):
            print(
                f"dc1_parity_progress checkpoint={expected_sha256[:8]} "
                f"rows={row_index + 1}/{len(rows)}",
                flush=True,
            )
    after = parameter_fingerprint(wrapper)
    if before != after:
        raise RuntimeError("read-only parity evaluation mutated the checkpoint")
    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0), {
        "checkpoint_sha256": expected_sha256,
        "checkpoint_mutated": False,
        "tie_policy": "fp32 logits; exact ties choose the lowest token id",
        "tie_cells": tie_cells,
        "argmax_cells": argmax_cells,
        "tie_rate": tie_cells / argmax_cells if argmax_cells else None,
    }


def score_saved_floor(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["all_position_rows"]
    predictions = torch.tensor([row["predictions"] for row in rows], dtype=torch.long)
    teacher = torch.tensor([row["teacher_7b"] for row in rows], dtype=torch.long)
    split = predictions[:, 0].eq(teacher)
    return {
        "transition_1_to_2": transition_ledger(
            predictions, teacher, before_depth=1, after_depth=2, split_mask=split
        ),
        "transition_2_to_3": transition_ledger(
            predictions, teacher, before_depth=2, after_depth=3, split_mask=split
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor_private_rows", required=True)
    parser.add_argument("--floor_summary", required=True)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--teacher_cache_summary", required=True)
    parser.add_argument("--pre_checkpoint", required=True)
    parser.add_argument("--post_checkpoint", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--private_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    floor_payload = json.loads(Path(args.floor_private_rows).read_text(encoding="utf-8"))
    floor_summary = json.loads(Path(args.floor_summary).read_text(encoding="utf-8"))
    has_all = floor_payload_has_all_positions(
        floor_payload,
        rejected_positions=int(floor_summary["rejected_positions"]),
    )
    coverage_audit = {
        "floor_private_rows_sha256": sha256_file(args.floor_private_rows),
        "rejected_positions": int(floor_summary["rejected_positions"]),
        "all_position_rows_present": has_all,
        "saved_rows_key": (
            "all_position_rows" if has_all else "rejected_only_rows_or_legacy_floor_payload"
        ),
    }

    if has_all:
        pre = score_saved_floor(floor_payload)
        summary = {
            "kind": "paper2_dc1_pre_post_d0_parity_ledger",
            "status": "complete_saved_full_population_branch",
            "case_applied": "saved_D0_calibration_floor_predictions_cover_all_positions",
            "coverage_audit": coverage_audit,
            "population": {
                "partition": "D0 calibration",
                "row_identical_to_DEV_C": False,
                "population_definition_matched": True,
            },
            "pre_d0": pre,
            "post_d0_same_rows": None,
            "ungated_fixed_depth2_pre_d0_net_utility": pre["transition_1_to_2"][
                "all_positions"
            ]["net_correct_delta"],
            "training_started": False,
            "optimizer_steps": 0,
            "evaluation_c_touched": False,
        }
        write_json(args.output_summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0

    rows = read_jsonl(args.data_jsonl)
    selected_indices = select_probe_indices(
        rows, position_budget=PREFLIGHT_POSITION_BUDGET
    )
    selected_rows = [rows[index] for index in selected_indices]
    teacher_summary = json.loads(Path(args.teacher_cache_summary).read_text(encoding="utf-8"))
    teacher_rows = load_partition_cache(teacher_summary, "teacher_7b", "dev_c")
    targets = torch.cat(
        [teacher_rows[index]["teacher_greedy_token_id"].long() for index in selected_indices]
    )
    private = Path(args.private_dir)
    pre_predictions, pre_integrity = forced_depth_grid(
        Path(args.pre_checkpoint),
        selected_rows,
        expected_sha256=PRE_D0_CHECKPOINT_SHA256,
        resume_dir=private / "pre_d0_rows",
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    post_predictions, post_integrity = forced_depth_grid(
        Path(args.post_checkpoint),
        selected_rows,
        expected_sha256=POST_D0_CHECKPOINT_SHA256,
        resume_dir=private / "post_d0_rows",
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    pre_split = pre_predictions[:, 0].eq(targets)
    post_split = post_predictions[:, 0].eq(targets)
    pre = {
        "transition_1_to_2": transition_ledger(
            pre_predictions, targets, before_depth=1, after_depth=2, split_mask=pre_split
        ),
        "transition_2_to_3": transition_ledger(
            pre_predictions, targets, before_depth=2, after_depth=3, split_mask=pre_split
        ),
        "integrity": pre_integrity,
    }
    post = {
        "transition_1_to_2": transition_ledger(
            post_predictions, targets, before_depth=1, after_depth=2, split_mask=post_split
        ),
        "transition_2_to_3": transition_ledger(
            post_predictions, targets, before_depth=2, after_depth=3, split_mask=post_split
        ),
        "integrity": post_integrity,
    }
    pre_net = pre["transition_1_to_2"]["all_positions"]["net_correct_delta"]
    post_net = post["transition_1_to_2"]["all_positions"]["net_correct_delta"]
    summary = {
        "kind": "paper2_dc1_pre_post_d0_parity_ledger",
        "status": "complete_exact_dev_c_fallback_branch",
        "case_applied": "legacy_D0_floor_predictions_are_rejected_only",
        "coverage_audit": coverage_audit,
        "population": {
            "partition": "DEV-C reusable development surface",
            "rows": len(selected_rows),
            "positions": len(targets),
            "row_identical_pre_post": True,
            "teacher_targets_identical_pre_post": True,
            "data_jsonl_sha256": sha256_file(args.data_jsonl),
            "teacher_cache_summary_sha256": sha256_file(args.teacher_cache_summary),
            "evaluation_c_touched": False,
        },
        "pre_d0": pre,
        "post_d0": post,
        "paired_comparison": {
            "pre_d0_net": pre_net,
            "post_d0_net": post_net,
            "pre_minus_post_net": pre_net - post_net,
            "scope": "descriptive DEV-C diagnostic; not a registered headline surface",
        },
        "ungated_fixed_depth2_pre_d0_net_utility": pre_net,
        "training_started": False,
        "optimizer_steps": 0,
        "evaluation_c_touched": False,
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
