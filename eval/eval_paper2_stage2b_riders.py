"""Score-only Stage 2B-D seed-ensemble and R-1 desk audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _aligned(left: Iterable[dict[str, Any]], right: Iterable[dict[str, Any]]):
    left_index = {str(row["item_id"]): row for row in left}
    right_index = {str(row["item_id"]): row for row in right}
    if set(left_index) != set(right_index):
        raise RuntimeError("rider row sets are not identical")
    for item_id in sorted(left_index):
        yield item_id, left_index[item_id], right_index[item_id]


def seed_ensemble_probe(
    seed0_rows: Iterable[dict[str, Any]], seed1_rows: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose the endpoint with the higher row-minimum margin; ties use seed 0."""

    output = []
    selected = Counter()
    for item_id, left, right in _aligned(seed0_rows, seed1_rows):
        left_margin = float(left["answer_token_margin_minimum"])
        right_margin = float(right["answer_token_margin_minimum"])
        winner = 1 if right_margin > left_margin else 0
        chosen = right if winner else left
        selected[f"seed_{winner}"] += 1
        output.append(
            {
                "item_id": item_id,
                "battery": chosen["battery"],
                "selected_seed": winner,
                "seed_0_margin": left_margin,
                "seed_1_margin": right_margin,
                "seed_0_correct": bool(left["augmented_correct"]),
                "seed_1_correct": bool(right["augmented_correct"]),
                "ensemble_correct": bool(chosen["augmented_correct"]),
                "ensemble_prediction": chosen.get("prediction"),
            }
        )
    counts = {
        "seed_0": sum(bool(row["seed_0_correct"]) for row in output),
        "seed_1": sum(bool(row["seed_1_correct"]) for row in output),
        "ensemble": sum(bool(row["ensemble_correct"]) for row in output),
    }
    by_battery = {}
    for battery in sorted({row["battery"] for row in output}):
        rows = [row for row in output if row["battery"] == battery]
        by_battery[battery] = {
            "rows": len(rows),
            "seed_0_correct": sum(bool(row["seed_0_correct"]) for row in rows),
            "seed_1_correct": sum(bool(row["seed_1_correct"]) for row in rows),
            "ensemble_correct": sum(bool(row["ensemble_correct"]) for row in rows),
        }
    summary = {
        "kind": "paper2_stage2b_seed_ensemble_probe_v1",
        "status": "complete_score_only_cached_rows",
        "arbitration": "higher answer_token_margin_minimum wins; exact ties select seed 0",
        "rows": len(output),
        "counts": counts,
        "ensemble_minus_best_seed": counts["ensemble"] - max(counts["seed_0"], counts["seed_1"]),
        "selected_counts": dict(selected),
        "by_battery": by_battery,
        "optimizer_constructed": False,
        "training_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    return output, summary


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def runtime_discordance_audit(
    source_rows: Iterable[dict[str, Any]], diagnostic_rows: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    changed = []
    stable_margins = []
    changed_margins = []
    first_divergence = Counter()
    high_margin_changed = []
    for item_id, source, diagnostic in _aligned(source_rows, diagnostic_rows):
        source_prediction = source.get("prediction", source.get("generated_token_ids"))
        diagnostic_prediction = diagnostic.get("prediction", diagnostic.get("generated_token_ids"))
        source_margin = float(source["answer_token_margin_minimum"])
        diagnostic_margin = float(diagnostic["answer_token_margin_minimum"])
        margin = min(source_margin, diagnostic_margin)
        if source_prediction == diagnostic_prediction:
            stable_margins.append(margin)
            continue
        token_left = source.get("generated_token_ids")
        token_right = diagnostic.get("generated_token_ids")
        if token_left is not None and token_right is not None:
            divergence = _first_divergence(token_left, token_right)
            first_divergence[str(divergence)] += 1
        else:
            divergence = 0
            first_divergence["0_mcq"] += 1
        changed_margins.append(margin)
        row = {
            "item_id": item_id,
            "battery": source["battery"],
            "source_margin_minimum": source_margin,
            "diagnostic_margin_minimum": diagnostic_margin,
            "first_divergence_position": divergence,
            "source_correct": bool(source["augmented_correct"]),
            "diagnostic_correct": bool(diagnostic["augmented_correct"]),
        }
        changed.append(row)
        if margin > 0.01:
            high_margin_changed.append(row)

    def describe(values: list[float]) -> dict[str, float | int | None]:
        ordered = sorted(values)
        return {
            "rows": len(values),
            "mean": None if not values else sum(values) / len(values),
            "median": None if not values else ordered[len(values) // 2],
            "maximum": None if not values else ordered[-1],
        }

    return {
        "kind": "paper2_stage2b_r1_runtime_discordance_audit_v1",
        "status": "desk_components_complete_fixed_prompt_pending",
        "prediction_changed_rows": len(changed),
        "correctness_gains": sum(not row["source_correct"] and row["diagnostic_correct"] for row in changed),
        "correctness_losses": sum(row["source_correct"] and not row["diagnostic_correct"] for row in changed),
        "changed_row_margin": describe(changed_margins),
        "stable_row_margin": describe(stable_margins),
        "changed_rows_above_0p01_margin": len(high_margin_changed),
        "high_margin_changed_rows": high_margin_changed,
        "first_divergence_histogram": dict(sorted(first_divergence.items())),
        "fixed_prompt_cross_runtime": "pending paired A100/L4 logit receipts",
        "optimizer_constructed": False,
        "training_steps": 0,
    }


def compare_fixed_prompt_logits(
    a100_logits: torch.Tensor, l4_logits: torch.Tensor
) -> dict[str, Any]:
    if a100_logits.shape != l4_logits.shape or a100_logits.ndim != 1:
        raise ValueError("R-1 fixed-prompt logits must share a flat vocabulary shape")
    delta = (a100_logits.float() - l4_logits.float()).abs()
    a100_top = int(a100_logits.argmax())
    l4_top = int(l4_logits.argmax())
    return {
        "kind": "paper2_stage2b_r1_fixed_prompt_comparison_v1",
        "vocabulary": int(delta.numel()),
        "max_absolute_logit_delta": float(delta.max()),
        "mean_absolute_logit_delta": float(delta.mean()),
        "a100_top_token_id": a100_top,
        "l4_top_token_id": l4_top,
        "top_token_identical": a100_top == l4_top,
        "status": "runtime_aligned" if a100_top == l4_top else "runtime_discordant",
    }


def fixed_prompt_comparison_receipt(
    a100_path: str | Path, l4_path: str | Path
) -> dict[str, Any]:
    paths = {"a100_40gb": Path(a100_path), "l4": Path(l4_path)}
    tensors = {
        label: torch.load(path, map_location="cpu", weights_only=True)
        for label, path in paths.items()
    }
    receipt = compare_fixed_prompt_logits(tensors["a100_40gb"], tensors["l4"])
    receipt["source_sha256"] = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in paths.items()
    }
    receipt.update(
        {
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "confirm_scored": False,
            "eval_e_scored": False,
        }
    )
    return receipt


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8", newline="")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0-rows", required=True)
    parser.add_argument("--seed1-rows", required=True)
    parser.add_argument("--r1-source-t3a", required=True)
    parser.add_argument("--r1-diagnostic-t3a", required=True)
    parser.add_argument("--r1-source-t3b", required=True)
    parser.add_argument("--r1-diagnostic-t3b", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, ensemble = seed_ensemble_probe(read_jsonl(args.seed0_rows), read_jsonl(args.seed1_rows))
    ensemble["rows_sha256"] = _write_jsonl(output / "seed_ensemble_rows.jsonl", rows)
    (output / "seed_ensemble_summary.json").write_text(
        json.dumps(ensemble, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    r1 = {
        "kind": "paper2_stage2b_r1_combined_runtime_discordance_v1",
        "t3a": runtime_discordance_audit(
            read_jsonl(args.r1_source_t3a), read_jsonl(args.r1_diagnostic_t3a)
        ),
        "t3b": runtime_discordance_audit(
            read_jsonl(args.r1_source_t3b), read_jsonl(args.r1_diagnostic_t3b)
        ),
    }
    (output / "r1_runtime_discordance_summary.json").write_text(
        json.dumps(r1, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ensemble": ensemble, "r1": r1}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
