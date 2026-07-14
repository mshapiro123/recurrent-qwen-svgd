"""Build the durable Phase-A recurrent-versus-dense surpass receipt."""

from __future__ import annotations

import hashlib
import gzip
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.run_stage5_inverse_composition_staircase import _publish
from colab.stage5_phase_a_surpass import surpass_gate


ROOT = Path(os.environ.get("STAGE5_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_A_ROWS = ROOT / "outputs/stage5/stage5_same_reader_final_symbol_20260707_021010/eval/same_reader_final_rows.jsonl"
DEFAULT_DENSE_ROWS = ROOT / "outputs/stage5/stage5_phase_a_checkpoint_comparison_20260713/paired_rows.jsonl"
DEFAULT_DENSE_SUMMARY = ROOT / "outputs/stage5/stage5_phase_a_checkpoint_comparison_20260713/summary.json"
DEFAULT_A_SUMMARY = ROOT / "outputs/stage5/stage5_same_reader_final_symbol_20260707_021010/summary.json"
DEFAULT_DENSE_RAW = {
    label: ROOT
    / "outputs/stage5/stage5_phase_a_checkpoint_comparison_20260713/eval"
    / label
    / "rows.jsonl.gz"
    for label in ("B_step4000", "C_step4000", "D_step4000")
}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_gzip_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _binomial_upper_tail(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    return sum(math.comb(trials, value) for value in range(successes, trials + 1)) / (2**trials)


def paired_binary_test(a_hits: Iterable[bool], b_hits: Iterable[bool]) -> dict[str, Any]:
    pairs = [(bool(a), bool(b)) for a, b in zip(a_hits, b_hits, strict=True)]
    helped = sum(a and not b for a, b in pairs)
    hurt = sum(b and not a for a, b in pairs)
    tied = len(pairs) - helped - hurt
    discordant = helped + hurt
    one_sided = _binomial_upper_tail(helped, discordant)
    lower = sum(math.comb(discordant, value) for value in range(0, min(helped, hurt) + 1)) / (2**discordant) if discordant else 1.0
    two_sided = min(1.0, 2.0 * lower)
    return {
        "helped": helped,
        "hurt": hurt,
        "tied": tied,
        "discordant": discordant,
        "net_correct": helped - hurt,
        "one_sided_p": one_sided,
        "two_sided_p": two_sided,
        "test": "exact_paired_sign_mcnemar",
    }


def _counts_by_depth(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        depth = str(int(row["depth"]))
        counts[depth] = counts.get(depth, 0) + int(bool(row[field]))
    return counts


def row_hash_receipt(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row["id"]))
    ids = "\n".join(str(row["id"]) for row in ordered).encode("utf-8")
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in ordered).encode("utf-8")
    return {
        "rows": len(ordered),
        "row_id_sha256": hashlib.sha256(ids).hexdigest(),
        "row_sha256": hashlib.sha256(payload).hexdigest(),
    }


def summarize_retokenized_decode(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    by_depth: dict[str, list[int]] = {}
    for row in rows:
        token_ids = tokenizer(
            str(row.get("continuation") or ""),
            add_special_tokens=False,
        )["input_ids"]
        by_depth.setdefault(str(int(row["depth"])), []).append(len(token_ids))
    summaries = {}
    for depth, values in sorted(by_depth.items(), key=lambda item: int(item[0])):
        summaries[depth] = {
            "rows": len(values),
            "retokenized_generated_tokens_min": min(values),
            "retokenized_generated_tokens_mean": sum(values) / len(values),
            "retokenized_generated_tokens_max": max(values),
            "fraction_at_generation_cap": sum(value >= max_new_tokens for value in values) / len(values),
            "sequential_decode_rounds_mean": sum(values) / len(values),
            "text_context_growth_tokens_mean": sum(values) / len(values),
        }
    return {
        "rows": len(rows),
        "max_new_tokens": int(max_new_tokens),
        "by_depth": summaries,
        "count_method": "retokenized decoded continuation; special tokens excluded",
    }


def build_compute_ledger(
    a_rows: list[dict[str, Any]],
    dense_rows: dict[str, list[dict[str, Any]]],
    *,
    tokenizers: dict[str, Any],
) -> dict[str, Any]:
    depths = sorted({int(row["depth"]) for row in a_rows})
    recurrent = {
        str(depth): {
            "rows": sum(int(row["depth"]) == depth for row in a_rows),
            "latent_recurrent_transitions": depth,
            "autoregressive_generated_tokens": 0,
            "text_context_growth_tokens": 0,
            "sequential_decode_rounds": 0,
            "candidate_scoring_sequences": 16,
        }
        for depth in depths
    }
    caps = {"B_step4000": 32, "C_step4000": 96, "D_step4000": 32}
    dense = {
        label: summarize_retokenized_decode(
            rows,
            tokenizer=tokenizers[label],
            max_new_tokens=caps[label],
        )
        for label, rows in dense_rows.items()
    }
    return {
        "A": {
            "reader": "same-reader full-symbol candidate scoring at forced loop depth",
            "by_depth": recurrent,
        },
        "dense": dense,
        "comparison_boundary": (
            "Arm A used latent recurrent transitions plus candidate scoring; dense arms used greedy "
            "autoregressive decoding. This ledger is not a FLOP, latency, or matched-compute claim."
        ),
    }


def score_phase_a_rows(
    a_rows: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
    *,
    rows_per_depth: int = 128,
) -> dict[str, Any]:
    a_by_id = {str(row["id"]): row for row in a_rows}
    dense_by_id = {str(row["id"]): row for row in dense_rows}
    if len(a_by_id) != len(a_rows) or len(dense_by_id) != len(dense_rows):
        raise ValueError("Phase A rows must have unique IDs")
    if set(a_by_id) != set(dense_by_id):
        raise ValueError("Arm A and dense comparisons must contain identical row IDs")
    ordered_ids = sorted(a_by_id, key=lambda row_id: (int(a_by_id[row_id]["depth"]), row_id))
    merged: list[dict[str, Any]] = []
    for row_id in ordered_ids:
        a_row = a_by_id[row_id]
        dense_row = dense_by_id[row_id]
        if int(a_row["depth"]) != int(dense_row["depth"]):
            raise ValueError(f"Depth mismatch for row {row_id}")
        merged.append(
            {
                "id": row_id,
                "depth": int(a_row["depth"]),
                "A": bool(a_row["same_reader_final_hit"]),
                **{label: bool(hit) for label, hit in dense_row["correct"].items()},
            }
        )
    depth_totals: dict[str, int] = {}
    for row in merged:
        depth = str(row["depth"])
        depth_totals[depth] = depth_totals.get(depth, 0) + 1
    if any(total != int(rows_per_depth) for total in depth_totals.values()):
        raise ValueError(f"Unexpected per-depth totals: {depth_totals}")

    a_counts = _counts_by_depth(merged, "A")
    comparisons: dict[str, Any] = {}
    for label, role in (
        ("B_step4000", "preregistered_primary"),
        ("C_step4000", "analysis_extension"),
    ):
        if any(label not in row for row in merged):
            raise ValueError(f"Dense rows are missing {label}")
        b_counts = _counts_by_depth(merged, label)
        per_depth = {
            depth: paired_binary_test(
                [row["A"] for row in merged if str(row["depth"]) == depth],
                [row[label] for row in merged if str(row["depth"]) == depth],
            )
            for depth in sorted(depth_totals, key=int)
        }
        comparisons[f"A_vs_{label}"] = {
            "role": role,
            "count_based_fisher": surpass_gate(
                a_counts,
                b_counts,
                rows_per_depth=rows_per_depth,
            ),
            "paired": paired_binary_test(
                [row["A"] for row in merged],
                [row[label] for row in merged],
            ),
            "paired_by_depth": per_depth,
        }
    return {
        "rows": len(merged),
        "depth_totals": depth_totals,
        "counts": {
            label: _counts_by_depth(merged, label)
            for label in ("A", "B_step4000", "C_step4000", "D_step4000")
            if all(label in row for row in merged)
        },
        "comparisons": comparisons,
        "row_ids_match": True,
    }


def sha256_path(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {"path": str(target), "status": "missing"}
    digest = hashlib.sha256()
    files = [target] if target.is_file() else sorted(item for item in target.rglob("*") if item.is_file())
    for item in files:
        if target.is_dir():
            digest.update(item.relative_to(target).as_posix().encode("utf-8"))
            digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "path": str(target),
        "status": "verified",
        "sha256": digest.hexdigest(),
        "files": len(files),
    }


def _write_svg(path: Path, counts: dict[str, dict[str, int]], rows_per_depth: int) -> None:
    width, height = 900, 520
    left, right, top, bottom = 70, 25, 35, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    depths = sorted({int(depth) for values in counts.values() for depth in values})
    colors = {"A": "#0B6E4F", "B_step4000": "#C44536", "C_step4000": "#2667FF", "D_step4000": "#6F4E7C"}
    labels = {"A": "Recurrent 0.5B", "B_step4000": "Dense direct 0.5B", "C_step4000": "Dense scratchpad 0.5B", "D_step4000": "Dense direct 1.5B"}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for tick in range(0, 101, 20):
        y = top + plot_h * (1 - tick / 100)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick}%</text>')
    for depth in depths:
        x = left + plot_w * (depth - depths[0]) / max(1, depths[-1] - depths[0])
        lines.append(f'<text x="{x:.1f}" y="{height - 30}" text-anchor="middle" font-family="Arial" font-size="12">{depth}</text>')
    for index, (label, values) in enumerate(counts.items()):
        points = []
        for depth in depths:
            x = left + plot_w * (depth - depths[0]) / max(1, depths[-1] - depths[0])
            y = top + plot_h * (1 - values[str(depth)] / rows_per_depth)
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[label]
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<line x1="{left + 12 + index * 195}" y1="16" x2="{left + 36 + index * 195}" y2="16" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{left + 42 + index * 195}" y="20" font-family="Arial" font-size="12">{labels[label]}</text>')
    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 6}" text-anchor="middle" font-family="Arial" font-size="14">Composition depth</text>')
    lines.append(f'<text x="16" y="{top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 16 {top + plot_h / 2:.1f})" font-family="Arial" font-size="14">Accuracy</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    run_id = os.environ.get("STAGE5_PHASE_A_RECEIPT_RUN_ID") or time.strftime("stage5_phase_a_surpass_receipt_%Y%m%d_%H%M%S")
    run_dir = ROOT / "outputs/stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    a_rows_path = Path(os.environ.get("STAGE5_PHASE_A_A_ROWS", str(DEFAULT_A_ROWS)))
    dense_rows_path = Path(os.environ.get("STAGE5_PHASE_A_DENSE_ROWS", str(DEFAULT_DENSE_ROWS)))
    a_rows = read_jsonl(a_rows_path)
    dense_rows = read_jsonl(dense_rows_path)
    scoring = score_phase_a_rows(a_rows, dense_rows)
    checkpoint_paths = {
        "A": os.environ.get("STAGE5_PHASE_A_A_CHECKPOINT", "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_support8_dose_arm_20260706_153028/anneal_to_outcome_final/unfrozen_recurrent_step_2000.pt"),
        "B_step4000": os.environ.get("STAGE5_PHASE_A_B_CHECKPOINT", "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_phase_a_dense_full_bc_20260713/B/dense_full_step_4000"),
        "C_step4000": os.environ.get("STAGE5_PHASE_A_C_CHECKPOINT", "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_phase_a_dense_full_bc_20260713/C/dense_full_step_4000"),
        "D_step4000": os.environ.get("STAGE5_PHASE_A_D_CHECKPOINT", "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_phase_a_dense_full_d_20260713/D/dense_full_step_4000"),
    }
    checkpoint_receipts = {label: sha256_path(path) for label, path in checkpoint_paths.items()}
    hashes_complete = all(receipt["status"] == "verified" for receipt in checkpoint_receipts.values())
    compute_ledger: dict[str, Any]
    if hashes_complete:
        from transformers import AutoTokenizer

        tokenizer_05b = AutoTokenizer.from_pretrained(checkpoint_paths["B_step4000"])
        tokenizer_15b = AutoTokenizer.from_pretrained(checkpoint_paths["D_step4000"])
        dense_raw = {label: read_gzip_jsonl(path) for label, path in DEFAULT_DENSE_RAW.items()}
        compute_ledger = build_compute_ledger(
            a_rows,
            dense_raw,
            tokenizers={
                "B_step4000": tokenizer_05b,
                "C_step4000": tokenizer_05b,
                "D_step4000": tokenizer_15b,
            },
        )
    else:
        compute_ledger = {
            "status": "pending_checkpoint_hash_receipts",
            "comparison_boundary": "No compute equivalence claim is made.",
        }
    dense_summary = read_json(DEFAULT_DENSE_SUMMARY)
    a_summary = read_json(DEFAULT_A_SUMMARY)
    payload = {
        "kind": "stage5_phase_a_surpass_receipt",
        "run_id": run_id,
        "status": "finished" if hashes_complete else "analysis_complete_checkpoint_hashes_pending",
        "scoring": scoring,
        "row_receipts": {
            "A": row_hash_receipt(a_rows),
            "dense": row_hash_receipt(dense_rows),
            "row_ids_match": scoring["row_ids_match"],
        },
        "checkpoint_receipts": checkpoint_receipts,
        "compute_ledger": compute_ledger,
        "c_step2000_to_step4000": dense_summary["comparison"]["within_arm"]["C"],
        "sources": {
            "a_summary": str(DEFAULT_A_SUMMARY.relative_to(ROOT)),
            "dense_summary": str(DEFAULT_DENSE_SUMMARY.relative_to(ROOT)),
            "a_rows": str(a_rows_path.relative_to(ROOT)),
            "dense_rows": str(dense_rows_path.relative_to(ROOT)),
            "eval_sha256": dense_summary.get("eval_sha256"),
            "a_source_summary": a_summary.get("source_summary"),
        },
        "claim_boundary": "synthetic task-family system comparison; training lineage and FLOPs are not matched",
        "primary_comparator": "B_step4000",
        "strongest_dense_control": "C_step4000",
        "efficiency_secondary": "C_step2000",
    }
    write_json(run_dir / "summary.json", payload)
    _write_svg(run_dir / "phase_a_depth_profile.svg", scoring["counts"], 128)
    markdown = [
        "# Phase A Surpass Receipt",
        "",
        f"- Status: `{payload['status']}`",
        f"- Rows: `{scoring['rows']}`",
        f"- Preregistered A-over-B gate: `{scoring['comparisons']['A_vs_B_step4000']['count_based_fisher']['pass']}`",
        f"- A-over-C extension: `{scoring['comparisons']['A_vs_C_step4000']['count_based_fisher']['pass']}`",
        "- Claim boundary: synthetic task-family system comparison; no matched-lineage or FLOP claim.",
    ]
    (run_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    if (run_dir / "phase_a_depth_profile.svg").exists():
        subprocess.run(
            ["git", "add", "-f", str((run_dir / "phase_a_depth_profile.svg").relative_to(ROOT))],
            cwd=ROOT,
            check=True,
        )
    _publish(run_dir, f"Record Phase A surpass receipt {run_id} [skip ci]")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if hashes_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
