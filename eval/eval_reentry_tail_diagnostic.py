"""Tail-resolved recurrent re-entry diagnostic.

This read-only diagnostic follows the covariance pre-build gate. It removes the
dominant entry covariance axis, decomposes the lower-variance tail mismatch
into damping-vs-rotation components, and joins loop-tail dynamics to forced
depth harmed/rescued examples.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import MCQExample, format_prompt, read_examples  # noqa: E402
from eval.eval_reentry_drift import (  # noqa: E402
    load_wrapper,
    masked_token_matrix,
    prepare_recurrent_inputs,
    rms,
    run_recurrent_block,
)


@dataclass(frozen=True)
class PromptRow:
    id: str
    prompt: str
    group: str = "ungrouped"
    pattern: str = ""
    tipping_loop: int | None = None


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def finite_float(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        value = float(value.detach().double().cpu())
    value = float(value)
    if math.isfinite(value):
        return value
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def centered_covariance(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    samples = samples.detach().double().cpu()
    mean_vec = samples.mean(dim=0)
    centered = samples - mean_vec
    cov = centered.T @ centered / max(samples.shape[0] - 1, 1)
    return cov, mean_vec


def eig_desc(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    evals, evecs = torch.linalg.eigh(matrix.double())
    order = torch.argsort(evals, descending=True)
    return evals[order], evecs[:, order]


def relative_frobenius(a: torch.Tensor, b: torch.Tensor) -> float:
    return finite_float(torch.linalg.norm(a - b, ord="fro") / torch.linalg.norm(b, ord="fro").clamp_min(1e-12))


def offdiag(matrix: torch.Tensor) -> torch.Tensor:
    return matrix - torch.diag(torch.diag(matrix))


def tail_decomposition(
    sigma_entry: torch.Tensor,
    sigma_exit: torch.Tensor,
    *,
    n_tail: int = 7,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Project out PC1 and split the tail mismatch into damper vs rotation."""

    entry_evals, entry_evecs = eig_desc(sigma_entry)
    dim = sigma_entry.shape[0]
    n_tail = min(max(int(n_tail), 1), dim - 1)
    basis = entry_evecs[:, 1 : 1 + n_tail]
    ce = basis.T @ sigma_entry.double() @ basis
    cx = basis.T @ sigma_exit.double() @ basis
    tail_mismatch = relative_frobenius(cx, ce)

    diag_entry = torch.diag(ce).clamp_min(eps)
    diag_exit = torch.diag(cx).clamp_min(eps)
    damper = (diag_entry / diag_exit).sqrt()
    damped = damper[:, None] * cx * damper[None, :]
    after_damper = relative_frobenius(damped, ce)

    _exit_tail_evals, exit_tail_evecs = eig_desc(cx)
    _entry_tail_evals, entry_tail_evecs = eig_desc(ce)
    rotation = entry_tail_evecs @ exit_tail_evecs.T
    rotated = rotation @ cx @ rotation.T
    after_rotation = relative_frobenius(rotated, ce)
    rotated_then_damped = damper[:, None] * rotated * damper[None, :]
    after_rotation_damper = relative_frobenius(rotated_then_damped, ce)

    ce_norm = torch.linalg.norm(ce, ord="fro").clamp_min(1e-12)
    return {
        "n_tail": int(n_tail),
        "tail_mismatch": tail_mismatch,
        "after_damper": after_damper,
        "after_rotation": after_rotation,
        "after_rotation_then_damper": after_rotation_damper,
        "damper_reduction": 1.0 - (after_damper / tail_mismatch) if tail_mismatch > 0 else 0.0,
        "rotation_reduction": 1.0 - (after_rotation / tail_mismatch) if tail_mismatch > 0 else 0.0,
        "both_reduction": 1.0 - (after_rotation_damper / tail_mismatch) if tail_mismatch > 0 else 0.0,
        "offdiag_before": finite_float(torch.linalg.norm(offdiag(cx), ord="fro") / ce_norm),
        "offdiag_after_damper": finite_float(torch.linalg.norm(offdiag(damped), ord="fro") / ce_norm),
        "entry_tail_eigenvalues": [finite_float(v) for v in torch.diag(ce)],
        "exit_tail_diagonal": [finite_float(v) for v in torch.diag(cx)],
        "exit_over_entry_diag": [finite_float(diag_exit[i] / diag_entry[i]) for i in range(n_tail)],
        "damper_scale": [finite_float(v) for v in damper],
    }


def correction_class(decomp: dict[str, Any]) -> dict[str, Any]:
    mismatch = float(decomp["tail_mismatch"])
    damper = float(decomp["after_damper"])
    rotation = float(decomp["after_rotation"])
    both = float(decomp["after_rotation_then_damper"])
    damper_reduction = 1.0 - damper / mismatch if mismatch > 0 else 0.0
    rotation_reduction = 1.0 - rotation / mismatch if mismatch > 0 else 0.0
    both_reduction = 1.0 - both / mismatch if mismatch > 0 else 0.0
    if mismatch < 0.10:
        action = "no_tail_correction"
        reasons = ["tail_mismatch_small"]
    elif damper_reduction >= 0.50 and rotation_reduction < 0.50:
        action = "tail_damper"
        reasons = ["per_axis_tail_damping_removes_most_mismatch"]
    elif rotation_reduction >= 0.50 and damper_reduction < 0.50:
        action = "tail_rotation"
        reasons = ["orthogonal_tail_rotation_removes_most_mismatch"]
    elif both_reduction >= 0.65:
        action = "tail_damper_plus_rotation"
        reasons = ["combined_tail_damping_and_rotation_needed"]
    else:
        action = "tail_mismatch_needs_review"
        reasons = ["tail_mismatch_not_cleanly_explained_by_damper_or_rotation"]
    return {
        "action": action,
        "reasons": reasons,
        "tail_mismatch": mismatch,
        "damper_reduction": damper_reduction,
        "rotation_reduction": rotation_reduction,
        "both_reduction": both_reduction,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def forced_depth_patterns(summary_path: Path, *, benchmark: str, score_target: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    run_ids = [str(item) for item in payload.get("loop_run_ids", [])]
    loops: list[int] = [int(item) for item in payload.get("loops", [])]
    if not run_ids or not loops or len(run_ids) != len(loops):
        raise ValueError(f"Cannot resolve loop run ids from {summary_path}")

    by_id: dict[str, dict[int, bool]] = defaultdict(dict)
    answers: dict[str, str] = {}
    predictions: dict[str, dict[int, str]] = defaultdict(dict)
    for run_id, loop in zip(run_ids, loops):
        path = ROOT / "outputs" / "stage5" / run_id / f"{benchmark}_recurrent_{score_target}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            row_id = str(row["id"])
            by_id[row_id][loop] = bool(row.get("hit"))
            answers[row_id] = str(row.get("answer", ""))
            predictions[row_id][loop] = str(row.get("prediction", ""))

    out: dict[str, dict[str, Any]] = {}
    for row_id, hits in by_id.items():
        if any(loop not in hits for loop in loops):
            continue
        pattern = "".join("1" if hits[loop] else "0" for loop in loops)
        if pattern[0] == "1" and "0" in pattern[1:]:
            group = "harmed"
            tipping = next((loop for loop in loops[1:] if not hits[loop]), None)
        elif pattern[0] == "0" and "1" in pattern[1:]:
            group = "rescued"
            tipping = next((loop for loop in loops[1:] if hits[loop]), None)
        elif set(pattern) == {"1"}:
            group = "stable_correct"
            tipping = None
        elif set(pattern) == {"0"}:
            group = "stable_wrong"
            tipping = None
        else:
            group = "mixed_other"
            tipping = None
        out[row_id] = {
            "id": row_id,
            "hits": {str(loop): bool(hits[loop]) for loop in loops},
            "loops": loops,
            "pattern": pattern,
            "group": group,
            "tipping_loop": tipping,
            "answer": answers.get(row_id, ""),
            "predictions": {str(loop): predictions[row_id].get(loop, "") for loop in loops},
        }
    return out


def arc_examples(*, config: str, split: str, offset: int, limit: int, seed: int) -> list[MCQExample]:
    from datasets import load_dataset

    from eval.prepare_arc_mcq import row_to_mcq

    dataset = load_dataset("allenai/ai2_arc", config, split=split)
    start = max(int(offset), 0)
    end = len(dataset) if limit <= 0 else min(start + int(limit), len(dataset))
    rows = []
    for idx in range(start, end):
        prepared = row_to_mcq(dict(dataset[idx]), index=idx, seed=seed, shuffle_choices=True)
        rows.append(prepared)
    tmp = ROOT / "outputs" / "stage5" / "_tmp_tail_arc_examples.jsonl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    try:
        return read_examples(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def prompt_rows(args: argparse.Namespace) -> list[PromptRow]:
    patterns = (
        forced_depth_patterns(
            Path(args.forced_depth_summary),
            benchmark=args.benchmark,
            score_target=args.score_target,
        )
        if args.forced_depth_summary
        else {}
    )
    if args.mcq_jsonl:
        examples = read_examples(args.mcq_jsonl)
    else:
        examples = arc_examples(
            config=args.arc_config,
            split=args.arc_split,
            offset=args.arc_offset,
            limit=args.arc_limit,
            seed=args.arc_seed,
        )
    rows = []
    for ex in examples:
        meta = patterns.get(ex.id, {})
        if patterns and ex.id not in patterns:
            continue
        rows.append(
            PromptRow(
                id=ex.id,
                prompt=format_prompt(ex, args.prompt_style),
                group=str(meta.get("group", "ungrouped")),
                pattern=str(meta.get("pattern", "")),
                tipping_loop=meta.get("tipping_loop"),
            )
        )
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No prompt rows available for tail diagnostic")
    return rows


def collect_loop_tokens(wrapper: Any, tokenizer: Any, rows: list[PromptRow], args: argparse.Namespace) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    loop_set = sorted({int(item) for item in args.loop_counts})
    max_loop = max(loop_set)
    tokens_by_stage: dict[str, list[torch.Tensor]] = {"entry": []}
    for loop in loop_set:
        tokens_by_stage[f"loop{loop}"] = []
    compact_records: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        encoded = tokenizer(row.prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        with torch.no_grad():
            entry_state, mask, causal_mask, position_ids, cache_position, position_embeddings = prepare_recurrent_inputs(
                wrapper,
                input_ids,
                attention_mask,
            )
            entry_rms = rms(entry_state, mask).clamp_min(1e-8)
            recurrent_state = entry_state
            tokens_by_stage["entry"].append(masked_token_matrix(entry_state, mask).cpu())
            loop_rms: dict[str, float] = {}
            for loop_idx in range(max_loop):
                loop_number = loop_idx + 1
                loop_input = recurrent_state if loop_idx == 0 else wrapper.bridge(recurrent_state)
                if loop_idx > 0 and args.reentry_rescale_mode == "entry_rms":
                    current_rms = rms(loop_input, mask).clamp_min(1e-8)
                    loop_input = loop_input * (entry_rms / current_rms).to(dtype=loop_input.dtype)
                if loop_idx > 0 and args.use_reentry_adapter:
                    loop_input = wrapper.reentry_adapter(loop_input, loop_idx=loop_idx, mode=args.reentry_adapter_mode)
                loop_output = run_recurrent_block(
                    wrapper,
                    loop_input,
                    causal_mask,
                    position_ids,
                    cache_position,
                    position_embeddings,
                )
                if loop_number in loop_set:
                    tokens_by_stage[f"loop{loop_number}"].append(masked_token_matrix(loop_output, mask).cpu())
                    loop_rms[str(loop_number)] = finite_float(rms(loop_output, mask) / entry_rms)
                recurrent_state = loop_output
        compact_records.append(
            {
                "id": row.id,
                "prompt_index": idx,
                "tokens": int(attention_mask.sum().item()),
                "group": row.group,
                "pattern": row.pattern,
                "tipping_loop": row.tipping_loop,
                "loop_output_over_entry_rms": loop_rms,
            }
        )
    return {key: torch.cat(value, dim=0) for key, value in tokens_by_stage.items()}, compact_records


def tail_basis(entry_tokens: torch.Tensor, n_tail: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma_entry, mean_entry = centered_covariance(entry_tokens)
    evals, evecs = eig_desc(sigma_entry)
    n_tail = min(max(int(n_tail), 1), entry_tokens.shape[1] - 1)
    return sigma_entry, mean_entry, evecs[:, 1 : 1 + n_tail], evals


def projected_cov(tokens: torch.Tensor, mean_entry: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    projected = (tokens.double().cpu() - mean_entry) @ basis
    cov, _ = centered_covariance(projected)
    return cov


def tail_trace(tokens: torch.Tensor, mean_entry: torch.Tensor, basis: torch.Tensor) -> float:
    cov = projected_cov(tokens, mean_entry, basis)
    return finite_float(torch.trace(cov))


def per_record_tail_metrics(
    records: list[dict[str, Any]],
    rows: list[PromptRow],
    wrapper: Any,
    tokenizer: Any,
    mean_entry: torch.Tensor,
    basis: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Compute per-prompt tail energy ratios for harmed/rescued grouping."""

    loop_set = sorted({int(item) for item in args.loop_counts})
    max_loop = max(loop_set)
    out: list[dict[str, Any]] = []
    for row, base_record in zip(rows, records):
        encoded = tokenizer(row.prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        with torch.no_grad():
            entry_state, mask, causal_mask, position_ids, cache_position, position_embeddings = prepare_recurrent_inputs(
                wrapper,
                input_ids,
                attention_mask,
            )
            entry_tokens = masked_token_matrix(entry_state, mask).cpu()
            entry_projected = (entry_tokens.double() - mean_entry) @ basis
            entry_energy = finite_float(entry_projected.square().sum(dim=-1).mean())
            recurrent_state = entry_state
            loop_energy: dict[str, float] = {}
            entry_rms = rms(entry_state, mask).clamp_min(1e-8)
            for loop_idx in range(max_loop):
                loop_number = loop_idx + 1
                loop_input = recurrent_state if loop_idx == 0 else wrapper.bridge(recurrent_state)
                if loop_idx > 0 and args.reentry_rescale_mode == "entry_rms":
                    current_rms = rms(loop_input, mask).clamp_min(1e-8)
                    loop_input = loop_input * (entry_rms / current_rms).to(dtype=loop_input.dtype)
                if loop_idx > 0 and args.use_reentry_adapter:
                    loop_input = wrapper.reentry_adapter(loop_input, loop_idx=loop_idx, mode=args.reentry_adapter_mode)
                loop_output = run_recurrent_block(
                    wrapper,
                    loop_input,
                    causal_mask,
                    position_ids,
                    cache_position,
                    position_embeddings,
                )
                if loop_number in loop_set:
                    loop_tokens = masked_token_matrix(loop_output, mask).cpu()
                    loop_projected = (loop_tokens.double() - mean_entry) @ basis
                    loop_energy[str(loop_number)] = finite_float(loop_projected.square().sum(dim=-1).mean())
                recurrent_state = loop_output
        ratios = {
            loop: value / max(entry_energy, 1e-12)
            for loop, value in loop_energy.items()
        }
        tipping_loop = base_record.get("tipping_loop")
        tipping_ratio = ratios.get(str(tipping_loop)) if tipping_loop is not None else None
        out.append(
            {
                **base_record,
                "entry_tail_energy": entry_energy,
                "loop_tail_energy": loop_energy,
                "loop_tail_energy_ratio": ratios,
                "tipping_tail_energy_ratio": tipping_ratio,
            }
        )
    return out


def group_summary(records: list[dict[str, Any]], loop_counts: list[int]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_group[str(row.get("group", "ungrouped"))].append(row)
    out: dict[str, Any] = {}
    for group, rows in sorted(by_group.items()):
        tipping_values = [
            float(row["tipping_tail_energy_ratio"])
            for row in rows
            if row.get("tipping_tail_energy_ratio") is not None
        ]
        out[group] = {
            "n": len(rows),
            "patterns": dict(Counter(str(row.get("pattern", "")) for row in rows)),
            "mean_tipping_tail_energy_ratio": mean(tipping_values),
            "by_loop_mean_tail_energy_ratio": {
                str(loop): mean([
                    float((row.get("loop_tail_energy_ratio") or {}).get(str(loop)))
                    for row in rows
                    if (row.get("loop_tail_energy_ratio") or {}).get(str(loop)) is not None
                ])
                for loop in loop_counts
            },
        }
    harmed = out.get("harmed", {})
    rescued = out.get("rescued", {})
    out["harmed_minus_rescued"] = {
        "mean_tipping_tail_energy_ratio_delta": float(harmed.get("mean_tipping_tail_energy_ratio", 0.0))
        - float(rescued.get("mean_tipping_tail_energy_ratio", 0.0)),
        "by_loop_delta": {
            str(loop): float((harmed.get("by_loop_mean_tail_energy_ratio") or {}).get(str(loop), 0.0))
            - float((rescued.get("by_loop_mean_tail_energy_ratio") or {}).get(str(loop), 0.0))
            for loop in loop_counts
        },
    }
    return out


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    decomp = summary["tail_decomposition_loop1"]
    rec = summary["correction_class"]
    groups = summary.get("group_summary", {})
    lines = [
        f"# Tail-Resolved Re-entry Diagnostic - {summary.get('run_id', '')}",
        "",
        "## Decision",
        f"- Correction class: `{rec['action']}`",
        f"- Reasons: `{', '.join(rec['reasons'])}`",
        "",
        "## Tail Decomposition",
        f"- Tail mismatch: `{decomp['tail_mismatch']:.6f}`",
        f"- After damper: `{decomp['after_damper']:.6f}`",
        f"- After rotation: `{decomp['after_rotation']:.6f}`",
        f"- After rotation then damper: `{decomp['after_rotation_then_damper']:.6f}`",
        f"- Exit/entry diagonal ratios: `{decomp['exit_over_entry_diag']}`",
        "",
        "## Loop Tail Trace",
        "| stage | tail trace | ratio vs entry |",
        "|---|---:|---:|",
    ]
    for stage, row in summary["loop_tail_trace"].items():
        lines.append(f"| {stage} | {row['tail_trace']:.6f} | {row['ratio_vs_entry']:.6f} |")
    lines.extend(["", "## Harmed vs Rescued"])
    for group in ("harmed", "rescued", "stable_correct", "stable_wrong"):
        row = groups.get(group)
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{group}`: n={row.get('n')}, mean tipping tail ratio={row.get('mean_tipping_tail_energy_ratio')}"
        )
    delta = groups.get("harmed_minus_rescued", {})
    lines.append(f"- Harmed minus rescued tipping ratio delta: `{delta.get('mean_tipping_tail_energy_ratio_delta')}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_damper_artifact(
    *,
    path: Path,
    mean_entry: torch.Tensor,
    basis: torch.Tensor,
    decomp: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """Persist the calibrated tail basis and per-axis damping scales."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "reentry_tail_damper",
            "source_kind": summary.get("kind"),
            "source_run_id": summary.get("run_id"),
            "checkpoint": summary.get("checkpoint"),
            "benchmark": summary.get("benchmark"),
            "score_target": summary.get("score_target"),
            "n_tail": int(summary.get("n_tail") or basis.shape[1]),
            "hidden_dim": int(summary.get("hidden_dim") or basis.shape[0]),
            "mean": mean_entry.detach().float().cpu(),
            "basis": basis.detach().float().cpu(),
            "damper_scale": torch.tensor(decomp["damper_scale"], dtype=torch.float32),
            "exit_over_entry_diag": torch.tensor(decomp["exit_over_entry_diag"], dtype=torch.float32),
            "tail_decomposition_loop1": decomp,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--forced_depth_summary", default="")
    parser.add_argument("--benchmark", default="arc_challenge")
    parser.add_argument("--score_target", default="content_question_only")
    parser.add_argument("--mcq_jsonl", default="")
    parser.add_argument("--arc_config", default="ARC-Challenge")
    parser.add_argument("--arc_split", default="validation")
    parser.add_argument("--arc_offset", type=int, default=0)
    parser.add_argument("--arc_limit", type=int, default=256)
    parser.add_argument("--arc_seed", type=int, default=0)
    parser.add_argument("--prompt_style", default="question_only", choices=("question_only", "with_options"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--loop_counts", default="1,2,3,4,8")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--n_tail", type=int, default=7)
    parser.add_argument("--reentry_rescale_mode", default="none", choices=("none", "entry_rms"))
    parser.add_argument("--use_reentry_adapter", action="store_true")
    parser.add_argument("--reentry_adapter_mode", default="affine", choices=("affine", "spectral", "affine_spectral"))
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    parser.add_argument("--output_damper", default="")
    args = parser.parse_args()
    args.loop_counts = [int(item) for item in str(args.loop_counts).split(",") if item.strip()]
    return args


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows = prompt_rows(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_wrapper(args)
    tokens_by_stage, compact_records = collect_loop_tokens(wrapper, tokenizer, rows, args)
    sigma_entry, mean_entry, basis, entry_evals = tail_basis(tokens_by_stage["entry"], args.n_tail)
    loop1_cov = centered_covariance(tokens_by_stage["loop1"])[0]
    decomp = tail_decomposition(sigma_entry, loop1_cov, n_tail=args.n_tail)
    rec = correction_class(decomp)

    entry_tail = tail_trace(tokens_by_stage["entry"], mean_entry, basis)
    loop_tail_trace = {
        "entry": {"tail_trace": entry_tail, "ratio_vs_entry": 1.0},
    }
    for loop in args.loop_counts:
        trace = tail_trace(tokens_by_stage[f"loop{loop}"], mean_entry, basis)
        loop_tail_trace[f"loop{loop}"] = {
            "tail_trace": trace,
            "ratio_vs_entry": trace / max(entry_tail, 1e-12),
        }

    detailed_records = per_record_tail_metrics(
        compact_records,
        rows,
        wrapper,
        tokenizer,
        mean_entry,
        basis,
        args,
    )
    groups = group_summary(detailed_records, args.loop_counts)
    summary = {
        "kind": "stage5_reentry_tail_diagnostic",
        "run_id": Path(args.output_json).parent.name if args.output_json else None,
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "forced_depth_summary": args.forced_depth_summary,
        "benchmark": args.benchmark,
        "score_target": args.score_target,
        "examples": len(rows),
        "tokens": int(tokens_by_stage["entry"].shape[0]),
        "hidden_dim": int(tokens_by_stage["entry"].shape[1]),
        "n_tail": args.n_tail,
        "loop_counts": args.loop_counts,
        "dominant_entry_eigenvalue": finite_float(entry_evals[0]),
        "entry_eigenvalues_top16": [finite_float(v) for v in entry_evals[:16]],
        "tail_decomposition_loop1": decomp,
        "correction_class": rec,
        "loop_tail_trace": loop_tail_trace,
        "group_summary": groups,
        "records": detailed_records,
    }
    print(
        json.dumps(
            {
                "examples": summary["examples"],
                "tokens": summary["tokens"],
                "tail_decomposition_loop1": decomp,
                "correction_class": rec,
                "loop_tail_trace": loop_tail_trace,
                "group_summary": groups,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved_summary={path_for_cli(path)}", flush=True)
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(summary, path)
        print(f"saved_markdown={path_for_cli(path)}", flush=True)
    if args.output_damper:
        path = Path(args.output_damper)
        write_damper_artifact(
            path=path,
            mean_entry=mean_entry,
            basis=basis,
            decomp=decomp,
            summary=summary,
        )
        print(f"saved_tail_damper={path_for_cli(path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
