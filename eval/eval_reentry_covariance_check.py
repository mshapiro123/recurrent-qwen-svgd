"""Offline covariance gate for recurrent loop re-entry repair.

This read-only diagnostic answers the pre-build question for directional
re-entry correction: after matching global RMS, is the recurrent block exit
distribution close to the recurrent block entry distribution up to an
orthogonal rotation, a general linear covariance map, or neither safely?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_effective_pathways import read_prompts  # noqa: E402
from eval.eval_reentry_drift import load_wrapper, run_prompt, subspace_overlap  # noqa: E402


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


def centered_covariance(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    samples = samples.detach().double().cpu()
    if samples.dim() != 2:
        raise ValueError("samples must be 2D")
    if samples.shape[0] < 2:
        raise ValueError("at least two samples are required")
    mean = samples.mean(dim=0)
    centered = samples - mean
    cov = centered.T @ centered / max(samples.shape[0] - 1, 1)
    return cov, mean


def sorted_eigh(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    evals, evecs = torch.linalg.eigh(matrix.double())
    order = torch.argsort(evals, descending=True)
    return evals[order], evecs[:, order]


def psd_sqrt(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    evals, evecs = sorted_eigh(matrix)
    evals = evals.clamp_min(eps)
    return (evecs * evals.sqrt().unsqueeze(0)) @ evecs.T


def psd_invsqrt(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    evals, evecs = sorted_eigh(matrix)
    evals = evals.clamp_min(eps)
    return (evecs * evals.rsqrt().unsqueeze(0)) @ evecs.T


def relative_frobenius(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.norm(b, ord="fro").clamp_min(1e-12)
    return finite_float(torch.linalg.norm(a - b, ord="fro") / denom)


def trace_match(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_trace = (torch.trace(source) if source.dim() == 2 else source.sum()).clamp_min(1e-12)
    target_trace = (torch.trace(target) if target.dim() == 2 else target.sum()).clamp_min(1e-12)
    return source * (target_trace / source_trace)


def effective_rank(values: torch.Tensor) -> float:
    values = values.detach().double().abs()
    denom = values.square().sum().clamp_min(1e-12)
    return finite_float(values.sum().square() / denom)


def cumulative_dims(values: torch.Tensor, thresholds: tuple[float, ...] = (0.8, 0.9, 0.95)) -> dict[str, int]:
    values = values.detach().double().abs()
    total = values.sum()
    if total <= 0:
        return {str(threshold): 0 for threshold in thresholds}
    cumulative = torch.cumsum(values, dim=0) / total
    out: dict[str, int] = {}
    for threshold in thresholds:
        hits = torch.nonzero(cumulative >= threshold)
        out[str(threshold)] = int(hits[0].item() + 1) if hits.numel() else int(values.numel())
    return out


def pca_basis(samples: torch.Tensor, rank: int) -> torch.Tensor:
    samples = samples.detach().double().cpu()
    centered = samples - samples.mean(dim=0, keepdim=True)
    max_rank = min(int(rank), centered.shape[0], centered.shape[1])
    if max_rank <= 0:
        return centered.new_zeros((centered.shape[1], 0))
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:max_rank].T.contiguous()


def union_pca_basis(entry: torch.Tensor, exit: torch.Tensor, rank: int) -> torch.Tensor:
    entry_basis = pca_basis(entry, rank)
    exit_basis = pca_basis(exit, rank)
    if entry_basis.numel() == 0:
        return exit_basis
    if exit_basis.numel() == 0:
        return entry_basis
    q, _r = torch.linalg.qr(torch.cat([entry_basis, exit_basis], dim=1), mode="reduced")
    max_cols = min(q.shape[1], max(1, 2 * int(rank)))
    return q[:, :max_cols].contiguous()


def covariance_match_check(
    sigma_exit: torch.Tensor,
    sigma_entry: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Return covariance mismatch before/after orthogonal and full linear maps."""

    sy = sigma_entry.detach().double().cpu()
    sx = trace_match(sigma_exit.detach().double().cpu(), sy)
    dim = sy.shape[0]
    eye = torch.eye(dim, dtype=sy.dtype)
    scale = finite_float(torch.trace(sy) / max(dim, 1))
    ridge = max(float(eps) * max(scale, 1.0), float(eps))
    sx_reg = sx + ridge * eye
    sy_reg = sy + ridge * eye

    before = relative_frobenius(sx, sy)
    exit_evals, exit_evecs = sorted_eigh(sx_reg)
    entry_evals, entry_evecs = sorted_eigh(sy_reg)
    rotation = entry_evecs @ exit_evecs.T
    orth_cov = rotation @ sx @ rotation.T
    after_orthogonal = relative_frobenius(orth_cov, sy)

    linear_map = psd_sqrt(sy_reg, ridge) @ psd_invsqrt(sx_reg, ridge)
    linear_cov = linear_map @ sx_reg @ linear_map.T
    after_linear = relative_frobenius(linear_cov, sy_reg)
    singular_values = torch.linalg.svdvals(linear_map)

    spectrum_rel_l2 = finite_float(
        torch.linalg.norm(trace_match(exit_evals, entry_evals) - entry_evals)
        / torch.linalg.norm(entry_evals).clamp_min(1e-12)
    )
    delta_evals = torch.linalg.eigvalsh(sx - sy).abs().sort(descending=True).values
    return {
        "dimension": int(dim),
        "ridge": ridge,
        "before": before,
        "after_orthogonal": after_orthogonal,
        "after_linear": after_linear,
        "orthogonal_reduction": 1.0 - (after_orthogonal / before) if before > 0 else 0.0,
        "linear_reduction": 1.0 - (after_linear / before) if before > 0 else 0.0,
        "spectrum_rel_l2": spectrum_rel_l2,
        "linear_map_singular_min": finite_float(singular_values.min()),
        "linear_map_singular_max": finite_float(singular_values.max()),
        "linear_map_condition": finite_float(singular_values.max() / singular_values.min().clamp_min(1e-12)),
        "delta_effective_rank": effective_rank(delta_evals),
        "delta_cumulative_dims": cumulative_dims(delta_evals),
        "entry_eigenvalues_top16": [finite_float(v) for v in entry_evals[:16]],
        "exit_scaled_eigenvalues_top16": [finite_float(v) for v in trace_match(exit_evals, entry_evals)[:16]],
        "delta_abs_eigenvalues_top16": [finite_float(v) for v in delta_evals[:16]],
    }


def adapter_rank(wrapper: Any) -> int:
    correction = getattr(getattr(wrapper, "reentry_adapter", None), "spectral_correction", None)
    return int(getattr(correction, "rank", 0) or 0)


def recommendation(
    projected: dict[str, Any],
    *,
    current_rank: int,
    subspace_rank: int,
    max_condition: float,
) -> dict[str, Any]:
    before = float(projected["before"])
    orth = float(projected["after_orthogonal"])
    linear = float(projected["after_linear"])
    condition = float(projected["linear_map_condition"])
    spectrum_rel = float(projected["spectrum_rel_l2"])
    orth_reduction = 1.0 - (orth / before) if before > 0 else 0.0
    linear_reduction = 1.0 - (linear / before) if before > 0 else 0.0

    reasons: list[str] = []
    if current_rank < subspace_rank:
        reasons.append(f"current_rank_{current_rank}_below_subspace_rank_{subspace_rank}")
    if before < 0.10:
        action = "no_material_covariance_mismatch"
        reasons.append("covariance_mismatch_small")
    elif orth_reduction >= 0.50 and orth <= 0.50 and spectrum_rel <= 0.25:
        action = "orthogonal_directional_adapter"
        reasons.append("orthogonal_rotation_reduces_mismatch_and_spectra_match")
    elif linear_reduction >= 0.75 and condition <= max_condition:
        action = "general_linear_directional_adapter"
        reasons.append("full_covariance_map_reduces_mismatch_with_bounded_condition")
    elif linear_reduction >= 0.75:
        action = "general_linear_high_condition_needs_review"
        reasons.append("full_covariance_map_reduces_mismatch_but_condition_is_high")
    else:
        action = "do_not_build_linear_adapter"
        reasons.append("covariance_mismatch_not_safely_reduced_by_linear_gate")

    return {
        "action": action,
        "reasons": reasons,
        "current_adapter_rank": current_rank,
        "required_min_rank": int(subspace_rank),
        "rank_sufficient_for_subspace": current_rank >= subspace_rank,
        "orthogonal_reduction": orth_reduction,
        "linear_reduction": linear_reduction,
        "max_allowed_condition": float(max_condition),
    }


def collect_tokens(wrapper: Any, tokenizer: Any, prompts: list[str], args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    records = [
        run_prompt(wrapper, tokenizer, prompt, prompt_index=idx, args=args)
        for idx, prompt in enumerate(prompts)
    ]
    entry_tokens = torch.cat([row["entry_tokens"] for row in records], dim=0)
    exit_tokens = torch.cat([row["exit_tokens"] for row in records], dim=0)
    compact_records = [
        {
            "prompt_index": int(row["prompt_index"]),
            "tokens": int(row["tokens"]),
            "entry_rms": float(row["entry_rms"]),
            "exit_rms": float(row["exit_rms"]),
            "exit_over_entry_rms": float(row["exit_over_entry_rms"]),
            "pooled_entry_exit_cosine": float(row["pooled_entry_exit_cosine"]),
        }
        for row in records
    ]
    return entry_tokens, exit_tokens, compact_records


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    rec = summary["recommendation"]
    proj = summary["projected_covariance_check"]
    full = summary["full_hidden_covariance_check"]
    rank = summary["rank_audit"]
    lines = [
        f"# Re-entry Covariance Check - {summary.get('run_id', '')}",
        "",
        "## Decision",
        f"- Recommended action: `{rec['action']}`",
        f"- Reasons: `{', '.join(rec['reasons'])}`",
        f"- Current adapter rank: `{rank['current_adapter_rank']}`",
        f"- Required minimum rank: `{rank['required_min_rank']}`",
        "",
        "## Projected Union-Subspace Gate",
        f"- Dimension: `{proj['dimension']}`",
        f"- Before residual: `{proj['before']:.6f}`",
        f"- After orthogonal: `{proj['after_orthogonal']:.6f}`",
        f"- After full linear: `{proj['after_linear']:.6f}`",
        f"- Spectrum relative L2: `{proj['spectrum_rel_l2']:.6f}`",
        f"- Linear map condition: `{proj['linear_map_condition']:.6f}`",
        f"- Delta effective rank: `{proj['delta_effective_rank']:.3f}`",
        "",
        "## Full Hidden-Space Check",
        f"- Dimension: `{full['dimension']}`",
        f"- Before residual: `{full['before']:.6f}`",
        f"- After orthogonal: `{full['after_orthogonal']:.6f}`",
        f"- After full linear: `{full['after_linear']:.6f}`",
        f"- Linear map condition: `{full['linear_map_condition']:.6f}`",
        "",
        "## Readout",
        "This is a pre-build gate. Do not implement or train a directional adapter until this output is reviewed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts_jsonl", default="eval/smoke_exact_tasks_v2.jsonl")
    parser.add_argument("--limit", type=int, default=14)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--max_loops", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--subspace_rank", type=int, default=8)
    parser.add_argument("--analysis_rank", type=int, default=8)
    parser.add_argument("--reentry_rescale_mode", default="none", choices=("none", "entry_rms"))
    parser.add_argument("--use_reentry_adapter", action="store_true")
    parser.add_argument("--reentry_adapter_mode", default="affine", choices=("affine", "spectral", "affine_spectral"))
    parser.add_argument("--covariance_eps", type=float, default=1e-6)
    parser.add_argument("--max_condition", type=float, default=16.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_wrapper(args)
    prompts = read_prompts(args.prompts_jsonl or None, limit=args.limit or None)
    entry_tokens, exit_tokens, records = collect_tokens(wrapper, tokenizer, prompts, args)

    entry_cov, entry_mean = centered_covariance(entry_tokens)
    exit_cov, exit_mean = centered_covariance(exit_tokens)
    basis = union_pca_basis(entry_tokens, exit_tokens, args.analysis_rank)
    entry_projected = (entry_tokens.double() - entry_mean) @ basis
    exit_projected = (exit_tokens.double() - exit_mean) @ basis
    entry_proj_cov, _ = centered_covariance(entry_projected)
    exit_proj_cov, _ = centered_covariance(exit_projected)

    projected_check = covariance_match_check(exit_proj_cov, entry_proj_cov, eps=args.covariance_eps)
    full_check = covariance_match_check(exit_cov, entry_cov, eps=args.covariance_eps)
    current_rank = adapter_rank(wrapper)
    rec = recommendation(
        projected_check,
        current_rank=current_rank,
        subspace_rank=args.subspace_rank,
        max_condition=args.max_condition,
    )
    subspace = subspace_overlap(entry_tokens, exit_tokens, rank=args.subspace_rank)
    mean_distance = torch.linalg.norm(exit_mean - entry_mean) / torch.linalg.norm(entry_mean).clamp_min(1e-12)

    summary = {
        "kind": "stage5_reentry_covariance_check",
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "prompts_jsonl": args.prompts_jsonl,
        "limit": args.limit,
        "tokens": int(entry_tokens.shape[0]),
        "hidden_dim": int(entry_tokens.shape[1]),
        "split": args.split,
        "max_length": args.max_length,
        "subspace_rank": args.subspace_rank,
        "analysis_rank": args.analysis_rank,
        "entry_exit_subspace": subspace,
        "mean_relative_distance": finite_float(mean_distance),
        "rank_audit": {
            "current_adapter_rank": current_rank,
            "required_min_rank": int(args.subspace_rank),
            "rank_sufficient_for_subspace": current_rank >= int(args.subspace_rank),
            "projected_delta_effective_rank": projected_check["delta_effective_rank"],
            "projected_delta_cumulative_dims": projected_check["delta_cumulative_dims"],
        },
        "projected_covariance_check": projected_check,
        "full_hidden_covariance_check": full_check,
        "recommendation": rec,
        "records": records,
    }
    print(
        json.dumps(
            {
                "rank_audit": summary["rank_audit"],
                "entry_exit_subspace": summary["entry_exit_subspace"],
                "projected_covariance_check": {
                    key: projected_check[key]
                    for key in (
                        "before",
                        "after_orthogonal",
                        "after_linear",
                        "spectrum_rel_l2",
                        "linear_map_condition",
                        "delta_effective_rank",
                    )
                },
                "recommendation": rec,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
