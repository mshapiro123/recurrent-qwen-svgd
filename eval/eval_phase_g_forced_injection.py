"""Evaluate one preserved Phase G A0 arm under forced posterior-residual scales."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_mcq import load_recurrent_wrapper  # noqa: E402
from eval.eval_phase_g_alpha import (  # noqa: E402
    phase_g_predictions,
    read_jsonl,
    read_resume_cache,
    sha256_file,
)
from training.checkpointing import load_trainable_checkpoint  # noqa: E402
from training.phase_g_alpha_spec import phase_g_active_lineage_hash  # noqa: E402
from training.phase_g_forced_injection_spec import (  # noqa: E402
    LOCKED_CONTROL_GROUPS,
    LOCKED_CONTROL_ROWS,
    LOCKED_INJECTION_FACTORS,
    summarize_factor_rows,
)


def loader_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=args.model_name,
        checkpoint=args.keeper,
        split=args.split,
        bridge_projection_mode="split",
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device=args.device,
        lora_rank=0,
        lora_alpha=16,
        adapter_dtype="float32",
        base_lora_layer_range="all",
    )


def factor_slug(factor: float) -> str:
    return f"x{factor:g}".replace(".", "p")


def parse_factors(raw: str) -> tuple[float, ...]:
    factors = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if factors != LOCKED_INJECTION_FACTORS:
        raise AssertionError(
            "Forced-injection factors are locked to "
            + ",".join(f"{value:g}" for value in LOCKED_INJECTION_FACTORS)
        )
    return factors


def _prediction(row: dict[str, Any]) -> str:
    predictions = list(row.get("predictions") or [])
    if len(predictions) != 1:
        raise AssertionError("Published A0 posterior receipt must contain exactly K=1")
    return str(predictions[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--keeper", required=True)
    parser.add_argument("--expected_keeper_sha256", required=True)
    parser.add_argument("--guidance_checkpoint", required=True)
    parser.add_argument("--expected_guidance_sha256", required=True)
    parser.add_argument("--baseline_posterior_jsonl", required=True)
    parser.add_argument("--rng_manifest_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume_cache_dir", required=True)
    parser.add_argument("--arm_label", required=True)
    parser.add_argument("--factors", default="1,3,10,30,100")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--projection_seed", type=int, default=20260717)
    parser.add_argument("--injection_scale_init", type=float, default=1e-3)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress_every", type=int, default=16)
    args = parser.parse_args()

    factors = parse_factors(args.factors)
    if sha256_file(args.keeper) != args.expected_keeper_sha256:
        raise AssertionError("Forced-injection keeper SHA mismatch")
    guidance_sha256 = sha256_file(args.guidance_checkpoint)
    if guidance_sha256 != args.expected_guidance_sha256:
        raise AssertionError("Forced-injection guidance checkpoint SHA mismatch")

    rows = read_jsonl(args.data_jsonl)
    baseline_rows = read_jsonl(args.baseline_posterior_jsonl)
    rng_rows = read_jsonl(args.rng_manifest_jsonl)
    if len(rows) != LOCKED_CONTROL_ROWS or len(rng_rows) != len(rows):
        raise AssertionError("Forced-injection probe must use the locked 106-row surface")
    if [row["id"] for row in rows] != [row["id"] for row in baseline_rows]:
        raise AssertionError("Published posterior baseline order differs from control rows")
    if [row["id"] for row in rows] != [row["id"] for row in rng_rows]:
        raise AssertionError("Published RNG manifest order differs from control rows")
    if len({str(row["base_problem_id"]) for row in rows}) != LOCKED_CONTROL_GROUPS:
        raise AssertionError("Forced-injection control surface must contain 32 groups")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(loader_args(args), args.keeper)
    lineage_before = phase_g_active_lineage_hash(wrapper.named_parameters())
    wrapper.enable_phase_g_guidance(
        latent_dim=args.latent_dim,
        projection_seed=args.projection_seed,
        injection_scale_init=args.injection_scale_init,
    )
    load_info = load_trainable_checkpoint(wrapper, args.guidance_checkpoint)
    loaded_phase_g = [
        name for name in load_info["loaded_keys"] if str(name).startswith("phase_g_")
    ]
    if not loaded_phase_g:
        raise AssertionError("Forced-injection checkpoint restored no Phase G tensors")
    skipped_phase_g = [
        name for name in load_info["skipped"] if str(name).startswith("phase_g_")
    ]
    if skipped_phase_g:
        raise AssertionError(f"Skipped Phase G tensors: {skipped_phase_g}")
    lineage_loaded = phase_g_active_lineage_hash(wrapper.named_parameters())
    if lineage_loaded != lineage_before:
        raise AssertionError("Loading guidance changed the frozen deterministic lineage")

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.resume_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    factor_summaries: dict[str, Any] = {}
    factor_1_predictions = {
        str(row["id"]): _prediction(baseline)
        for row, baseline in zip(rows, baseline_rows)
    }

    for factor in factors:
        slug = factor_slug(factor)
        cache_path = cache_dir / f"{slug}.jsonl"
        cached = read_resume_cache(cache_path)
        for index, payload in enumerate(cached):
            if index >= len(rows) or payload["id"] != rows[index]["id"]:
                raise AssertionError(f"Resume cache row mismatch for factor {factor:g}")
            if float(payload["injection_multiplier"]) != factor:
                raise AssertionError(f"Resume cache factor mismatch for factor {factor:g}")
            if payload["guidance_sha256"] != guidance_sha256:
                raise AssertionError("Resume cache checkpoint SHA mismatch")

        with cache_path.open("a", encoding="utf-8") as cache_handle:
            for index, (row, rng_row) in enumerate(zip(rows, rng_rows), start=1):
                if index <= len(cached):
                    continue
                if index == 1 or index % args.progress_every == 0 or index == len(rows):
                    print(
                        f"forced_injection_progress arm={args.arm_label} factor={factor:g} "
                        f"row={index}/{len(rows)}",
                        flush=True,
                    )
                predictions, metrics = phase_g_predictions(
                    wrapper,
                    tokenizer,
                    row,
                    k_max=1,
                    seed_base=int(rng_row["seed_base"]),
                    posterior_teacher=True,
                    device=args.device,
                    injection_multiplier=factor,
                )
                prediction = str(predictions[0])
                factor_1_prediction = factor_1_predictions[str(row["id"])]
                if factor == 1.0 and prediction != factor_1_prediction:
                    raise AssertionError(
                        "Factor-1 equivalence failed for "
                        f"{row['id']}: {prediction!r} != {factor_1_prediction!r}"
                    )
                payload = {
                    "id": str(row["id"]),
                    "base_problem_id": str(row["base_problem_id"]),
                    "depth": int(row["depth"]),
                    "target": str(row["target"]),
                    "reachable_symbols": list(map(str, row["reachable_symbols"])),
                    "prediction": prediction,
                    "factor_1_prediction": factor_1_prediction,
                    "seed_base": int(rng_row["seed_base"]),
                    "trajectory_seeds": list(rng_row["trajectory_seeds"]),
                    "injection_multiplier": factor,
                    "guidance_sha256": guidance_sha256,
                    "phase_g_metrics": metrics,
                }
                cache_handle.write(json.dumps(payload, sort_keys=True) + "\n")
                cache_handle.flush()
                cached.append(payload)

        summary = summarize_factor_rows(cached)
        summary["injection_multiplier"] = factor
        summary["factor_1_exact_equivalence"] = (
            all(row["prediction"] == row["factor_1_prediction"] for row in cached)
            if factor == 1.0
            else None
        )
        factor_summaries[f"{factor:g}"] = summary
        (output_dir / f"{slug}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in cached),
            encoding="utf-8",
        )
        (output_dir / f"{slug}_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"arm": args.arm_label, "factor": factor, **summary}, sort_keys=True))

    lineage_after = phase_g_active_lineage_hash(wrapper.named_parameters())
    if lineage_after != lineage_before:
        raise AssertionError("Forced-injection evaluation mutated the frozen keeper lineage")
    result = {
        "kind": "phase_g_forced_injection_arm",
        "status": "finished",
        "label": args.arm_label,
        "keeper": args.keeper,
        "keeper_sha256": args.expected_keeper_sha256,
        "guidance_checkpoint": args.guidance_checkpoint,
        "guidance_sha256": guidance_sha256,
        "data_jsonl": args.data_jsonl,
        "baseline_posterior_jsonl": args.baseline_posterior_jsonl,
        "rng_manifest_jsonl": args.rng_manifest_jsonl,
        "factors": factor_summaries,
        "factor_1_exact_equivalence": bool(
            factor_summaries["1"]["factor_1_exact_equivalence"]
        ),
        "frozen_lineage_before": lineage_before,
        "frozen_lineage_after": lineage_after,
        "frozen_lineage_unchanged": lineage_before == lineage_after,
        "loaded_phase_g_keys": loaded_phase_g,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

