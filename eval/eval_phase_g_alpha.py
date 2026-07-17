"""Evaluate Phase G-alpha coverage without pooling trajectory logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from eval.eval_synthetic_depth_active_labels import (  # noqa: E402
    candidates_for_row,
    prompt_for_row,
    single_token_candidate_ids,
)
from eval.phase_g_branching import (  # noqa: E402
    exact_branching_coverage,
    solve_global_temperature,
    summarize_coverage_rows,
)
from training.checkpointing import load_trainable_checkpoint  # noqa: E402
from training.phase_g_alpha_spec import phase_g_active_lineage_hash  # noqa: E402


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_resume_cache(path: str | Path) -> list[dict[str, Any]]:
    """Read complete cache rows and discard only a torn final write."""

    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            print(f"discarding_torn_phase_g_cache_line={path}", flush=True)
    if len(rows) != len([line for line in lines if line.strip()]):
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    return rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def paired_sign_test(left: list[float], right: list[float]) -> dict[str, Any]:
    helped = sum(a > b for a, b in zip(left, right))
    hurt = sum(a < b for a, b in zip(left, right))
    n = helped + hurt
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, value) for value in range(helped, n + 1)) / (2**n)
        p_value = min(1.0, tail)
    return {"helped": helped, "hurt": hurt, "tied": len(left) - n, "one_sided_p": p_value}


def target_embeddings(
    wrapper: Any,
    row: dict[str, Any],
    candidate_ids: dict[str, int],
    *,
    device: str,
) -> torch.Tensor:
    chain = list(row["sampled_chain"])[1:]
    token_ids = torch.tensor(
        [[candidate_ids[str(symbol)] for symbol in chain]],
        device=device,
        dtype=torch.long,
    )
    with torch.no_grad():
        return wrapper.base_model.get_input_embeddings()(token_ids).detach()


def phase_g_predictions(
    wrapper: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    k_max: int,
    seed_base: int,
    posterior_teacher: bool,
    device: str,
) -> tuple[list[str], dict[str, float]]:
    prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")
    candidates = candidates_for_row(row, prediction_space="full_symbols", value_prefix="name:")
    candidate_ids = single_token_candidate_ids(tokenizer, prompt, candidates)
    if candidate_ids is None:
        raise RuntimeError(
            f"Phase G requires the frozen N20 one-token reader; row {row['id']} violated it"
        )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    seeds = [seed_base + trajectory for trajectory in range(k_max)]
    posterior_targets = (
        target_embeddings(wrapper, row, candidate_ids, device=device)
        if posterior_teacher
        else None
    )
    wrapper.train(mode=posterior_teacher)
    with torch.no_grad():
        output = wrapper(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=None,
            max_loops=int(row["depth"]),
            num_trajectories=k_max,
            particle_update_mode="none",
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
            logits_to_keep=1,
            phase_g_enabled=True,
            phase_g_use_posterior=posterior_teacher,
            phase_g_posterior_targets=posterior_targets,
            phase_g_trajectory_seeds=seeds,
        )
    if output.loop_logits is None or output.loop_logits.dim() != 5:
        raise RuntimeError("Phase G evaluation requires [batch,K,loop,seq,vocab] loop logits")
    forced = output.loop_logits[0, :, int(row["depth"]) - 1, -1, :]
    names = list(candidate_ids)
    ids = torch.tensor([candidate_ids[name] for name in names], device=forced.device)
    selected = forced.index_select(dim=-1, index=ids).argmax(dim=-1).tolist()
    predictions = [names[int(index)] for index in selected]
    metrics = {
        name: float(value.detach().float().cpu().item())
        for name, value in output.metrics.items()
        if value.numel() == 1 and name.startswith("phase_g_")
    }
    return predictions, metrics


def deterministic_iso_predictions(
    wrapper: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    sample_counts: list[int],
    device: str,
) -> dict[int, str]:
    prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")
    candidates = candidates_for_row(row, prediction_space="full_symbols", value_prefix="name:")
    candidate_ids = single_token_candidate_ids(tokenizer, prompt, candidates)
    if candidate_ids is None:
        raise RuntimeError(f"Iso-compute reader is not one-token on row {row['id']}")
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    loop_counts = [int(row["depth"]) * count for count in sample_counts]
    wrapper.eval()
    with torch.no_grad():
        output = wrapper(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=None,
            max_loops=max(loop_counts),
            num_trajectories=1,
            particle_update_mode="none",
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
            logits_to_keep=1,
            phase_g_enabled=False,
        )
    if output.loop_logits is None or output.loop_logits.shape[2] != max(loop_counts):
        raise RuntimeError("Iso-compute forced loop request was not honored")
    names = list(candidate_ids)
    ids = torch.tensor([candidate_ids[name] for name in names], device=args_device(output))
    predictions: dict[int, str] = {}
    for count, loop in zip(sample_counts, loop_counts):
        forced = output.loop_logits[0, 0, loop - 1, -1, :]
        selected = int(forced.index_select(dim=-1, index=ids).argmax().item())
        predictions[count] = names[selected]
    return predictions


def args_device(output: Any) -> torch.device:
    return output.loop_logits.device


def sample_temperature_predictions(
    scores: dict[str, float],
    *,
    temperature: float,
    k_max: int,
    seed: int,
) -> list[str]:
    names = list(scores)
    probabilities = torch.softmax(
        torch.tensor([scores[name] for name in names], dtype=torch.float64) / temperature,
        dim=-1,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.multinomial(
        probabilities,
        num_samples=k_max,
        replacement=True,
        generator=generator,
    ).tolist()
    return [names[int(index)] for index in indices]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--deterministic_rows_jsonl", required=True)
    parser.add_argument("--keeper", required=True)
    parser.add_argument("--expected_keeper_sha256", required=True)
    parser.add_argument("--guidance_checkpoint")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume_cache_path")
    parser.add_argument("--sample_counts", default="1,2,4,8,20")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--target_entropy", type=float, default=0.1432)
    parser.add_argument("--coverage_margin", type=float, default=0.05)
    parser.add_argument("--k1_pooled_tolerance", type=float, default=0.03)
    parser.add_argument("--k1_cell_tolerance", type=float, default=0.08)
    parser.add_argument("--expected_k1_correct", type=int, default=389)
    parser.add_argument("--include_temperature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_iso_compute", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include_posterior_teacher", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--projection_seed", type=int, default=20260717)
    parser.add_argument("--injection_scale_init", type=float, default=1e-3)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress_every", type=int, default=16)
    args = parser.parse_args()

    sample_counts = sorted({int(value) for value in args.sample_counts.split(",")})
    if sample_counts[0] != 1:
        raise ValueError("Phase G sample_counts must include K=1")
    if sha256_file(args.keeper) != args.expected_keeper_sha256:
        raise RuntimeError("Keeper SHA mismatch")
    rows = read_jsonl(args.data_jsonl)
    deterministic = {row["id"]: row for row in read_jsonl(args.deterministic_rows_jsonl)}
    if [row["id"] for row in rows] != list(deterministic):
        raise RuntimeError("Phase G rows do not match the frozen deterministic row order")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(loader_args(args), args.keeper)
    lineage_before = phase_g_active_lineage_hash(wrapper.named_parameters())
    wrapper.enable_phase_g_guidance(
        latent_dim=args.latent_dim,
        projection_seed=args.projection_seed,
        injection_scale_init=args.injection_scale_init,
    )
    if args.guidance_checkpoint:
        load_info = load_trainable_checkpoint(wrapper, args.guidance_checkpoint)
        loaded_phase_g = [name for name in load_info["loaded_keys"] if name.startswith("phase_g_")]
        if not loaded_phase_g:
            raise RuntimeError("Guidance checkpoint restored no Phase G tensors")
    lineage_after = phase_g_active_lineage_hash(wrapper.named_parameters())
    if lineage_before != lineage_after:
        raise AssertionError("Installing Phase G guidance changed the frozen keeper lineage")

    temperature = None
    if args.include_temperature:
        temperature = solve_global_temperature(
            [dict(deterministic[row["id"]]["scores"]) for row in rows],
            target_mean_entropy=args.target_entropy,
        )
        if temperature["absolute_error"] > 0.1 * args.target_entropy:
            raise RuntimeError("Entropy-matched comparator missed the 10% tolerance")

    results: dict[str, dict[int, list[dict[str, Any]]]] = {
        "prior": {count: [] for count in sample_counts}
    }
    if args.include_temperature:
        results["temperature"] = {count: [] for count in sample_counts}
    if args.include_iso_compute:
        results["iso_compute_depth"] = {count: [] for count in sample_counts}
    if args.include_posterior_teacher:
        results["posterior_teacher"] = {count: [] for count in sample_counts}
    rng_manifest: list[dict[str, Any]] = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        Path(args.resume_cache_path)
        if args.resume_cache_path
        else output_dir / "row_cache.jsonl"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = read_resume_cache(cache_path)
    for index, payload in enumerate(cached):
        if index >= len(rows) or payload["id"] != rows[index]["id"]:
            raise RuntimeError("Phase G resume cache does not match the frozen row order")
        rng_manifest.append(payload["rng"])
        for arm, by_k in results.items():
            for count in sample_counts:
                by_k[count].append(payload["arms"][arm][str(count)])
    cache_handle = cache_path.open("a", encoding="utf-8")

    try:
        for index, row in enumerate(rows, start=1):
            if index <= len(cached):
                continue
            if index == 1 or index % args.progress_every == 0 or index == len(rows):
                print(f"phase_g_eval_progress row={index}/{len(rows)} depth={row['depth']}", flush=True)
            row_seed = args.seed + index * 1_000_003
            prior_predictions, prior_metrics = phase_g_predictions(
                wrapper,
                tokenizer,
                row,
                k_max=max(sample_counts),
                seed_base=row_seed,
                posterior_teacher=False,
                device=args.device,
            )
            teacher_predictions = None
            if args.include_posterior_teacher:
                teacher_predictions, _ = phase_g_predictions(
                    wrapper,
                    tokenizer,
                    row,
                    k_max=max(sample_counts),
                    seed_base=row_seed,
                    posterior_teacher=True,
                    device=args.device,
                )
            temp_predictions = (
                sample_temperature_predictions(
                    deterministic[row["id"]]["scores"],
                    temperature=float(temperature["temperature"]),
                    k_max=max(sample_counts),
                    seed=row_seed,
                )
                if temperature is not None
                else None
            )
            iso_predictions = (
                deterministic_iso_predictions(
                    wrapper,
                    tokenizer,
                    row,
                    sample_counts=sample_counts,
                    device=args.device,
                )
                if args.include_iso_compute
                else None
            )
            rng_row = {
                "id": row["id"],
                "seed_base": row_seed,
                "trajectory_seeds": [row_seed + value for value in range(max(sample_counts))],
                "temperature_seed": row_seed if temp_predictions is not None else None,
            }
            rng_manifest.append(rng_row)
            metadata = {
                "id": row["id"],
                "depth": int(row["depth"]),
                "reachable_set_stratum": row["reachable_set_stratum"],
                "reachable_symbols": row["reachable_symbols"],
            }
            for count in sample_counts:
                results["prior"][count].append(
                    {
                        **metadata,
                        **exact_branching_coverage(prior_predictions[:count], row["reachable_symbols"]),
                        "predictions": prior_predictions[:count],
                        "phase_g_metrics": prior_metrics,
                    }
                )
                if temp_predictions is not None:
                    results["temperature"][count].append(
                        {
                            **metadata,
                            **exact_branching_coverage(temp_predictions[:count], row["reachable_symbols"]),
                            "predictions": temp_predictions[:count],
                        }
                    )
                if teacher_predictions is not None:
                    results["posterior_teacher"][count].append(
                        {
                            **metadata,
                            **exact_branching_coverage(teacher_predictions[:count], row["reachable_symbols"]),
                            "predictions": teacher_predictions[:count],
                        }
                    )
                if iso_predictions is not None:
                    results["iso_compute_depth"][count].append(
                        {
                            **metadata,
                            **exact_branching_coverage([iso_predictions[count]], row["reachable_symbols"]),
                            "predictions": [iso_predictions[count]],
                            "forced_loop_count": count * int(row["depth"]),
                        }
                    )
            cache_payload = {
                "id": row["id"],
                "rng": rng_row,
                "arms": {
                    arm: {
                        str(count): results[arm][count][-1]
                        for count in sample_counts
                    }
                    for arm in results
                },
            }
            cache_handle.write(json.dumps(cache_payload, sort_keys=True) + "\n")
            cache_handle.flush()
    finally:
        cache_handle.close()
    summaries: dict[str, dict[str, Any]] = {}
    for arm, by_k in results.items():
        summaries[arm] = {}
        for count, arm_rows in by_k.items():
            path = output_dir / f"{arm}_K{count}.jsonl"
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in arm_rows),
                encoding="utf-8",
            )
            summaries[arm][str(count)] = summarize_coverage_rows(arm_rows)
    (output_dir / "rng_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rng_manifest),
        encoding="utf-8",
    )

    baseline_cells: dict[tuple[int, str], float] = {}
    current_cells: dict[tuple[int, str], float] = {}
    for row in rows:
        key = (int(row["depth"]), str(row["reachable_set_stratum"]))
        baseline_cells.setdefault(key, 0.0)
        current_cells.setdefault(key, 0.0)
    for key in baseline_cells:
        matching = [row for row in rows if (int(row["depth"]), str(row["reachable_set_stratum"])) == key]
        baseline_cells[key] = sum(bool(deterministic[row["id"]]["valid"]) for row in matching) / len(matching)
        prior_k1 = {row["id"]: row for row in results["prior"][1]}
        current_cells[key] = sum(bool(prior_k1[row["id"]]["valid_samples"]) for row in matching) / len(matching)
    k1_correct = sum(bool(row["valid_samples"]) for row in results["prior"][1])
    k1_gate = {
        "correct": k1_correct,
        "total": len(rows),
        "expected_correct": args.expected_k1_correct,
        "pooled_absolute_delta": abs(k1_correct / len(rows) - args.expected_k1_correct / len(rows)),
        "pooled_tolerance": args.k1_pooled_tolerance,
        "per_cell_tolerance": args.k1_cell_tolerance,
        "per_cell": {
            f"d{depth}_{stratum}": {
                "baseline": baseline_cells[(depth, stratum)],
                "phase_g_k1": current_cells[(depth, stratum)],
                "absolute_delta": abs(
                    baseline_cells[(depth, stratum)] - current_cells[(depth, stratum)]
                ),
            }
            for depth, stratum in sorted(baseline_cells)
        },
    }
    k1_gate["passed"] = (
        k1_gate["pooled_absolute_delta"] <= args.k1_pooled_tolerance
        and all(
            cell["absolute_delta"] <= args.k1_cell_tolerance
            for cell in k1_gate["per_cell"].values()
        )
    )

    comparisons: dict[str, Any] = {}
    k_final = max(sample_counts)
    prior_values = [float(row["coverage"]) for row in results["prior"][k_final]]
    for comparator in ("temperature", "iso_compute_depth"):
        if comparator not in results:
            continue
        comparator_values = [
            float(row["coverage"]) for row in results[comparator][k_final]
        ]
        comparisons[comparator] = {
            "K": k_final,
            "mean_delta": sum(a - b for a, b in zip(prior_values, comparator_values))
            / len(prior_values),
            "sign_test": paired_sign_test(prior_values, comparator_values),
        }
    amortization_gap = None
    if "posterior_teacher" in results:
        teacher_values = [
            float(row["coverage"]) for row in results["posterior_teacher"][k_final]
        ]
        amortization_gap = {
            "K": k_final,
            "posterior_minus_prior_mean_coverage": (
                sum(a - b for a, b in zip(teacher_values, prior_values)) / len(prior_values)
            ),
        }
    summary = {
        "kind": "phase_g_alpha_coverage",
        "status": "finished",
        "keeper": args.keeper,
        "keeper_sha256": args.expected_keeper_sha256,
        "guidance_checkpoint": args.guidance_checkpoint,
        "active_lineage_sha256": lineage_after,
        "sample_counts": sample_counts,
        "summaries": summaries,
        "temperature_match": temperature,
        "k1_parity_gate": k1_gate,
        "locked_coverage_margin": args.coverage_margin,
        "comparisons": comparisons,
        "amortization_gap": amortization_gap,
    }
    if {"temperature", "iso_compute_depth"}.issubset(comparisons):
        summary["verdict"] = (
            "open_G_beta"
            if k1_gate["passed"]
            and all(
                comparison["mean_delta"] >= args.coverage_margin
                and comparison["sign_test"]["one_sided_p"] < 0.05
                for comparison in comparisons.values()
            )
            else "G_alpha_did_not_clear_both_locked_comparators"
        )
    else:
        summary["verdict"] = "partial_readout_no_final_verdict"
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
