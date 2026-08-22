"""Execute the signed Stage 2B-S preflight, probes, or CPU desk audit."""

from __future__ import annotations

import argparse
import json
import random
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

from eval.eval_paper2_phase3_p31_references import MODEL_SPECS
from eval.eval_paper2_phase3_p34_task_trajectory import score_generation
from eval.eval_paper2_stage2b_autopsy import (
    _apply_state,
    _build_model,
    _checkpoint_state,
    _k_sweep,
    _named_trainable_state,
    _state_digest,
)
from eval.eval_paper2_stage2b_campaign import (
    Stage2BTaskInferenceGraph,
    read_jsonl,
    write_jsonl,
)
from training.paper2_stage2b_autopsy import battery_counts
from training.paper2_stage2bs_preludes import (
    EXPECTED_K_SWEEP,
    alignment_receipt,
    correction_references,
    dependency_verdict,
    load_lock,
    noise_verdict,
    prelude1_decision,
    sha256_file,
    starvation_verdict,
    top_singular_receipt,
    transplant_verdict,
)


RUN_KIND = "paper2_stage2bs_preludes_v1"
GENERATION_BATTERIES = {"tier1", "gsm8k", "mbpp"}


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _runtime_receipt() -> dict[str, Any]:
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "weights_dtype": "bfloat16",
        "attention_backend": "sdpa",
    }


def _generation_rows(panel: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in panel if str(row["battery"]) in GENERATION_BATTERIES]
    if len(rows) != 461:
        raise RuntimeError(f"Stage 2B-S generative slice changed: observed {len(rows)}")
    return rows


def _compare_k_rows(actual: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]) -> None:
    by_id = {str(row["item_id"]): row for row in expected}
    if set(by_id) != {str(row["item_id"]) for row in actual}:
        raise RuntimeError("Stage 2B-S preflight row identity changed")
    for row in actual:
        reference = by_id[str(row["item_id"])]
        for key in ("generated_token_ids", "prediction", "augmented_correct"):
            if row.get(key) != reference.get(key):
                raise RuntimeError(
                    f"Stage 2B-S preflight bit-exact mismatch item={row['item_id']} key={key}"
                )


def _model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, dict[str, torch.Tensor], list[dict[str, Any]]]:
    _seed_all(20260819 + int(args.seed))
    wrapper, chain, _groups = _build_model(args)
    initialization = _named_trainable_state(wrapper)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SPECS["base"]["model"], revision=MODEL_SPECS["base"]["revision"]
    )
    return wrapper, tokenizer, initialization, chain


def _save_initialization(
    *, args: argparse.Namespace, initialization: Mapping[str, torch.Tensor]
) -> tuple[Path, str]:
    args.private_dir.mkdir(parents=True, exist_ok=True)
    path = args.private_dir / "initialization_state.pt"
    digest = _state_digest(initialization)
    torch.save(
        {
            "kind": "paper2_stage2bs_initialization_state_v1",
            "seed": args.seed,
            "state": dict(initialization),
            "state_digest": digest,
        },
        path,
    )
    return path, digest


def run_initialize(args: argparse.Namespace) -> dict[str, Any]:
    load_lock(args.lock)
    _wrapper, _tokenizer, initialization, chain = _model_and_tokenizer(args)
    path, digest = _save_initialization(args=args, initialization=initialization)
    result = {
        "kind": "paper2_stage2bs_initialization_v1",
        "status": "complete_score_blind",
        "seed": args.seed,
        "lock_sha256": sha256_file(args.lock),
        "checkpoint_chain": chain,
        "initialization_state": {
            "path": str(path),
            "sha256": sha256_file(path),
            "state_digest": digest,
        },
        "model_loaded": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "task_rows_scored": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / "initialization.json", result)
    return result


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_lock(args.lock)
    wrapper, tokenizer, initialization, chain = _model_and_tokenizer(args)
    rows = _generation_rows(read_jsonl(args.dev1_panel))
    init_state_path, init_state_digest = _save_initialization(
        args=args, initialization=initialization
    )
    sweep_dir = args.private_dir / "preflight_k_sweep"
    sweep = _k_sweep(
        wrapper=wrapper,
        tokenizer=tokenizer,
        rows=rows,
        seed=args.seed,
        condition="stage2bs_preflight",
        private_dir=sweep_dir,
        batch_size=args.generation_batch_size,
    )
    counts = [int(sweep[str(loop)]["correct"]) for loop in range(1, 5)]
    if counts != EXPECTED_K_SWEEP[int(args.seed)]:
        raise RuntimeError(
            f"Stage 2B-S preflight count mismatch seed={args.seed}: {counts}"
        )
    reference_hashes = {}
    for loop in range(1, 5):
        actual_path = sweep_dir / f"k_sweep__stage2bs_preflight__k{loop}.jsonl"
        reference_path = args.reference_k_sweep_dir / f"k_sweep__initialization__k{loop}.jsonl"
        _compare_k_rows(read_jsonl(actual_path), read_jsonl(reference_path))
        reference_hashes[str(loop)] = {
            "reference_sha256": sha256_file(reference_path),
            "rerun_sha256": sha256_file(actual_path),
            "row_predictions_bit_exact": True,
        }
    result = {
        "kind": "paper2_stage2bs_preflight_v1",
        "status": "PASS",
        "seed": args.seed,
        "lock_sha256": sha256_file(args.lock),
        "checkpoint_chain": chain,
        "initialization_state": {
            "path": str(init_state_path),
            "sha256": sha256_file(init_state_path),
            "state_digest": init_state_digest,
        },
        "runtime": _runtime_receipt(),
        "rows": len(rows),
        "battery_counts": battery_counts(rows),
        "correct_by_k": counts,
        "expected_correct_by_k": EXPECTED_K_SWEEP[int(args.seed)],
        "row_comparisons": reference_hashes,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / "preflight.json", result)
    return result


def _validate_preflight(args: argparse.Namespace, initialization: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    receipt = json.loads(args.preflight_receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("kind") != "paper2_stage2bs_preflight_v1"
        or receipt.get("status") != "PASS"
        or int(receipt.get("seed", -1)) != args.seed
        or receipt.get("lock_sha256") != sha256_file(args.lock)
        or receipt.get("initialization_state", {}).get("state_digest")
        != _state_digest(initialization)
    ):
        raise RuntimeError("Stage 2B-S probe phase lacks a matching passed preflight")
    return receipt


def _score_condition(
    *, graph: Stage2BTaskInferenceGraph, tokenizer: Any, rows: Sequence[Mapping[str, Any]],
    path: Path, batch_size: int
) -> list[dict[str, Any]]:
    if path.is_file():
        cached = read_jsonl(path)
        expected = [str(row["item_id"]) for row in rows]
        observed = [str(row["item_id"]) for row in cached]
        if len(observed) != len(expected) or set(observed) != set(expected):
            raise RuntimeError(f"Stage 2B-S cached score row order changed: {path}")
        return cached
    scored = score_generation(graph, tokenizer, rows, batch_size=batch_size)
    write_jsonl(path, scored)
    return scored


def _correct(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(bool(row["augmented_correct"]) for row in rows)


def _transplant_pairs(
    panel: Sequence[Mapping[str, Any]], native: Sequence[Mapping[str, Any]], *, count: int
) -> list[dict[str, Any]]:
    correct = {str(row["item_id"]) for row in native if row["augmented_correct"]}
    by_battery: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in panel:
        by_battery[str(row["battery"])].append(dict(row))
    pairs = []
    for battery in sorted(by_battery):
        candidates = by_battery[battery]
        for index in range(0, len(candidates) - 1, 2):
            pair = (candidates[index], candidates[index + 1])
            if all(str(row["item_id"]) in correct for row in pair):
                pairs.append(pair)
    if len(pairs) < count:
        raise RuntimeError(f"Stage 2B-S has only {len(pairs)} eligible transplant pairs")
    selected = pairs[:count]
    return [row for pair in selected for row in pair]


def run_probes(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_lock(args.lock)
    wrapper, tokenizer, initialization, chain = _model_and_tokenizer(args)
    preflight = _validate_preflight(args, initialization)
    rows = _generation_rows(read_jsonl(args.dev1_panel))
    args.private_dir.mkdir(parents=True, exist_ok=True)
    native = read_jsonl(
        args.preflight_private_dir / "preflight_k_sweep/k_sweep__stage2bs_preflight__k4.jsonl"
    )
    k3 = read_jsonl(
        args.preflight_private_dir / "preflight_k_sweep/k_sweep__stage2bs_preflight__k3.jsonl"
    )
    native_correct = _correct(native)
    k3_correct = _correct(k3)
    native_margin = native_correct - k3_correct
    if native_margin <= 0:
        raise RuntimeError("Stage 2B-S native K4-over-K3 margin is not positive")

    noise_cells = {}
    retained = {}
    for epsilon in lock["prelude_1"]["noise_epsilons"]:
        graph = Stage2BTaskInferenceGraph(
            wrapper=wrapper, stage="M2", amplitude=0.05, flow_loops=4,
            incremental_cache=True, probe_noise_epsilon=float(epsilon),
            probe_noise_loop_index=1,
            probe_noise_seed_prefix="20260821",
        )
        scored = _score_condition(
            graph=graph, tokenizer=tokenizer, rows=rows,
            path=args.private_dir / f"noise_epsilon_{epsilon:.3g}.jsonl",
            batch_size=args.generation_batch_size,
        )
        correct = _correct(scored)
        fraction = (correct - k3_correct) / native_margin
        retained[float(epsilon)] = fraction
        noise_cells[str(epsilon)] = {
            "correct": correct,
            "k4_over_k3_margin": correct - k3_correct,
            "native_margin_retained_fraction": fraction,
        }

    dependency_graph = Stage2BTaskInferenceGraph(
        wrapper=wrapper, stage="M2", amplitude=0.05, flow_loops=4,
        incremental_cache=True,
        loop_diagnostic_modes={1: "inherited_flow_off", 2: "inherited_flow_off"},
    )
    dependency_path = args.private_dir / "k2_k3_inherited_flow_off.jsonl"
    activation_path = args.private_dir / "k2_k3_inherited_flow_off_activation.json"
    if dependency_path.is_file() and not activation_path.is_file():
        dependency_path.unlink()
    dependency_rows = _score_condition(
        graph=dependency_graph, tokenizer=tokenizer, rows=rows,
        path=dependency_path,
        batch_size=args.generation_batch_size,
    )
    activation = (
        json.loads(activation_path.read_text(encoding="utf-8"))
        if activation_path.is_file()
        else getattr(dependency_graph, "_probe_metric_maxima", {})
    )
    if activation.get("stage2b_flow_update_loop_2_max_abs", -1.0) != 0.0 or activation.get(
        "stage2b_flow_update_loop_3_max_abs", -1.0
    ) != 0.0:
        raise RuntimeError("Stage 2B-S K2/K3 inherited-flow zero assertion failed")
    if activation.get("stage2b_writeback_ratio_loop_2_max", 0.0) <= 0.0 or activation.get(
        "stage2b_writeback_ratio_loop_3_max", 0.0
    ) <= 0.0:
        raise RuntimeError("Stage 2B-S K2/K3 bridge-preservation assertion failed")
    atomic_json(activation_path, activation)

    paired_rows = _transplant_pairs(rows, native, count=64)
    pair_manifest = [
        {
            "pair_index": index // 2,
            "battery": str(paired_rows[index]["battery"]),
            "left_item_id": str(paired_rows[index]["item_id"]),
            "right_item_id": str(paired_rows[index + 1]["item_id"]),
        }
        for index in range(0, len(paired_rows), 2)
    ]
    pair_manifest_path = args.private_dir / "cross_question_pair_manifest.jsonl"
    write_jsonl(pair_manifest_path, pair_manifest)
    transplant_graph = Stage2BTaskInferenceGraph(
        wrapper=wrapper, stage="M2", amplitude=0.05, flow_loops=4,
        incremental_cache=True, probe_transplant_loop_index=3,
    )
    transplanted = _score_condition(
        graph=transplant_graph, tokenizer=tokenizer, rows=paired_rows,
        path=args.private_dir / "cross_question_k3_to_k4.jsonl", batch_size=2,
    )
    transplant_correct = _correct(transplanted)
    seed_verdict = {
        "noise": noise_verdict(retained),
        "dependency": dependency_verdict(native_correct, _correct(dependency_rows)),
        "transplant": transplant_verdict(len(paired_rows), transplant_correct),
    }
    result = {
        "kind": "paper2_stage2bs_prelude1_v1",
        "status": "complete_score_only",
        "seed": args.seed,
        "lock_sha256": sha256_file(args.lock),
        "preflight_sha256": sha256_file(args.preflight_receipt),
        "checkpoint_chain": chain,
        "runtime": _runtime_receipt(),
        "native": {"k3_correct": k3_correct, "k4_correct": native_correct, "margin": native_margin},
        "noise": {"cells": noise_cells, "verdict": seed_verdict["noise"]},
        "k2_k3_inherited_flow_off": {
            "correct": _correct(dependency_rows),
            "native_fraction": _correct(dependency_rows) / native_correct,
            "activation_maxima": activation,
            "verdict": seed_verdict["dependency"],
        },
        "cross_question_transplant": {
            "pairs": 64,
            "rows": len(paired_rows),
            "native_correct": len(paired_rows),
            "transplanted_correct": transplant_correct,
            "native_fraction": transplant_correct / len(paired_rows),
            "pair_manifest_sha256": sha256_file(pair_manifest_path),
            "verdict": seed_verdict["transplant"],
        },
        "seed_verdict": seed_verdict,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / "prelude1.json", result)
    return result


def _parameter_group(name: str) -> str | None:
    if name == "stage2b_depth_attachment.flow.hidden_innovation.weight":
        return "W_H"
    if name == "stage2b_depth_attachment.flow.prompt_gate.weight":
        return "W_P"
    if name.startswith("stage2b_depth_attachment.bridge."):
        if "gate" in name:
            return "bridge_g_L"
        return "bridge_B_L"
    if name.startswith("stage2b_depth_attachment.initializer."):
        return "B0"
    if "lora_a" in name or "lora_b" in name:
        return "loop_lora"
    return None


def run_desk(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_lock(args.lock)
    if lock["prelude_2"].get("f1_estimator") != "absolute_delta_WP_over_absolute_delta_WH":
        raise RuntimeError("Stage 2B-S F1 estimator amendment has not been ratified")
    seed_rows = []
    same_shapes = []
    private_vectors = {}
    for seed in (0, 1):
        initial_payload = torch.load(args.initialization_states[seed], map_location="cpu", weights_only=False)
        initial = initial_payload["state"]
        final = _checkpoint_state(args.stop_checkpoints[seed])
        if set(initial) != set(final):
            raise RuntimeError(f"Stage 2B-S desk state schema changed for seed {seed}")
        movement = {}
        groups: dict[str, list[dict[str, float | None]]] = defaultdict(list)
        for name in sorted(initial):
            group = _parameter_group(name)
            if group is None:
                continue
            initial_norm = float(torch.linalg.vector_norm(initial[name].float()))
            delta_norm = float(torch.linalg.vector_norm((final[name] - initial[name]).float()))
            relative = delta_norm / initial_norm if initial_norm > 0.0 else None
            measurement = {
                "group": group,
                "initial_frobenius": initial_norm,
                "delta_frobenius": delta_norm,
                "relative_frobenius": relative,
            }
            movement[name] = measurement
            groups[group].append(measurement)
        for required in ("W_H", "W_P", "bridge_B_L", "bridge_g_L", "loop_lora"):
            if not groups.get(required):
                raise RuntimeError(f"Stage 2B-S desk audit missing parameter group {required}")
        group_summary = {}
        for name, values in groups.items():
            relative_values = [
                float(value["relative_frobenius"])
                for value in values
                if value["relative_frobenius"] is not None
            ]
            group_summary[name] = {
                "parameter_tensors": len(values),
                "relative_defined_tensors": len(relative_values),
                "mean_relative_frobenius": (
                    sum(relative_values) / len(relative_values) if relative_values else None
                ),
                "maximum_relative_frobenius": max(relative_values) if relative_values else None,
                "joint_delta_frobenius": float(
                    sum(float(value["delta_frobenius"]) ** 2 for value in values) ** 0.5
                ),
            }
        wh_name = "stage2b_depth_attachment.flow.hidden_innovation.weight"
        wp_name = "stage2b_depth_attachment.flow.prompt_gate.weight"
        same_shapes.append(tuple(initial[wh_name].shape) == tuple(initial[wp_name].shape))
        wh_move = float(movement[wh_name]["delta_frobenius"])
        wp_move = float(movement[wp_name]["delta_frobenius"])
        wh_dtype = initial[wh_name].dtype
        if not wh_dtype.is_floating_point:
            raise RuntimeError("Stage 2B-S W_H must use a floating-point dtype")
        wh_epsilon_scale = float(torch.finfo(wh_dtype).eps) * float(initial[wh_name].numel()) ** 0.5
        wh_degenerate = wh_move < wh_epsilon_scale
        artifact = torch.load(args.correction_artifacts[seed], map_location="cpu", weights_only=False)
        references = correction_references(artifact, seed=20260819 + seed)
        singular = {}
        for label, name in (("W_H", wh_name), ("W_P", wp_name)):
            receipt = top_singular_receipt(final[name] - initial[name], rank=3)
            vectors = receipt.pop("right_singular_vectors")
            singular[label] = {
                **receipt,
                "alignment_cosines": alignment_receipt(vectors, references),
            }
            private_vectors[f"seed_{seed}_{label}"] = vectors
        lora_pairs = {}
        for name in sorted(initial):
            if "lora_a" not in name:
                continue
            partner = name.replace("lora_a", "lora_b")
            if partner not in initial:
                raise RuntimeError(f"Stage 2B-S loop-LoRA pair is incomplete: {name}")
            numerator = torch.sqrt(
                torch.linalg.vector_norm((final[name] - initial[name]).float()).square()
                + torch.linalg.vector_norm((final[partner] - initial[partner]).float()).square()
            )
            denominator = torch.sqrt(
                torch.linalg.vector_norm(initial[name].float()).square()
                + torch.linalg.vector_norm(initial[partner].float()).square()
            ).clamp_min(1e-12)
            lora_pairs[name.rsplit(".lora_a", 1)[0]] = {
                "a_parameter": name,
                "b_parameter": partner,
                "joint_relative_frobenius": float(numerator / denominator),
            }
        seed_rows.append(
            {
                "seed": seed,
                "state_digests": {
                    "initialization": initial_payload["state_digest"],
                    "stop": _state_digest(final),
                },
                "movement_by_parameter": movement,
                "movement_by_group": group_summary,
                "loop_lora_factor_pairs": lora_pairs,
                "W_H_relative_frobenius": movement[wh_name]["relative_frobenius"],
                "W_P_relative_frobenius": movement[wp_name]["relative_frobenius"],
                "W_P_relative_frobenius_defined": movement[wp_name]["relative_frobenius"] is not None,
                "F1_raw_values": {
                    "W_P_delta_frobenius": wp_move,
                    "W_H_delta_frobenius": wh_move,
                    "W_H_initial_frobenius": float(torch.linalg.vector_norm(initial[wh_name].float())),
                    "W_P_initial_frobenius": float(torch.linalg.vector_norm(initial[wp_name].float())),
                    "W_H_final_frobenius": float(torch.linalg.vector_norm(final[wh_name].float())),
                    "W_P_final_frobenius": float(torch.linalg.vector_norm(final[wp_name].float())),
                    "W_H_dtype": str(wh_dtype),
                    "W_H_dtype_epsilon_scale": wh_epsilon_scale,
                    "W_H_degenerate": wh_degenerate,
                },
                "W_P_delta_over_W_H_delta": None if wh_degenerate else wp_move / wh_move,
                "top_singular_geometry": singular,
            }
        )
    degenerate = any(row["F1_raw_values"]["W_H_degenerate"] for row in seed_rows)
    ratios = [row["W_P_delta_over_W_H_delta"] for row in seed_rows]
    args.private_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = args.private_dir / "top_singular_vectors.pt"
    torch.save(private_vectors, vectors_path)
    result = {
        "kind": "paper2_stage2bs_prelude2_v1",
        "status": "complete_cpu_only",
        "lock_sha256": sha256_file(args.lock),
        "seeds": seed_rows,
        "F1": {
            "estimator": "absolute_delta_WP_over_absolute_delta_WH",
            "same_shape_asserted": all(same_shapes),
            "ratios": ratios,
            "verdict": "DEGENERATE" if degenerate else starvation_verdict(ratios),
        },
        "private_singular_vectors": {"path": str(vectors_path), "sha256": sha256_file(vectors_path)},
        "model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / "prelude2.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("initialize", "preflight", "probes", "desk"), required=True
    )
    parser.add_argument("--seed", type=int, choices=(0, 1))
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--dev1_panel", type=Path)
    parser.add_argument("--reference_k_sweep_dir", type=Path)
    parser.add_argument("--preflight_receipt", type=Path)
    parser.add_argument("--preflight_private_dir", type=Path)
    for name in ("migrated", "p33", "i1", "p34", "p35", "model_cache", "output_dir", "private_dir"):
        parser.add_argument(f"--{name}", type=Path)
    for name in ("migrated_sha256", "p33_sha256", "i1_sha256", "p34_sha256", "p35_sha256"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--initialization_states", type=Path, nargs=2)
    parser.add_argument("--stop_checkpoints", type=Path, nargs=2)
    parser.add_argument("--correction_artifacts", type=Path, nargs=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generation_batch_size", type=int, default=8)
    args = parser.parse_args()
    common = ["output_dir", "private_dir"]
    if args.phase in {"initialize", "preflight", "probes"}:
        common += [
            "seed", "dev1_panel", "migrated", "p33", "i1", "p34", "p35", "model_cache",
            "migrated_sha256", "p33_sha256", "i1_sha256", "p34_sha256", "p35_sha256",
        ]
    if args.phase == "initialize":
        pass
    elif args.phase == "preflight":
        common += ["reference_k_sweep_dir"]
    elif args.phase == "probes":
        common += ["preflight_receipt", "preflight_private_dir"]
    else:
        common += ["initialization_states", "stop_checkpoints", "correction_artifacts"]
    missing = [name for name in common if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {missing}")
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "initialize":
            result = run_initialize(args)
        elif args.phase == "preflight":
            result = run_preflight(args)
        elif args.phase == "probes":
            result = run_probes(args)
        else:
            result = run_desk(args)
    except Exception as error:
        atomic_json(
            args.output_dir / "status.json",
            {
                "kind": RUN_KIND,
                "status": "failed",
                "phase": args.phase,
                "seed": args.seed,
                "exception_type": type(error).__name__,
                "exception": str(error),
                "traceback": traceback.format_exc(),
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "confirm_scored": False,
                "eval_e_scored": False,
            },
        )
        raise
    atomic_json(args.output_dir / "status.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
