"""Run the signed, score-only Stage 2B-A stop autopsy."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from eval.eval_paper2_phase3_p31_references import MODEL_SPECS
from eval.eval_paper2_phase3_p34_task_trajectory import score_generation
from eval.eval_paper2_stage2b_campaign import (
    Stage2BTaskInferenceGraph,
    _forced_target,
    read_jsonl,
    score_dev1,
    score_dev2_margins,
    write_jsonl,
)
from training.paper2_stage2b_autopsy import (
    battery_counts,
    load_and_validate_autopsy_lock,
    margin_correlation_receipt,
    sha256_file,
    stable_dev2_subsample,
)
from training.paper2_stage2b_depth import (
    monotonicity_hinge,
    sparse_forward_kl_per_example,
)
from training.run_paper2_stage2b_depth import (
    _apply_named_state,
    _build_model,
    _named_trainable_state,
)
from training.run_paper2_phase3_p33 import tensor_digest


RUN_KIND = "paper2_stage2b_autopsy_v1"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _checkpoint_state(path: Path, *, expected_sha256: str | None = None) -> dict[str, torch.Tensor]:
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise RuntimeError(f"Stage 2B-A checkpoint hash changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("trainable_state", payload.get("ema_state"))
    if not isinstance(state, Mapping):
        raise RuntimeError(f"Stage 2B-A checkpoint lacks an EMA state: {path}")
    return {str(name): value.detach().cpu() for name, value in state.items()}


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    return tensor_digest({name: value for name, value in state.items()})


def _apply_state(wrapper: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    _apply_named_state(wrapper, state)
    wrapper.eval()


def _condition_name(prefix: str, *, gamma: float | None = None, mode: str | None = None) -> str:
    parts = [prefix]
    if gamma is not None:
        parts.append(f"gamma_{gamma:.2f}".replace(".", "p"))
    if mode is not None:
        parts.append(mode)
    return "__".join(parts)


def _score_dev1_condition(
    *,
    wrapper: Any,
    tokenizer: Any,
    panel: Sequence[Mapping[str, Any]],
    base_rows: Mapping[str, Mapping[str, Any]],
    initialization_rows: Mapping[str, Mapping[str, Any]],
    seed: int,
    gamma: float,
    mode: str,
    condition: str,
    private_dir: Path,
    mcq_batch_size: int,
    generation_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph = Stage2BTaskInferenceGraph(
        wrapper=wrapper,
        stage="M2",
        amplitude=gamma,
        flow_loops=4,
        diagnostic_mode=mode,
    )
    rows, summary = score_dev1(
        graph=graph,
        tokenizer=tokenizer,
        panel=panel,
        base_rows=base_rows,
        initialization_rows=initialization_rows,
        seed=seed,
        look=1000,
        mcq_batch_size=mcq_batch_size,
        generation_batch_size=generation_batch_size,
    )
    for row in rows:
        row["autopsy_condition"] = condition
    summary["autopsy_condition"] = condition
    summary["gamma"] = gamma
    summary["diagnostic_mode"] = mode
    write_jsonl(private_dir / f"dev1__{condition}.jsonl", rows)
    return rows, summary


def _score_dev2_condition(
    *,
    wrapper: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    gamma: float,
    mode: str,
    condition: str,
    private_dir: Path,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph = Stage2BTaskInferenceGraph(
        wrapper=wrapper,
        stage="M2",
        amplitude=gamma,
        flow_loops=4,
        diagnostic_mode=mode,
    )
    scored, summary = score_dev2_margins(
        graph=graph,
        tokenizer=tokenizer,
        rows=rows,
        seed=seed,
        look=1000,
        batch_size=batch_size,
    )
    for row in scored:
        row["autopsy_condition"] = condition
    summary["autopsy_condition"] = condition
    summary["gamma"] = gamma
    summary["diagnostic_mode"] = mode
    write_jsonl(private_dir / f"dev2__{condition}.jsonl", scored)
    return scored, summary


def _same_predictions(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> bool:
    left_rows = {str(row["item_id"]): row for row in left}
    right_rows = {str(row["item_id"]): row for row in right}
    if set(left_rows) != set(right_rows):
        return False
    keys = ("prediction", "augmented_correct", "generated_token_ids")
    return all(
        all(left_rows[item].get(key) == right_rows[item].get(key) for key in keys)
        for item in left_rows
    )


@torch.inference_mode()
def _zero_write_logits(wrapper: Any, tokenizer: Any, row: Mapping[str, Any]) -> torch.Tensor:
    prompt, target = _forced_target(tokenizer, row)
    tokens = torch.tensor([prompt + target], dtype=torch.long, device="cuda")
    attention = torch.ones_like(tokens)
    output = wrapper(
        input_ids=tokens,
        attention_mask=attention,
        max_loops=4,
        stage2b_depth_enabled=True,
        stage2b_stage="M2",
        stage2b_amplitude=0.0,
        stage2b_diagnostic_mode="zero_write",
        return_loop_logits=True,
        use_cache=False,
        return_dict=True,
    )
    return output.loop_logits.detach().float().cpu()


def _offdiagonal_cosine(states: torch.Tensor, *, center: bool) -> dict[str, float]:
    values = states.float()
    if center:
        values = values - values.mean(dim=0, keepdim=True)
    values = F.normalize(values, dim=-1, eps=1e-12)
    matrix = values @ values.transpose(0, 1)
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    off = matrix[mask]
    return {
        "mean": float(off.mean()),
        "median": float(off.median()),
        "p95": float(torch.quantile(off, 0.95)),
    }


@torch.inference_mode()
def _state_similarity(
    *, wrapper: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], batch_size: int
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    prepared = [(row, *_forced_target(tokenizer, row)) for row in rows]
    loop1_states = []
    loop4_states = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        widths = [len(prompt) + len(target) for _row, prompt, target in batch]
        width = max(widths)
        input_ids = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
        attention = torch.zeros_like(input_ids)
        for index, (_row, prompt, target) in enumerate(batch):
            tokens = prompt + target
            input_ids[index, : len(tokens)] = torch.tensor(tokens, device="cuda")
            attention[index, : len(tokens)] = 1
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention,
            max_loops=4,
            stage2b_depth_enabled=True,
            stage2b_stage="M2",
            stage2b_amplitude=0.05,
            stage2b_diagnostic_mode="standard",
            return_loop_recurrent_states=True,
            use_cache=False,
            return_dict=True,
        )
        states = output.loop_recurrent_states[:, 0].float()
        weights = attention.float().unsqueeze(1).unsqueeze(-1)
        pooled = (states * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        loop1_states.append(pooled[:, 0].cpu())
        loop4_states.append(pooled[:, 3].cpu())
    loop1 = torch.cat(loop1_states)
    loop4 = torch.cat(loop4_states)
    direction = loop4 - loop1
    receipt = {
        "rows": len(rows),
        "raw_loop4_offdiagonal_cosine": _offdiagonal_cosine(loop4, center=False),
        "centered_loop4_offdiagonal_cosine": _offdiagonal_cosine(loop4, center=True),
        "loop4_minus_loop1_direction_offdiagonal_cosine": _offdiagonal_cosine(
            direction, center=False
        ),
        "loop4_state_variance": float(loop4.var(dim=0, unbiased=False).mean()),
        "loop1_state_variance": float(loop1.var(dim=0, unbiased=False).mean()),
    }
    return receipt, {"loop1": loop1, "loop4": loop4, "direction": direction}


@torch.inference_mode()
def _objective_read(wrapper: Any, teacher_cache: Path) -> dict[str, Any]:
    teacher = torch.load(teacher_cache, map_location="cpu", weights_only=False)
    if teacher.get("kind") != "paper2_stage2b_calibration_teacher_cache_v1":
        raise RuntimeError("wrong Stage 2B-A heldout teacher cache")
    ce_by_loop: list[list[float]] = [[] for _ in range(4)]
    kl_by_loop: list[list[float]] = [[] for _ in range(4)]
    mono = []
    for row in teacher["rows"]:
        input_ids = row["input_ids"].long().unsqueeze(0).to("cuda")
        attention = torch.ones_like(input_ids)
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention,
            max_loops=4,
            stage2b_depth_enabled=True,
            stage2b_stage="M2",
            stage2b_amplitude=0.05,
            return_loop_logits=True,
            use_cache=False,
            return_dict=True,
        )
        loops = output.loop_logits[:, 0, :, :-1]
        top_ids = row["teacher_topk_token_ids"].long().unsqueeze(0).to("cuda")
        top_logits = row["teacher_topk_logits"].unsqueeze(0).to("cuda")
        targets = top_ids[..., 0]
        mask = torch.ones(targets.shape, dtype=torch.bool, device="cuda")
        row_kls = []
        for index in range(4):
            kl = sparse_forward_kl_per_example(
                loops[:, index], top_ids, top_logits, mask
            )
            ce = F.cross_entropy(
                loops[:, index].float().reshape(-1, loops.shape[-1]),
                targets.reshape(-1),
            )
            kl_by_loop[index].append(float(kl.mean().cpu()))
            ce_by_loop[index].append(float(ce.cpu()))
            row_kls.append(kl)
        mono.append(float(monotonicity_hinge(row_kls, delta=0.01).mean().cpu()))
    mean = lambda values: sum(values) / len(values)
    return {
        "rows": len(teacher["rows"]),
        "next_token_positions": sum(
            int(row["teacher_topk_token_ids"].shape[0]) for row in teacher["rows"]
        ),
        "per_loop_ce": [mean(values) for values in ce_by_loop],
        "per_loop_forward_kl": [mean(values) for values in kl_by_loop],
        "monotonicity_component": mean(mono),
        "teacher_manifest_sha256": teacher["manifest_sha256"],
    }


def _k_sweep(
    *,
    wrapper: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    condition: str,
    private_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    summary = {}
    for loops in (1, 2, 3, 4):
        graph = Stage2BTaskInferenceGraph(
            wrapper=wrapper,
            stage="M2",
            amplitude=0.05,
            flow_loops=loops,
        )
        scored = score_generation(graph, tokenizer, rows, batch_size=batch_size)
        for row in scored:
            row["seed"] = seed
            row["flow_loops"] = loops
            row["autopsy_condition"] = condition
        write_jsonl(private_dir / f"k_sweep__{condition}__k{loops}.jsonl", scored)
        summary[str(loops)] = {
            "rows": len(scored),
            "correct": sum(bool(row["augmented_correct"]) for row in scored),
            "battery_counts": battery_counts(scored),
        }
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_and_validate_autopsy_lock(args.lock, require_signature=True)
    if args.dev2_subsample_manifest is None:
        raise RuntimeError("signed Stage 2B-A run requires a frozen DEV-2 subsample")
    if sha256_file(args.dev2_subsample_manifest) != lock["dev2_subsample"]["manifest_sha256"]:
        raise RuntimeError("Stage 2B-A DEV-2 subsample hash changed")
    if sha256_file(args.heldout_teacher_cache) != lock["heldout_training_slice"]["teacher_cache_sha256"]:
        raise RuntimeError("Stage 2B-A heldout teacher cache hash changed")
    random.seed(20260819 + args.seed)
    np.random.seed(20260819 + args.seed)
    torch.manual_seed(20260819 + args.seed)
    torch.cuda.manual_seed_all(20260819 + args.seed)

    wrapper, chain, _groups = _build_model(args)
    initialization_state = _named_trainable_state(wrapper)
    stop_spec = lock["stop_checkpoints"][str(args.seed)]
    stop_state = _checkpoint_state(
        args.stop_checkpoint, expected_sha256=stop_spec["sha256"]
    )
    dev1 = read_jsonl(args.dev1_panel)
    reference = {str(row["item_id"]): row for row in read_jsonl(args.reference_rows)}
    manifest = read_jsonl(args.dev2_subsample_manifest)
    dev2_subsample = [reference[str(row["item_id"])] for row in manifest]
    base_rows = {str(row["item_id"]): row for row in read_jsonl(args.base_scores)}
    initialization_rows = {
        str(row["item_id"]): row for row in read_jsonl(args.initialization_scores)
    }
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SPECS["base"]["model"], revision=MODEL_SPECS["base"]["revision"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)

    receipt: dict[str, Any] = {
        "kind": RUN_KIND,
        "status": "running",
        "seed": args.seed,
        "lock_sha256": sha256_file(args.lock),
        "checkpoint_chain": chain,
        "state_digests": {
            "initialization": _state_digest(initialization_state),
            "stop_ema": _state_digest(stop_state),
        },
        "dev2_subsample": {
            "rows": len(dev2_subsample),
            "manifest_sha256": sha256_file(args.dev2_subsample_manifest),
            "battery_counts": battery_counts(dev2_subsample),
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / "status.json", receipt)

    _apply_state(wrapper, initialization_state)
    zero_init_logits = _zero_write_logits(wrapper, tokenizer, dev2_subsample[0])
    _apply_state(wrapper, stop_state)
    zero_stop_logits = _zero_write_logits(wrapper, tokenizer, dev2_subsample[0])
    zero_logit_max_abs = float((zero_init_logits - zero_stop_logits).abs().max())
    zero_logit_exact = torch.equal(zero_init_logits, zero_stop_logits)
    if not zero_logit_exact:
        raise RuntimeError(
            f"Stage 2B-A zero-write logit identity failed: max_abs={zero_logit_max_abs}"
        )

    amplitude = defaultdict(dict)
    amplitude_rows: dict[str, dict[float, list[dict[str, Any]]]] = defaultdict(dict)
    for state_name, state in (("initialization", initialization_state), ("stop", stop_state)):
        _apply_state(wrapper, state)
        for gamma in (0.0, 0.01, 0.02, 0.05):
            mode = "zero_write" if gamma == 0.0 else "standard"
            condition = _condition_name(state_name, gamma=gamma)
            rows, summary = _score_dev1_condition(
                wrapper=wrapper,
                tokenizer=tokenizer,
                panel=dev1,
                base_rows=base_rows,
                initialization_rows=initialization_rows,
                seed=args.seed,
                gamma=gamma,
                mode=mode,
                condition=condition,
                private_dir=args.private_dir,
                mcq_batch_size=args.mcq_batch_size,
                generation_batch_size=args.generation_batch_size,
            )
            amplitude[state_name][str(gamma)] = summary
            amplitude_rows[state_name][gamma] = rows
    zero_identity = _same_predictions(
        amplitude_rows["initialization"][0.0], amplitude_rows["stop"][0.0]
    )
    if not zero_identity:
        raise RuntimeError("Stage 2B-A zero-write identity gate failed")
    receipt["amplitude_response"] = {
        "cells": dict(amplitude),
        "zero_write_checkpoint_independent": zero_identity,
        "zero_write_full_logit_bit_exact": zero_logit_exact,
        "zero_write_full_logit_max_abs_difference": zero_logit_max_abs,
    }
    atomic_json(args.output_dir / "status.json", receipt)

    _apply_state(wrapper, stop_state)
    component = {}
    for mode in ("standard", "constitutive_off", "fresh_state_each_loop", "inherited_flow_off"):
        condition = _condition_name("stop", gamma=0.05, mode=mode)
        _rows, dev1_summary = _score_dev1_condition(
            wrapper=wrapper,
            tokenizer=tokenizer,
            panel=dev1,
            base_rows=base_rows,
            initialization_rows=initialization_rows,
            seed=args.seed,
            gamma=0.05,
            mode=mode,
            condition=condition,
            private_dir=args.private_dir,
            mcq_batch_size=args.mcq_batch_size,
            generation_batch_size=args.generation_batch_size,
        )
        _margin_rows, dev2_summary = _score_dev2_condition(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=dev2_subsample,
            seed=args.seed,
            gamma=0.05,
            mode=mode,
            condition=condition,
            private_dir=args.private_dir,
            batch_size=args.margin_batch_size,
        )
        component[mode] = {"dev1": dev1_summary, "dev2": dev2_summary}
    receipt["component_attribution"] = component
    atomic_json(args.output_dir / "status.json", receipt)

    attractor = {}
    margin_rows_by_state = {}
    for state_name, state in (("initialization", initialization_state), ("stop", stop_state)):
        _apply_state(wrapper, state)
        condition = _condition_name(state_name, gamma=0.05, mode="standard")
        margin_rows, margin_summary = _score_dev2_condition(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=dev2_subsample,
            seed=args.seed,
            gamma=0.05,
            mode="standard",
            condition=condition,
            private_dir=args.private_dir,
            batch_size=args.margin_batch_size,
        )
        margin_rows_by_state[state_name] = margin_rows
        state_summary, features = _state_similarity(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=dev2_subsample,
            batch_size=args.margin_batch_size,
        )
        feature_path = args.private_dir / f"state_features__{state_name}.pt"
        torch.save(features, feature_path)
        attractor[state_name] = {
            "margin_summary": margin_summary,
            "k1_k4_margin_correlation": margin_correlation_receipt(margin_rows),
            "state_similarity": state_summary,
            "state_feature_sha256": sha256_file(feature_path),
        }
    generative = [row for row in dev1 if row["battery"] in {"gsm8k", "mbpp", "tier1"}]
    for state_name, state in (("initialization", initialization_state), ("stop", stop_state)):
        _apply_state(wrapper, state)
        attractor[state_name]["generative_k_sweep"] = _k_sweep(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=generative,
            seed=args.seed,
            condition=state_name,
            private_dir=args.private_dir,
            batch_size=args.generation_batch_size,
        )
    receipt["attractor_discriminators"] = attractor
    atomic_json(args.output_dir / "status.json", receipt)

    objective = {}
    for state_name, state in (("initialization", initialization_state), ("stop", stop_state)):
        _apply_state(wrapper, state)
        objective[state_name] = _objective_read(wrapper, args.heldout_teacher_cache)
    receipt["objective_task_divergence"] = objective

    onset = {}
    for step, state in ((0, initialization_state), (1000, stop_state)):
        _apply_state(wrapper, state)
        condition = f"onset_step_{step:05d}"
        _rows, summary = _score_dev2_condition(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=dev2_subsample,
            seed=args.seed,
            gamma=0.05,
            mode="standard",
            condition=condition,
            private_dir=args.private_dir,
            batch_size=args.margin_batch_size,
        )
        onset[str(step)] = summary
    for specification in args.trajectory_checkpoint:
        step_text, path_text, expected_sha = specification.split("=", 2)
        step = int(step_text)
        state = _checkpoint_state(Path(path_text), expected_sha256=expected_sha)
        _apply_state(wrapper, state)
        condition = f"onset_step_{step:05d}"
        _rows, summary = _score_dev2_condition(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=dev2_subsample,
            seed=args.seed,
            gamma=0.05,
            mode="standard",
            condition=condition,
            private_dir=args.private_dir,
            batch_size=args.margin_batch_size,
        )
        onset[str(step)] = summary
    if set(map(int, onset)) != set(lock["onset_trajectory"]["steps"]):
        raise RuntimeError("Stage 2B-A onset checkpoint set changed")
    receipt["onset_trajectory"] = onset
    receipt["training_log_monotonicity"] = args.training_log_monotonicity
    receipt["runtime"] = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "weights_dtype": "bfloat16",
        "attention_backend": "sdpa",
    }
    receipt["status"] = "complete_score_only"
    atomic_json(args.output_dir / "summary.json", receipt)
    atomic_json(args.output_dir / "status.json", receipt)
    return receipt


def freeze_dev2_subsample(args: argparse.Namespace) -> dict[str, Any]:
    source = read_jsonl(args.dev2_manifest)
    selected = stable_dev2_subsample(source, size=256, seed=20260819)
    manifest = [
        {"item_id": str(row["item_id"]), "battery": str(row["battery"])}
        for row in selected
    ]
    write_jsonl(args.dev2_subsample_manifest, manifest)
    receipt = {
        "kind": "paper2_stage2b_autopsy_dev2_subsample_v1",
        "status": "frozen_before_autopsy_model_contact",
        "rows": len(manifest),
        "seed": 20260819,
        "source_manifest_sha256": sha256_file(args.dev2_manifest),
        "manifest_sha256": sha256_file(args.dev2_subsample_manifest),
        "battery_counts": battery_counts(manifest),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.dev2_subsample_manifest.with_suffix(".receipt.json"), receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dev2-subsample", action="store_true")
    parser.add_argument("--seed", type=int, choices=(0, 1))
    for name in (
        "lock", "dev1_panel", "dev2_manifest", "dev2_subsample_manifest", "reference_rows", "base_scores",
        "initialization_scores", "heldout_teacher_cache", "stop_checkpoint", "migrated",
        "p33", "i1", "p34", "p35", "model_cache", "output_dir", "private_dir",
    ):
        parser.add_argument(f"--{name}", type=Path)
    for name in ("migrated_sha256", "p33_sha256", "i1_sha256", "p34_sha256", "p35_sha256"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--trajectory_checkpoint", action="append", default=[])
    parser.add_argument("--training_log_monotonicity", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mcq_batch_size", type=int, default=8)
    parser.add_argument("--generation_batch_size", type=int, default=2)
    parser.add_argument("--margin_batch_size", type=int, default=2)
    args = parser.parse_args()
    required = ["dev2_manifest", "dev2_subsample_manifest"]
    if not args.freeze_dev2_subsample:
        required.extend(
            [
                "seed", "lock", "dev1_panel", "reference_rows", "base_scores", "initialization_scores",
                "heldout_teacher_cache", "stop_checkpoint", "migrated", "p33", "i1",
                "p34", "p35", "model_cache", "output_dir", "private_dir",
                "migrated_sha256", "p33_sha256", "i1_sha256", "p34_sha256", "p35_sha256",
                "training_log_monotonicity",
            ]
        )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {missing}")
    return args


def main() -> int:
    args = parse_args()
    result = freeze_dev2_subsample(args) if args.freeze_dev2_subsample else run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
