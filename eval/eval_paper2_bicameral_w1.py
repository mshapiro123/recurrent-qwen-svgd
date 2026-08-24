"""Run the registered score-only W1 correction-causality ladder."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_paper2_stage2b_autopsy import _state_digest
from eval.eval_paper2_stage2b_campaign import _forced_target
from training.paper2_bicameral_w1 import (
    GAMMA,
    deterministic_permutation,
    project_cost_hours,
    scale_external_write,
    summarize_margin_deltas,
)
from training.paper2_stage2bs_depth_study import sha256_file
from training.paper2_stage2bs_depth_study import INITIALIZATION_SEED_BASE
from training.run_paper2_stage2b_depth import _build_model, _named_trainable_state


TEACHER_ID = "Qwen/Qwen2.5-14B-Instruct"
TEACHER_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
EVALUATOR_TAG = "EV-LADDER-1"
SCHEDULE = "sequential_shared_middle_v1"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="",
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_dev2(manifest: Path, reference_rows: Path) -> list[dict[str, Any]]:
    reference = {str(row["item_id"]): row for row in read_jsonl(reference_rows)}
    selected = read_jsonl(manifest)
    rows = [reference[str(row["item_id"])] for row in selected]
    if len(rows) != 2048 or len({str(row["item_id"]) for row in rows}) != 2048:
        raise RuntimeError("W1 DEV-2 identity changed")
    return rows


def dry_rows(rows: Sequence[Mapping[str, Any]], size: int) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"20260824:{row['battery']}:{row['item_id']}".encode("utf-8")
        ).hexdigest(),
    )
    return [dict(row) for row in ranked[: int(size)]]


def prepare_batch(
    tokenizer: Any, rows: Sequence[Mapping[str, Any]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, list[int]]]]:
    prepared = [(row, *_forced_target(tokenizer, row)) for row in rows]
    width = max(len(prompt) + len(target) for _row, prompt, target in prepared)
    input_ids = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    spans = []
    for index, (_row, prompt, target) in enumerate(prepared):
        tokens = prompt + target
        input_ids[index, : len(tokens)] = torch.tensor(tokens, device=device)
        attention[index, : len(tokens)] = 1
        spans.append((len(prompt) - 1, list(target)))
    return input_ids, attention, spans


@contextmanager
def interface_patch(
    wrapper: Any,
    *,
    direction: torch.Tensor | None = None,
    retain_grad: bool = False,
) -> Iterator[dict[str, Any]]:
    """Patch only the pass-one coda write interface for one sequential forward."""

    attachment = wrapper.stage2b_depth_attachment
    original = attachment.observe
    captured: dict[str, Any] = {}

    def observe(**kwargs: Any):
        trace = original(**kwargs)
        if int(kwargs["loop_index"]) != 0:
            return trace
        hidden = kwargs["coda_hidden"]
        if retain_grad:
            if not hidden.requires_grad:
                hidden.requires_grad_(True)
            hidden.retain_grad()
        captured["hidden"] = hidden
        captured["attention_mask"] = kwargs["attention_mask"]
        if direction is not None:
            deployed, telemetry = scale_external_write(
                hidden, direction.to(hidden.device), kwargs["attention_mask"], gamma=GAMMA
            )
            hidden.copy_(deployed)
            captured["telemetry"] = telemetry
        return trace

    attachment.observe = observe
    try:
        yield captured
    finally:
        attachment.observe = original


def forward_interface(
    wrapper: Any,
    input_ids: torch.Tensor,
    attention: torch.Tensor,
    *,
    direction: torch.Tensor | None = None,
    retain_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    with interface_patch(wrapper, direction=direction, retain_grad=retain_grad) as captured:
        output = wrapper(
            input_ids=input_ids,
            attention_mask=attention,
            max_loops=1,
            stage2b_depth_enabled=True,
            stage2b_stage="M2",
            stage2b_amplitude=0.05,
            stage2b_score_only_sparse_logits=True,
            return_loop_logits=True,
            use_cache=False,
            return_dict=True,
        )
    if "hidden" not in captured or output.loop_logits is None:
        raise RuntimeError("W1 failed to capture the registered write interface")
    return output.loop_logits[:, 0, -1].float(), captured["hidden"], captured


def row_margin(logits: torch.Tensor, first: int, targets: Sequence[int]) -> torch.Tensor:
    selected = logits[first : first + len(targets)]
    target_ids = torch.tensor(targets, device=logits.device)
    gold = selected.gather(-1, target_ids[:, None]).squeeze(-1)
    wrong = selected.clone()
    wrong.scatter_(-1, target_ids[:, None], -torch.inf)
    return gold - wrong.max(dim=-1).values


def extract_student_targets(
    *,
    wrapper: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    allow_dry_neighbor_fallback: bool = False,
) -> dict[str, Any]:
    state_before = _state_digest(_named_trainable_state(wrapper))
    families = {name: [] for name in ("l0a", "l0c", "l0d")}
    states = []
    baseline_margins = []
    solved = []
    started = time.perf_counter()
    device = next(wrapper.parameters()).device
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        input_ids, attention, spans = prepare_batch(tokenizer, batch, device)
        wrapper.zero_grad(set_to_none=True)
        logits, hidden, _capture = forward_interface(
            wrapper, input_ids, attention, retain_grad=True
        )
        ce_losses = []
        margin_scores = []
        row_predictions = []
        for index, (first, targets) in enumerate(spans):
            positions = logits[index, first : first + len(targets)]
            target_ids = torch.tensor(targets, device=device)
            ce_losses.append(F.cross_entropy(positions, target_ids))
            margins = row_margin(logits[index], first, targets)
            margin_scores.append(margins.mean())
            row_predictions.append(positions.argmax(dim=-1))
            baseline_margins.append(float(margins.mean().detach().cpu()))
            solved.append(bool(torch.equal(row_predictions[-1], target_ids)))
        ce_grad = torch.autograd.grad(sum(ce_losses), hidden, retain_graph=True)[0]
        margin_grad = torch.autograd.grad(sum(margin_scores), hidden)[0]

        pseudo = input_ids.detach().clone()
        for index, ((first, targets), predictions) in enumerate(zip(spans, row_predictions)):
            prompt_length = first + 1
            pseudo[index, prompt_length : prompt_length + len(targets)] = predictions
        with torch.inference_mode():
            _free_logits, free_hidden, _free_capture = forward_interface(
                wrapper, pseudo, attention, retain_grad=False
            )
        for index, (first, targets) in enumerate(spans):
            positions = slice(first, first + len(targets))
            gold_state = hidden[index, positions].detach().float().mean(dim=0)
            free_state = free_hidden[index, positions].detach().float().mean(dim=0)
            states.append(gold_state.cpu())
            families["l0a"].append((-ce_grad[index, positions].float().mean(dim=0)).cpu())
            families["l0c"].append((margin_grad[index, positions].float().mean(dim=0)).cpu())
            families["l0d"].append((gold_state - free_state).cpu())
        wrapper.zero_grad(set_to_none=True)
        print(f"w1_target_progress rows={min(start + batch_size, len(rows))}/{len(rows)}", flush=True)

    state_matrix = torch.stack(states)
    solved_mask = torch.tensor(solved, dtype=torch.bool)
    normalized = F.normalize(state_matrix.float(), dim=-1)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-torch.inf)
    l0g = []
    realized_neighbor_k = 8
    for index in range(len(rows)):
        candidates = similarity[index].clone()
        candidates[~solved_mask] = -torch.inf
        available = int(torch.isfinite(candidates).sum())
        if available < 8 and not allow_dry_neighbor_fallback:
            raise RuntimeError(f"W1 neighbor target has only {available} solved candidates")
        neighbor_k = min(8, available)
        if neighbor_k < 1:
            raise RuntimeError("W1 neighbor target has no solved candidate")
        realized_neighbor_k = min(realized_neighbor_k, neighbor_k)
        neighbors = candidates.topk(neighbor_k).indices
        l0g.append(state_matrix[neighbors].mean(dim=0) - state_matrix[index])
    families["l0g"] = l0g
    if _state_digest(_named_trainable_state(wrapper)) != state_before:
        raise RuntimeError("W1 target extraction mutated the model")
    return {
        "kind": "paper2_bicameral_w1_student_targets_v1",
        "item_ids": [str(row["item_id"]) for row in rows],
        "batteries": [str(row["battery"]) for row in rows],
        "families": {name: torch.stack(values) for name, values in families.items()},
        "interface_states": state_matrix,
        "baseline_margins": torch.tensor(baseline_margins),
        "solved_mask": solved_mask,
        "seconds": time.perf_counter() - started,
        "free_run_construction": "single_parallel_greedy_reconstruction_then_second_forward_v1",
        "neighbor_target": {
            "registered_k": 8,
            "realized_k": realized_neighbor_k,
            "dry_run_fallback": realized_neighbor_k < 8,
        },
        "optimizer_constructed": False,
        "optimizer_steps": 0,
    }


@torch.inference_mode()
def extract_teacher_target(
    *,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    student_states: torch.Tensor,
    geometry_path: Path,
    cache_dir: Path,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    geometry = torch.load(geometry_path, map_location="cpu", weights_only=False)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID,
        revision=TEACHER_REVISION,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    targets = []
    started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        input_ids, attention, spans = prepare_batch(tokenizer, batch, torch.device(device))
        output = teacher(
            input_ids=input_ids,
            attention_mask=attention,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.hidden_states[12].float()
        for local, (first, token_ids) in enumerate(spans):
            state = hidden[local, first : first + len(token_ids)].mean(dim=0).cpu()
            teacher_coordinate = (
                (state[None] - geometry["teacher_mean"].float())
                @ geometry["teacher_basis"].float()
            )
            student_coordinate = teacher_coordinate @ geometry["diagnostic_rotation"].float().T
            mapped = (
                geometry["student_mean"].float()
                + student_coordinate @ geometry["student_basis"].float().T
            ).squeeze(0)
            targets.append(mapped - student_states[start + local].float())
        print(f"w1_teacher_progress rows={min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    elapsed = time.perf_counter() - started
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    return torch.stack(targets), {
        "model": TEACHER_ID,
        "revision": TEACHER_REVISION,
        "geometry_sha256": sha256_file(geometry_path),
        "rows": len(rows),
        "seconds": elapsed,
    }


@torch.inference_mode()
def score_arm(
    *,
    wrapper: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    baseline_margins: torch.Tensor,
    directions: torch.Tensor,
    arm: str,
    seed: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    device = next(wrapper.parameters()).device
    scored = []
    ratios = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        input_ids, attention, spans = prepare_batch(tokenizer, batch, device)
        logits, _hidden, captured = forward_interface(
            wrapper,
            input_ids,
            attention,
            direction=directions[start : start + len(batch)].to(device),
        )
        telemetry = captured["telemetry"]
        ratios.extend(float(value) for value in telemetry["write_ratio"].cpu())
        for local, (first, targets) in enumerate(spans):
            margin = float(row_margin(logits[local], first, targets).mean().cpu())
            index = start + local
            scored.append(
                {
                    "kind": "paper2_bicameral_w1_margin_row_v1",
                    "evaluator": EVALUATOR_TAG,
                    "schedule": SCHEDULE,
                    "arm": arm,
                    "seed": seed,
                    "item_id": str(batch[local]["item_id"]),
                    "battery": str(batch[local]["battery"]),
                    "baseline_margin": float(baseline_margins[index]),
                    "injected_margin": margin,
                    "margin_delta": margin - float(baseline_margins[index]),
                    "write_ratio": float(telemetry["write_ratio"][local].cpu()),
                }
            )
    summary = {
        "kind": "paper2_bicameral_w1_margin_cell_v1",
        "arm": arm,
        "seed": seed,
        "evaluator": EVALUATOR_TAG,
        "schedule": SCHEDULE,
        "gamma": GAMMA,
        "seconds": time.perf_counter() - started,
        "write_ratio_mean": sum(ratios) / len(ratios),
        "write_ratio_max_abs_error": max(abs(value - GAMMA) for value in ratios),
        **summarize_margin_deltas(scored),
    }
    return scored, summary


def build_controls(families: Mapping[str, torch.Tensor], *, seed: int) -> dict[str, torch.Tensor]:
    result = {name: value.float() for name, value in families.items()}
    for family in ("l0a", "l0b", "l0c", "l0d", "l0g"):
        if family not in result:
            continue
        permutation = deterministic_permutation(len(result[family]), family=f"{family}:seed{seed}")
        result[f"l5_{family[-1]}"] = result[family][permutation]
    generator = torch.Generator().manual_seed(20260824 + seed)
    result["l4"] = torch.randn(result["l0a"].shape, generator=generator)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_dev2(args.dev2_manifest, args.reference_rows)
    if args.mode == "dry_run":
        rows = dry_rows(rows, args.dry_rows)
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
    initial_digest = _state_digest(_named_trainable_state(wrapper))
    target_path = args.private_dir / f"seed_{args.seed}_{args.mode}_student_targets.pt"
    if target_path.is_file():
        targets = torch.load(target_path, map_location="cpu", weights_only=False)
        if targets["item_ids"] != [str(row["item_id"]) for row in rows]:
            raise RuntimeError("W1 resumed target population changed")
    else:
        targets = extract_student_targets(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=rows,
            batch_size=args.target_batch_size,
            allow_dry_neighbor_fallback=args.mode == "dry_run",
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(targets, target_path)

    teacher_rows = rows if args.mode == "dry_run" else rows[: args.teacher_rows]
    del wrapper, _groups
    gc.collect()
    torch.cuda.empty_cache()
    if (
        "l0b" in targets["families"]
        and int(targets.get("teacher_population_rows", -1)) == len(teacher_rows)
        and isinstance(targets.get("teacher_receipt"), Mapping)
    ):
        teacher_receipt = dict(targets["teacher_receipt"])
        teacher_target = targets["families"]["l0b"]
    else:
        student_subset = targets["interface_states"][: len(teacher_rows)]
        teacher_target, teacher_receipt = extract_teacher_target(
            tokenizer=tokenizer,
            rows=teacher_rows,
            student_states=student_subset,
            geometry_path=args.geometry,
            cache_dir=args.teacher_cache,
            batch_size=args.teacher_batch_size,
            device=args.device,
        )
        targets["families"]["l0b"] = teacher_target
        targets["teacher_population_rows"] = len(teacher_rows)
        targets["teacher_receipt"] = teacher_receipt
        torch.save(targets, target_path)

    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    torch.cuda.manual_seed_all(initialization_seed)
    wrapper, _chain_again, _groups = _build_model(args)
    if _state_digest(_named_trainable_state(wrapper)) != initial_digest:
        raise RuntimeError("W1 model reconstruction changed the initialization state")
    controls = build_controls(targets["families"], seed=args.seed)
    cells = []
    arms = ["l0a", "l0c", "l0d", "l0g", "l5_a", "l5_c", "l5_d", "l5_g", "l4"]
    if len(teacher_rows) == len(rows):
        arms.extend(["l0b", "l5_b"])
    for arm in arms:
        row_path = args.private_dir / f"seed_{args.seed}_{args.mode}_{arm}.jsonl"
        summary_path = args.output_dir / f"seed_{args.seed}_{args.mode}_{arm}.json"
        if summary_path.is_file() and row_path.is_file():
            cells.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        scored, summary = score_arm(
            wrapper=wrapper,
            tokenizer=tokenizer,
            rows=rows,
            baseline_margins=targets["baseline_margins"],
            directions=controls[arm],
            arm=arm,
            seed=args.seed,
            batch_size=args.margin_batch_size,
        )
        write_jsonl(row_path, scored)
        atomic_json(summary_path, summary)
        cells.append(summary)

    target_seconds_per_row = {
        "student_families": float(targets["seconds"]) / len(rows),
        "teacher_l0b": float(teacher_receipt["seconds"]) / len(teacher_rows),
    }
    margin_seconds_per_row = sum(float(cell["seconds"]) for cell in cells) / (
        len(cells) * len(rows)
    )
    cost = project_cost_hours(
        target_seconds_per_row=target_seconds_per_row,
        margin_seconds_per_row=margin_seconds_per_row,
        rows=2048,
        seeds=2,
        phase_a_cells_per_seed=11,
        phase_b_cells_per_seed=4,
    )
    result = {
        "kind": "paper2_bicameral_w1_seed_v1",
        "status": f"{args.mode}_complete_score_only",
        "seed": args.seed,
        "rows": len(rows),
        "teacher_rows": len(teacher_rows),
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
        "initialization_state_digest": initial_digest,
        "initialization_seed": initialization_seed,
        "target_seconds_per_row": target_seconds_per_row,
        "margin_seconds_per_row": margin_seconds_per_row,
        "cost_projection": cost,
        "cells": cells,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / f"seed_{args.seed}_{args.mode}_summary.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--mode", choices=("dry_run", "full"), required=True)
    result.add_argument("--seed", type=int, choices=(0, 1), required=True)
    result.add_argument("--dev2_manifest", type=Path, required=True)
    result.add_argument("--reference_rows", type=Path, required=True)
    result.add_argument("--geometry", type=Path, required=True)
    result.add_argument("--output_dir", type=Path, required=True)
    result.add_argument("--private_dir", type=Path, required=True)
    result.add_argument("--model_cache", type=Path, required=True)
    result.add_argument("--teacher_cache", type=Path, required=True)
    result.add_argument("--dry_rows", type=int, default=16)
    result.add_argument("--teacher_rows", type=int, default=256)
    result.add_argument("--target_batch_size", type=int, default=2)
    result.add_argument("--margin_batch_size", type=int, default=4)
    result.add_argument("--teacher_batch_size", type=int, default=1)
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
