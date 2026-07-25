"""Post-hoc read-only localization of the T1-lite raw/EMA divergence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_identity import model_load_kwargs  # noqa: E402
from eval.eval_internal_think_token_t1_lite import evaluate_rows  # noqa: E402
from training.internal_think_token_runtime import (  # noqa: E402
    install_internal_control_tokens,
    split_internal_control_token_rows,
)
from training.run_internal_think_token_t1_lite import DeviceEMA  # noqa: E402
from training.run_internal_think_token_p0_cell import read_jsonl  # noqa: E402
from training.train_unfrozen_recurrent import prepare_wrapper  # noqa: E402


GROUPS = ("control_rows", "bridge", "recurrent_block")
RECURRENT_LAYER_GROUPS = ("early_6_9", "middle_10_13", "late_14_17")
INTERPOLATION_ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
STAGE_SUPPORT = {500: 1, 2500: 2, 6500: 4, 8500: 8}
MIN_CHECKPOINT_BYTES = 1024


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stage_checkpoint_coverage(progress_dir: str | Path) -> dict[str, Any]:
    root = Path(progress_dir)
    names = [f"t1_progress_step_{step}.pt" for step in STAGE_SUPPORT]
    available = [
        name
        for name in names
        if (root / name).is_file()
        and (root / name).stat().st_size >= MIN_CHECKPOINT_BYTES
    ]
    missing = [name for name in names if name not in available]
    return {
        "required": len(names),
        "available": len(available),
        "available_names": available,
        "missing_names": missing,
        "complete": not missing,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_group(name: str) -> str:
    if name.endswith("control_rows"):
        return "control_rows"
    if name.startswith("bridge."):
        return "bridge"
    if name.startswith("base_model.model.layers."):
        return "recurrent_block"
    return "other"


def recurrent_layer_group(name: str) -> str | None:
    prefix = "base_model.model.layers."
    if not name.startswith(prefix):
        return None
    try:
        layer = int(name[len(prefix) :].split(".", 1)[0])
    except (TypeError, ValueError):
        return None
    if 6 <= layer <= 9:
        return "early_6_9"
    if 10 <= layer <= 13:
        return "middle_10_13"
    if 14 <= layer <= 17:
        return "late_14_17"
    return None


def validate_state_pair(
    raw: dict[str, torch.Tensor],
    ema: dict[str, torch.Tensor],
) -> dict[str, Any]:
    raw_names = set(raw)
    ema_names = set(ema)
    shape_mismatches = sorted(
        name for name in raw_names & ema_names if tuple(raw[name].shape) != tuple(ema[name].shape)
    )
    nonfinite_raw = sorted(name for name, value in raw.items() if not torch.isfinite(value).all())
    nonfinite_ema = sorted(name for name, value in ema.items() if not torch.isfinite(value).all())
    grouped = {group: sum(parameter_group(name) == group for name in raw) for group in (*GROUPS, "other")}
    passed = (
        raw_names == ema_names
        and not shape_mismatches
        and not nonfinite_raw
        and not nonfinite_ema
        and grouped["other"] == 0
    )
    return {
        "passed": passed,
        "raw_only": sorted(raw_names - ema_names),
        "ema_only": sorted(ema_names - raw_names),
        "shape_mismatches": shape_mismatches,
        "nonfinite_raw": nonfinite_raw,
        "nonfinite_ema": nonfinite_ema,
        "tensor_counts": grouped,
    }


def state_geometry(
    raw: dict[str, torch.Tensor],
    ema: dict[str, torch.Tensor],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in GROUPS:
        raw_sq = ema_sq = diff_sq = dot = 0.0
        count = 0
        for name in raw:
            if parameter_group(name) != group:
                continue
            left = raw[name].float()
            right = ema[name].float()
            raw_sq += float(left.square().sum().item())
            ema_sq += float(right.square().sum().item())
            diff_sq += float((left - right).square().sum().item())
            dot += float((left * right).sum().item())
            count += int(left.numel())
        raw_norm = math.sqrt(raw_sq)
        ema_norm = math.sqrt(ema_sq)
        diff_norm = math.sqrt(diff_sq)
        output[group] = {
            "parameters": count,
            "raw_norm": raw_norm,
            "ema_norm": ema_norm,
            "difference_norm": diff_norm,
            "difference_over_raw_norm": diff_norm / raw_norm if raw_norm else None,
            "cosine": dot / (raw_norm * ema_norm) if raw_norm and ema_norm else None,
        }
    return output


def blend_states(
    raw: dict[str, torch.Tensor],
    ema: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError("interpolation alpha must be in [0, 1]")
    return {
        name: ((1.0 - value) * raw[name].float() + value * ema[name].float()).to(raw[name].dtype)
        for name in raw
    }


def swap_group(
    base: dict[str, torch.Tensor],
    donor: dict[str, torch.Tensor],
    group: str,
) -> dict[str, torch.Tensor]:
    if group not in GROUPS:
        raise ValueError(f"unknown parameter group: {group}")
    return {
        name: (donor[name] if parameter_group(name) == group else value)
        for name, value in base.items()
    }


def swap_recurrent_layer_group(
    base: dict[str, torch.Tensor],
    donor: dict[str, torch.Tensor],
    group: str,
) -> dict[str, torch.Tensor]:
    if group not in RECURRENT_LAYER_GROUPS:
        raise ValueError(f"unknown recurrent layer group: {group}")
    return {
        name: (donor[name] if recurrent_layer_group(name) == group else value)
        for name, value in base.items()
    }


def restore_state(wrapper: Any, state: dict[str, torch.Tensor]) -> None:
    current = dict(wrapper.named_parameters())
    missing = sorted(set(state) - set(current))
    if missing:
        raise RuntimeError(f"audit state keys unavailable in wrapper: {missing[:8]}")
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))


def fixed_screen_rows(rows: list[dict[str, Any]], *, seed: int, per_depth: int) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    selected: list[dict[str, Any]] = []
    for depth in sorted({int(row["depth"]) for row in rows}):
        candidates = [row for row in rows if int(row["depth"]) == depth]
        if len(candidates) < int(per_depth):
            raise ValueError(f"depth {depth} has {len(candidates)} rows, needs {per_depth}")
        selected.extend(rng.sample(candidates, int(per_depth)))
    return sorted(selected, key=lambda row: (int(row["depth"]), str(row.get("id") or "")))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def compact_metrics(result: dict[str, Any]) -> dict[str, Any]:
    control = result["control"]
    return {
        "total": int(result["total"]),
        "forced_correct": int(result["forced_correct"]),
        "forced_accuracy": float(result["forced_correct"]) / int(result["total"]),
        "self_halted_correct": int(result["self_halted_correct"]),
        "self_halted_accuracy": float(result["self_halted_correct"]) / int(result["total"]),
        "exact_selected_depth_correct": int(control["exact_selected_depth_correct"]),
        "exact_selected_depth_accuracy": float(control["exact_selected_depth_accuracy"]),
        "continue_recall": float(control["continue_recall"]),
        "stop_recall": float(control["stop_recall"]),
        "by_depth": control["by_depth"],
    }


def scalar_ema_integrity() -> dict[str, Any]:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    ema = DeviceEMA([("probe", parameter)], decay=0.9)
    with torch.no_grad():
        parameter.fill_(2.0)
    ema.update([("probe", parameter)])
    with torch.no_grad():
        parameter.fill_(4.0)
    ema.update([("probe", parameter)])
    observed = float(ema.shadow["probe"].item())
    expected = 0.9 * (0.9 * 1.0 + 0.1 * 2.0) + 0.1 * 4.0
    return {
        "passed": abs(observed - expected) <= 1e-7,
        "observed": observed,
        "expected": expected,
        "absolute_error": abs(observed - expected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_checkpoint", required=True)
    parser.add_argument("--ema_checkpoint", required=True)
    parser.add_argument("--progress_dir", required=True)
    parser.add_argument("--archived_stage_receipts_dir")
    parser.add_argument("--allow_missing_stage_checkpoints", action="store_true")
    parser.add_argument("--pilot_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_raw_sha256", required=True)
    parser.add_argument("--expected_ema_sha256", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(args.raw_checkpoint)
    ema_path = Path(args.ema_checkpoint)
    raw_sha = sha256_file(raw_path)
    ema_sha = sha256_file(ema_path)
    if raw_sha != args.expected_raw_sha256 or ema_sha != args.expected_ema_sha256:
        raise RuntimeError(
            f"endpoint checkpoint hash mismatch: raw={raw_sha}, ema={ema_sha}"
        )
    raw_payload = torch.load(raw_path, map_location="cpu")
    ema_payload = torch.load(ema_path, map_location="cpu")
    raw_state = raw_payload["trainable_state_dict"]
    ema_state = ema_payload["trainable_state_dict"]
    state_integrity = validate_state_pair(raw_state, ema_state)
    if not state_integrity["passed"]:
        raise RuntimeError(f"endpoint state integrity failed: {state_integrity}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_load_kwargs(args.dtype, args.attn_implementation),
    ).to(args.device)
    resize = install_internal_control_tokens(tokenizer, model)
    split_internal_control_token_rows(model, original_vocab_size=resize.original_vocab_size)
    config = {
        "layer_split": "6,18",
        "initial_halt_prob": 0.15,
        "bridge_projection_mode": "split",
        "adapter_dtype": "float32",
        "training_mode": "full_block",
        "resume_lora": {"enabled": False},
        "merge_lora_before_unfreeze": False,
        "train_auxiliary": {
            "bridge": True,
            "halting": False,
            "reentry_adapter": False,
            "latent": False,
        },
    }
    wrapper, _ = prepare_wrapper(model, config, device=args.device)
    wrapper.base_model.get_input_embeddings().control_rows.requires_grad_(True)
    continue_id, stop_id, readout_id = (int(value) for value in resize.control_token_ids)

    pilot_rows = read_jsonl(args.pilot_jsonl)
    screen_rows = fixed_screen_rows(pilot_rows, seed=args.seed, per_depth=8)
    screen_path = output_dir / "data" / "screen_64.jsonl"
    full_path = output_dir / "data" / "pilot_256.jsonl"
    write_jsonl(screen_path, screen_rows)
    write_jsonl(full_path, pilot_rows)

    def evaluate_variant(
        name: str,
        state: dict[str, torch.Tensor],
        data_path: Path,
        *,
        max_loops: int,
    ) -> dict[str, Any]:
        restore_state(wrapper, state)
        result = evaluate_rows(
            wrapper,
            tokenizer,
            data_path,
            device=args.device,
            max_loops=max_loops,
            batch_size=args.batch_size,
            continue_id=continue_id,
            stop_id=stop_id,
            readout_id=readout_id,
            include_features=False,
        )
        destination = output_dir / "variants" / name
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "rows.json", result["rows"])
        metrics = compact_metrics(result)
        write_json(destination / "summary.json", metrics)
        print(json.dumps({"variant": name, **metrics}, sort_keys=True), flush=True)
        return metrics

    progress_dir = Path(args.progress_dir)
    checkpoint_coverage = stage_checkpoint_coverage(progress_dir)
    if not checkpoint_coverage["complete"] and not args.allow_missing_stage_checkpoints:
        raise FileNotFoundError(
            "missing stage progress checkpoints: "
            + ", ".join(checkpoint_coverage["missing_names"])
        )
    stage_results: dict[str, Any] = {}
    unreadable_stage_checkpoints: dict[str, str] = {}
    archived_receipts = (
        Path(args.archived_stage_receipts_dir)
        if args.archived_stage_receipts_dir
        else None
    )
    for step, support in STAGE_SUPPORT.items():
        progress_path = progress_dir / f"t1_progress_step_{step}.pt"
        if not progress_path.exists():
            receipt_path = (
                archived_receipts / f"step_{step}.json"
                if archived_receipts is not None
                else None
            )
            stage_results[str(step)] = {
                "trained_support": support,
                "checkpoint_available": False,
                "raw_ema_comparison_available": False,
                "archived_raw_boundary_receipt": (
                    json.loads(receipt_path.read_text(encoding="utf-8"))
                    if receipt_path is not None and receipt_path.exists()
                    else None
                ),
            }
            continue
        try:
            payload = torch.load(progress_path, map_location="cpu")
            if not isinstance(payload, dict):
                raise ValueError("stage checkpoint payload is not a dictionary")
            if "trainable_state_dict" not in payload or "ema_state_dict" not in payload:
                raise KeyError("stage checkpoint lacks raw or EMA state")
        except (EOFError, OSError, RuntimeError, ValueError, KeyError, pickle.UnpicklingError) as error:
            if not args.allow_missing_stage_checkpoints:
                raise
            unreadable_stage_checkpoints[progress_path.name] = (
                f"{type(error).__name__}: {error}"
            )
            receipt_path = (
                archived_receipts / f"step_{step}.json"
                if archived_receipts is not None
                else None
            )
            stage_results[str(step)] = {
                "trained_support": support,
                "checkpoint_available": False,
                "raw_ema_comparison_available": False,
                "checkpoint_load_error": unreadable_stage_checkpoints[progress_path.name],
                "archived_raw_boundary_receipt": (
                    json.loads(receipt_path.read_text(encoding="utf-8"))
                    if receipt_path is not None and receipt_path.exists()
                    else None
                ),
            }
            continue
        stage_raw = payload["trainable_state_dict"]
        stage_ema = payload["ema_state_dict"]["shadow"]
        stage_integrity = validate_state_pair(stage_raw, stage_ema)
        if not stage_integrity["passed"]:
            raise RuntimeError(f"stage {step} state integrity failed: {stage_integrity}")
        stage_rows = [row for row in screen_rows if int(row["depth"]) <= support]
        stage_path = output_dir / "data" / f"stage_{step}_screen.jsonl"
        write_jsonl(stage_path, stage_rows)
        stage_results[str(step)] = {
            "trained_support": support,
            "checkpoint_available": True,
            "raw_ema_comparison_available": True,
            "integrity": stage_integrity,
            "raw": evaluate_variant(f"stage_{step}_raw", stage_raw, stage_path, max_loops=support),
            "ema": evaluate_variant(f"stage_{step}_ema", stage_ema, stage_path, max_loops=support),
        }
        del payload, stage_raw, stage_ema
        gc.collect()

    if unreadable_stage_checkpoints:
        available_names = [
            name
            for name in checkpoint_coverage["available_names"]
            if name not in unreadable_stage_checkpoints
        ]
        missing_names = sorted(
            set(checkpoint_coverage["missing_names"]) | set(unreadable_stage_checkpoints)
        )
        checkpoint_coverage.update(
            {
                "available": len(available_names),
                "available_names": available_names,
                "missing_names": missing_names,
                "unreadable_names": unreadable_stage_checkpoints,
                "complete": False,
            }
        )

    interpolation: dict[str, Any] = {}
    for alpha in INTERPOLATION_ALPHAS:
        state = blend_states(raw_state, ema_state, alpha)
        interpolation[str(alpha)] = evaluate_variant(
            f"interpolation_alpha_{str(alpha).replace('.', 'p')}",
            state,
            screen_path,
            max_loops=8,
        )
        del state
        gc.collect()

    interpolation_by_depth = {
        alpha: {
            depth: {
                "exact_selected_depth_correct": int(values["correct"]),
                "total": int(values["total"]),
                "accuracy": float(values["accuracy"]),
            }
            for depth, values in metrics["by_depth"].items()
        }
        for alpha, metrics in interpolation.items()
    }

    swaps: dict[str, Any] = {}
    for group in GROUPS:
        raw_name = f"raw_with_ema_{group}"
        ema_name = f"ema_with_raw_{group}"
        raw_damaged = swap_group(raw_state, ema_state, group)
        ema_rescued = swap_group(ema_state, raw_state, group)
        swaps[raw_name] = evaluate_variant(raw_name, raw_damaged, screen_path, max_loops=8)
        swaps[ema_name] = evaluate_variant(ema_name, ema_rescued, screen_path, max_loops=8)
        del raw_damaged, ema_rescued
        gc.collect()

    rescue_name = max(
        (name for name in swaps if name.startswith("ema_with_raw_")),
        key=lambda name: (
            swaps[name]["exact_selected_depth_accuracy"],
            swaps[name]["forced_accuracy"],
        ),
    )
    damage_name = min(
        (name for name in swaps if name.startswith("raw_with_ema_")),
        key=lambda name: (
            swaps[name]["exact_selected_depth_accuracy"],
            swaps[name]["forced_accuracy"],
        ),
    )
    rescue_group = rescue_name.removeprefix("ema_with_raw_")
    damage_group = damage_name.removeprefix("raw_with_ema_")
    confirmations = {
        rescue_name: evaluate_variant(
            f"confirmation_{rescue_name}",
            swap_group(ema_state, raw_state, rescue_group),
            full_path,
            max_loops=8,
        ),
        damage_name: evaluate_variant(
            f"confirmation_{damage_name}",
            swap_group(raw_state, ema_state, damage_group),
            full_path,
            max_loops=8,
        ),
    }

    layer_group_swaps: dict[str, Any] = {}
    for group in RECURRENT_LAYER_GROUPS:
        raw_name = f"raw_with_ema_{group}"
        ema_name = f"ema_with_raw_{group}"
        layer_group_swaps[raw_name] = evaluate_variant(
            raw_name,
            swap_recurrent_layer_group(raw_state, ema_state, group),
            screen_path,
            max_loops=8,
        )
        layer_group_swaps[ema_name] = evaluate_variant(
            ema_name,
            swap_recurrent_layer_group(ema_state, raw_state, group),
            screen_path,
            max_loops=8,
        )
        gc.collect()

    summary = {
        "kind": "paper2_t1_lite_ema_posthoc_audit",
        "status": (
            "finished"
            if checkpoint_coverage["complete"]
            else "finished_partial_missing_stage_checkpoints"
        ),
        "registered_verdict_immutable": "registered_negative",
        "training_performed": False,
        "checkpoint_mutation": False,
        "seed_1_launched": False,
        "inputs": {
            "raw_checkpoint": str(raw_path),
            "raw_sha256": raw_sha,
            "ema_checkpoint": str(ema_path),
            "ema_sha256": ema_sha,
            "pilot_jsonl": args.pilot_jsonl,
            "screen_seed": args.seed,
            "screen_rows": len(screen_rows),
        },
        "integrity": {
            "checkpoint_kinds": {
                "raw": raw_payload.get("kind"),
                "ema": ema_payload.get("kind"),
            },
            "checkpoint_steps": {
                "raw": raw_payload.get("step"),
                "ema": ema_payload.get("step"),
            },
            "state_pair": state_integrity,
            "scalar_ema_recurrence": scalar_ema_integrity(),
        },
        "geometry": state_geometry(raw_state, ema_state),
        "stage_checkpoint_coverage": checkpoint_coverage,
        "stage_boundaries": stage_results,
        "interpolation": interpolation,
        "interpolation_by_depth": interpolation_by_depth,
        "group_swaps_screen": swaps,
        "recurrent_layer_group_swaps_screen": layer_group_swaps,
        "full_pilot_confirmations": confirmations,
        "selected_confirmation_variants": {
            "strongest_ema_rescue": rescue_name,
            "strongest_raw_damage": damage_name,
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
