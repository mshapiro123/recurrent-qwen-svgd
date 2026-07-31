"""Run the no-training Phase-2 V1 expressivity and V2 iteration-gain checks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import load_drafter, read_jsonl  # noqa: E402
from eval.eval_paper2_dc0_depth_by_append import evaluate_append_arm  # noqa: E402
from eval.eval_reentry_drift import prepare_recurrent_inputs, run_recurrent_block  # noqa: E402
from eval.eval_speculative_depth_d0_floor import load_partition_cache  # noqa: E402
from models.coconut_composite import CoconutRecurrentQwen  # noqa: E402
from training.speculative_depth_d0_corpus import sha256_file  # noqa: E402


def quantile_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "count": len(values),
        "min": float(tensor.min()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "median": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
    }


def finite_difference_directional_gain(
    function: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    direction: torch.Tensor,
    *,
    epsilon: float,
) -> float:
    if not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    direction_norm = direction.float().norm().clamp_min(1e-12)
    unit = direction / direction_norm.to(dtype=direction.dtype)
    with torch.no_grad():
        plus = function(state + float(epsilon) * unit).float()
        minus = function(state - float(epsilon) * unit).float()
    derivative = (plus - minus) / (2.0 * float(epsilon))
    return float(derivative.norm().cpu())


def bound_compatible_fractions(
    *,
    margins: Sequence[float],
    sampled_max_gains: Sequence[float],
    state_rms: Sequence[float],
    hidden_size: int,
    c_values: Sequence[float],
    gamma: float,
    rho: float,
) -> dict[str, Any]:
    if not (len(margins) == len(sampled_max_gains) == len(state_rms)):
        raise ValueError("position-matched bound inputs must have equal lengths")
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1)")
    result: dict[str, Any] = {}
    for c_value in c_values:
        bounds = [
            gain
            * float(gamma)
            * float(c_value)
            * rms
            * math.sqrt(hidden_size)
            / (1.0 - float(rho))
            for gain, rms in zip(sampled_max_gains, state_rms)
        ]
        compatible = sum(
            float(margin) <= float(bound)
            for margin, bound in zip(margins, bounds)
        )
        result[f"{float(c_value):g}"] = {
            "positions": len(margins),
            "compatible": int(compatible),
            "fraction": compatible / len(margins) if margins else None,
            "empirical_bound": quantile_summary(bounds),
        }
    return result


def _stable_key(row_id: str, position: int = -1) -> str:
    return hashlib.sha256(f"phase2:{row_id}:{position}".encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _upper_stack_logits(
    wrapper: Any,
    hidden: torch.Tensor,
    *,
    causal_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    position_embeddings: Any,
    vocab_size: int,
) -> torch.Tensor:
    block = run_recurrent_block(
        wrapper,
        hidden,
        causal_mask,
        position_ids,
        cache_position,
        position_embeddings,
    )
    coda, _ = wrapper._run_layer_range(  # noqa: SLF001
        start=wrapper.layer_split.recurrent_end,
        end=len(wrapper.qwen.layers),
        hidden_states=block,
        causal_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        collect_hidden=False,
        hidden_history=None,
    )
    return wrapper.lm_head(wrapper.qwen.norm(coda))[:, -1, :vocab_size].float()


def _random_local_direction(state: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    local = torch.randn(
        state.shape[-1], generator=generator, dtype=torch.float32
    ).to(device=state.device, dtype=state.dtype)
    direction = torch.zeros_like(state)
    direction[:, -1] = local / local.float().norm().clamp_min(1e-12).to(local.dtype)
    return direction


def _load_stage_a_bridge(composite: CoconutRecurrentQwen, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("horizontal_bridge")
    if not isinstance(state, dict):
        raise RuntimeError("Stage A checkpoint lacks horizontal_bridge state")
    composite.horizontal_bridge.load_state_dict(state)


def _baseline_help_records(
    *,
    wrapper: Any,
    composite: CoconutRecurrentQwen,
    rows: Sequence[dict[str, Any]],
    teacher_rows: Sequence[torch.Tensor],
    vocab_size: int,
    device: str,
    cache_dir: Path,
    append_batch_size: int,
) -> list[dict[str, Any]]:
    append_grids, _ = evaluate_append_arm(
        composite,
        rows,
        arm="trained_append_k1",
        feedback_mode="raw",
        reference_rms=None,
        neutral_token_id=int(composite.latent_token_id),
        vocab_size=vocab_size,
        device=device,
        batch_size=append_batch_size,
        resume_dir=cache_dir / "append_predictions",
        append_steps=1,
    )
    records: list[dict[str, Any]] = []
    for row_index, (row, teacher, grid) in enumerate(
        zip(rows, teacher_rows, append_grids)
    ):
        values = torch.tensor([row["input_ids"]], device=device)
        with torch.inference_mode():
            output = wrapper(
                input_ids=values,
                attention_mask=torch.ones_like(values),
                max_loops=1,
                use_cache=False,
                return_dict=True,
            )
            logits = output.logits[0, :-1, :vocab_size].float().cpu()
        baseline = logits.argmax(dim=-1)
        candidate = grid[:, 1].long()
        target = teacher.long()
        if not (baseline.shape == candidate.shape == target.shape):
            raise ValueError("V1 baseline, append, and teacher positions are misaligned")
        helps = baseline.ne(target) & candidate.eq(target)
        for position in torch.where(helps)[0].tolist():
            pred_id = int(baseline[position])
            target_id = int(target[position])
            records.append(
                {
                    "row_index": row_index,
                    "position": int(position),
                    "baseline_token_id": pred_id,
                    "teacher_token_id": target_id,
                    "margin": float(
                        logits[position, pred_id] - logits[position, target_id]
                    ),
                    "sort_key": _stable_key(str(row["row_id"]), int(position)),
                }
            )
        if row_index == 0 or (row_index + 1) % 16 == 0:
            print(
                f"phase2_v1_help_scan rows={row_index + 1}/{len(rows)} "
                f"helps={len(records)}",
                flush=True,
            )
    return records


def _v1_probe_position(
    *,
    wrapper: Any,
    row: dict[str, Any],
    record: dict[str, Any],
    vocab_size: int,
    random_probes: int,
    epsilon_fraction: float,
) -> dict[str, Any]:
    position = int(record["position"])
    values = torch.tensor(
        [row["input_ids"][: position + 1]], device=next(wrapper.parameters()).device
    )
    attention = torch.ones_like(values)
    hidden, _mask, causal_mask, position_ids, cache_position, rotary = (
        prepare_recurrent_inputs(wrapper, values, attention)
    )
    hidden = hidden.detach()
    local = hidden[:, -1].float()
    state_rms = float(local.square().mean().sqrt().cpu())
    epsilon = float(epsilon_fraction) * state_rms * math.sqrt(hidden.shape[-1])

    def upper(value: torch.Tensor) -> torch.Tensor:
        return _upper_stack_logits(
            wrapper,
            value,
            causal_mask=causal_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=rotary,
            vocab_size=vocab_size,
        )[0]

    gains = []
    for probe in range(random_probes):
        direction = _random_local_direction(
            hidden, seed=31_000_003 + int(record["row_index"]) * 97 + position * 13 + probe
        )
        gains.append(
            finite_difference_directional_gain(
                upper, hidden, direction, epsilon=epsilon
            )
        )

    differentiable = hidden.detach().requires_grad_(True)
    logits = upper(differentiable)
    margin_tensor = (
        logits[int(record["baseline_token_id"])]
        - logits[int(record["teacher_token_id"])]
    )
    gradient = torch.autograd.grad(margin_tensor, differentiable)[0]
    targeted_gradient_l2 = float(gradient[:, -1].float().norm().cpu())
    return {
        "row_index": int(record["row_index"]),
        "position": position,
        "margin": float(record["margin"]),
        "state_rms": state_rms,
        "epsilon": epsilon,
        "directional_gains": gains,
        "sampled_max_gain": max(gains),
        "targeted_margin_gradient_l2": targeted_gradient_l2,
    }


def _v2_checkpoint(
    *,
    checkpoint: Path,
    rows: Sequence[dict[str, Any]],
    expected_sha256: str,
    device: str,
    random_probes: int,
    epsilon_fraction: float,
    private_dir: Path,
) -> dict[str, Any]:
    if sha256_file(checkpoint) != expected_sha256:
        raise RuntimeError("V2 checkpoint SHA-256 mismatch")
    _tokenizer, wrapper, _resize, _vocab = load_drafter(
        checkpoint=checkpoint,
        device=device,
        dtype="float32",
        attn_implementation="sdpa",
    )
    wrapper.eval()
    by_iterate: dict[int, list[float]] = {index: [] for index in range(1, 5)}
    for row_number, row in enumerate(rows):
        cache_path = private_dir / f"row_{row_number:04d}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            values = torch.tensor([row["input_ids"]], device=device)
            attention = torch.ones_like(values)
            state, _mask, causal_mask, position_ids, cache_position, rotary = (
                prepare_recurrent_inputs(wrapper, values, attention)
            )
            state = state.detach()
            row_gains: dict[str, list[float]] = {}
            for iterate in range(1, 5):
                local_rms = float(state[:, -1].float().square().mean().sqrt().cpu())
                epsilon = (
                    float(epsilon_fraction)
                    * local_rms
                    * math.sqrt(state.shape[-1])
                )

                def block(value: torch.Tensor) -> torch.Tensor:
                    return run_recurrent_block(
                        wrapper,
                        value,
                        causal_mask,
                        position_ids,
                        cache_position,
                        rotary,
                    )[:, -1].float()

                gains = []
                for probe in range(random_probes):
                    direction = _random_local_direction(
                        state,
                        seed=41_000_009 + row_number * 101 + iterate * 17 + probe,
                    )
                    gains.append(
                        finite_difference_directional_gain(
                            block, state, direction, epsilon=epsilon
                        )
                    )
                row_gains[str(iterate)] = gains
                with torch.no_grad():
                    state = run_recurrent_block(
                        wrapper,
                        state,
                        causal_mask,
                        position_ids,
                        cache_position,
                        rotary,
                    ).detach()
            payload = {"row_number": row_number, "gains": row_gains}
            _write_json(cache_path, payload)
        for iterate in range(1, 5):
            by_iterate[iterate].extend(
                float(value) for value in payload["gains"][str(iterate)]
            )
        print(
            f"phase2_v2_progress checkpoint={expected_sha256[:12]} "
            f"rows={row_number + 1}/{len(rows)}",
            flush=True,
        )
    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "checkpoint_sha256": expected_sha256,
        "rows": len(rows),
        "random_probes_per_row_iterate": random_probes,
        "by_iterate": {
            str(iterate): quantile_summary(values)
            for iterate, values in by_iterate.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--teacher_cache_summary", required=True)
    parser.add_argument("--post_checkpoint", required=True)
    parser.add_argument("--post_checkpoint_sha256", required=True)
    parser.add_argument("--pre_checkpoint", required=True)
    parser.add_argument("--pre_checkpoint_sha256", required=True)
    parser.add_argument("--trained_bridge_checkpoint", required=True)
    parser.add_argument("--trained_bridge_sha256", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--private_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_rows", type=int, default=128)
    parser.add_argument("--max_v1_positions", type=int, default=128)
    parser.add_argument("--max_v2_rows", type=int, default=32)
    parser.add_argument("--random_probes", type=int, default=2)
    parser.add_argument("--epsilon_fraction", type=float, default=0.01)
    parser.add_argument("--append_batch_size", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.8)
    args = parser.parse_args()

    for path, expected, label in (
        (args.post_checkpoint, args.post_checkpoint_sha256, "post-D0"),
        (args.pre_checkpoint, args.pre_checkpoint_sha256, "pre-D0"),
        (args.trained_bridge_checkpoint, args.trained_bridge_sha256, "Stage A bridge"),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"{label} checkpoint SHA-256 mismatch")
    all_rows = read_jsonl(args.data_jsonl)
    selected_indices = sorted(
        range(len(all_rows)),
        key=lambda index: _stable_key(str(all_rows[index]["row_id"])),
    )[: args.max_rows]
    rows = [all_rows[index] for index in selected_indices]
    teacher_summary = json.loads(
        Path(args.teacher_cache_summary).read_text(encoding="utf-8")
    )
    teacher_cache = load_partition_cache(teacher_summary, "teacher_7b", "dev_c")
    teacher_rows = [
        teacher_cache[index]["teacher_greedy_token_id"].long()
        for index in selected_indices
    ]
    private = Path(args.private_dir)
    private.mkdir(parents=True, exist_ok=True)
    private_config = {
        "kind": "paper2_phase2_v1_v2_private_config",
        "data_jsonl_sha256": sha256_file(args.data_jsonl),
        "selected_indices_sha256": hashlib.sha256(
            json.dumps(selected_indices, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "selected_rows": len(selected_indices),
        "max_v1_positions": args.max_v1_positions,
        "max_v2_rows": args.max_v2_rows,
        "random_probes": args.random_probes,
        "epsilon_fraction": args.epsilon_fraction,
        "append_batch_size": args.append_batch_size,
        "gamma": args.gamma,
        "rho": args.rho,
        "post_checkpoint_sha256": args.post_checkpoint_sha256,
        "pre_checkpoint_sha256": args.pre_checkpoint_sha256,
        "trained_bridge_sha256": args.trained_bridge_sha256,
    }
    private_config_path = private / "config.json"
    if private_config_path.exists():
        observed_config = json.loads(private_config_path.read_text(encoding="utf-8"))
        if observed_config != private_config:
            raise RuntimeError(
                "V1/V2 resume configuration differs from the existing private cache"
            )
    else:
        _write_json(private_config_path, private_config)

    _tokenizer, wrapper, resize, _vocab = load_drafter(
        checkpoint=Path(args.post_checkpoint),
        device=args.device,
        dtype="float32",
        attn_implementation="sdpa",
    )
    wrapper.eval()
    composite = CoconutRecurrentQwen(
        wrapper, latent_token_id=int(resize.control_token_ids[2])
    ).to(device=args.device, dtype=torch.float32).eval()
    _load_stage_a_bridge(composite, Path(args.trained_bridge_checkpoint))
    help_cache = private / "v1_help_records.json"
    if help_cache.exists():
        help_records = json.loads(help_cache.read_text(encoding="utf-8"))
    else:
        help_records = _baseline_help_records(
            wrapper=wrapper,
            composite=composite,
            rows=rows,
            teacher_rows=teacher_rows,
            vocab_size=resize.original_tokenizer_size,
            device=args.device,
            cache_dir=private / "v1",
            append_batch_size=args.append_batch_size,
        )
        _write_json(help_cache, help_records)
    sampled = sorted(help_records, key=lambda record: record["sort_key"])[
        : args.max_v1_positions
    ]
    v1_probe_dir = private / "v1/probes"
    v1_results = []
    for number, record in enumerate(sampled):
        cache_path = v1_probe_dir / f"probe_{number:04d}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            payload = _v1_probe_position(
                wrapper=wrapper,
                row=rows[int(record["row_index"])],
                record=record,
                vocab_size=resize.original_tokenizer_size,
                random_probes=args.random_probes,
                epsilon_fraction=args.epsilon_fraction,
            )
            _write_json(cache_path, payload)
        v1_results.append(payload)
        print(f"phase2_v1_probe positions={number + 1}/{len(sampled)}", flush=True)

    margins = [float(row["margin"]) for row in v1_results]
    gains = [float(row["sampled_max_gain"]) for row in v1_results]
    margin_gradients = [
        float(row["targeted_margin_gradient_l2"]) for row in v1_results
    ]
    rms_values = [float(row["state_rms"]) for row in v1_results]
    v1 = {
        "dev_rows": len(rows),
        "oracle_help_positions": len(help_records),
        "position_matched_probe_sample": len(v1_results),
        "margin_distribution_all_oracle_helps": quantile_summary(
            [float(record["margin"]) for record in help_records]
        ),
        "margin_distribution_probed": quantile_summary(margins),
        "sampled_max_directional_gain": quantile_summary(gains),
        "targeted_margin_gradient_l2": quantile_summary(margin_gradients),
        "state_rms": quantile_summary(rms_values),
        "constants": {
            "c_values": [0.01, 0.02, 0.05],
            "gamma": args.gamma,
            "rho": args.rho,
            "hidden_size": int(wrapper.config.hidden_size),
        },
        "bound_compatible_fraction_using_sampled_max_gain": bound_compatible_fractions(
            margins=margins,
            sampled_max_gains=gains,
            state_rms=rms_values,
            hidden_size=int(wrapper.config.hidden_size),
            c_values=[0.01, 0.02, 0.05],
            gamma=args.gamma,
            rho=args.rho,
        ),
        "first_order_margin_compatible_fraction": bound_compatible_fractions(
            margins=margins,
            sampled_max_gains=margin_gradients,
            state_rms=rms_values,
            hidden_size=int(wrapper.config.hidden_size),
            c_values=[0.01, 0.02, 0.05],
            gamma=args.gamma,
            rho=args.rho,
        ),
        "method_caveat": (
            "Centered finite-difference JVP probes provide sampled directional "
            "gains, not a certified Lipschitz upper bound. Fractions are empirical "
            "bound compatibility and do not guarantee reachability. The targeted "
            "margin-gradient calculation is an exact local first-order diagnostic, "
            "not a finite-radius guarantee."
        ),
    }
    del composite, wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    v2_rows = rows[: args.max_v2_rows]
    v2 = {
        "map": "recurrent block only, without bridge or coda",
        "state_position": "last causal position of each deterministic dev prefix",
        "checkpoints": {
            "post_d0": _v2_checkpoint(
                checkpoint=Path(args.post_checkpoint),
                rows=v2_rows,
                expected_sha256=args.post_checkpoint_sha256,
                device=args.device,
                random_probes=args.random_probes,
                epsilon_fraction=args.epsilon_fraction,
                private_dir=private / "v2/post_d0",
            ),
            "pre_d0": _v2_checkpoint(
                checkpoint=Path(args.pre_checkpoint),
                rows=v2_rows,
                expected_sha256=args.pre_checkpoint_sha256,
                device=args.device,
                random_probes=args.random_probes,
                epsilon_fraction=args.epsilon_fraction,
                private_dir=private / "v2/pre_d0",
            ),
        },
        "method_caveat": (
            "Each value is a centered finite-difference directional JVP gain. "
            "Values above one establish an expanding sampled direction; values "
            "below one do not certify contraction."
        ),
    }
    public = {
        "kind": "paper2_phase2_v1_v2",
        "status": "complete_no_training_dev_only",
        "training_started": False,
        "optimizer_steps": 0,
        "frozen_evaluation_partitions_touched": [],
        "sources": {
            "dev_data_sha256": sha256_file(args.data_jsonl),
            "teacher_cache_summary_sha256": sha256_file(args.teacher_cache_summary),
            "post_d0_checkpoint_sha256": args.post_checkpoint_sha256,
            "pre_d0_checkpoint_sha256": args.pre_checkpoint_sha256,
            "stage_a_bridge_checkpoint_sha256": args.trained_bridge_sha256,
        },
        "v1": v1,
        "v2": v2,
        "do_not_claim": [
            "sampled directional gain is a certified Lipschitz upper bound",
            "bound-compatible positions are guaranteed reachable",
            "absence of sampled gain above one proves contraction",
            "V1 or V2 is a confirmatory result",
        ],
    }
    _write_json(Path(args.output_summary), public)
    print(json.dumps(public, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
