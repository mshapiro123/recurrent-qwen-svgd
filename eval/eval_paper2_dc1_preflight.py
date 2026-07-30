"""Run the forward-only DC1-P diagnostics on reusable DEV-C rows."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import load_drafter, read_jsonl  # noqa: E402
from eval.eval_paper2_dc0_depth_by_append import (  # noqa: E402
    anchor_registered_k0,
    evaluate_append_arm,
    flatten_rows,
    parameter_fingerprint,
    transition_counts,
)
from eval.eval_speculative_depth_d0_floor import load_partition_cache  # noqa: E402
from models.coconut_composite import CoconutRecurrentQwen  # noqa: E402
from training.paper2_dc1 import (  # noqa: E402
    PREFLIGHT_POSITION_BUDGET,
    dc1_preflight_spec,
    scale_interpolation_schedule,
)
from training.speculative_depth_d0_corpus import sha256_file, stable_fraction  # noqa: E402


def select_probe_indices(
    rows: Sequence[dict[str, Any]],
    *,
    position_budget: int,
    seed: int = 20260729,
) -> list[int]:
    ordered = sorted(
        range(len(rows)),
        key=lambda index: stable_fraction(str(rows[index]["row_id"]), seed=seed),
    )
    selected: list[int] = []
    positions = 0
    for index in ordered:
        row_positions = len(rows[index]["input_ids"]) - 1
        selected.append(index)
        positions += row_positions
        if positions >= int(position_budget):
            break
    if positions < int(position_budget):
        raise RuntimeError(
            f"DC1-P could not reach {position_budget} positions; observed {positions}"
        )
    return selected


def aggregate_attention_profiles(
    batch_dir: Path,
    *,
    layer_split: tuple[int, int],
) -> dict[str, Any]:
    sums: dict[tuple[str, int], float] = defaultdict(float)
    counts: dict[tuple[str, int], int] = defaultdict(int)
    profiles = 0
    for path in sorted(batch_dir.glob("batch_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        weight = len(payload["indices"])
        for profile in payload["diagnostics"].get("slot_attention_profiles", []):
            profiles += weight
            for layer in profile["layers"]:
                index = int(layer["layer_index"])
                for metric in ("prefix_mass", "prior_slot_mass", "self_mass"):
                    sums[(metric, index)] += float(layer[metric]) * weight
                    counts[(metric, index)] += weight
    if not profiles:
        raise RuntimeError(f"DC1-P attention capture produced no profiles in {batch_dir}")
    prelude_end, recurrent_end = layer_split
    groups = {
        "prelude": range(0, prelude_end),
        "recurrent": range(prelude_end, recurrent_end),
        "coda": range(recurrent_end, max(index for _, index in sums) + 1),
    }
    by_layer = {}
    for index in sorted({index for _, index in sums}):
        by_layer[str(index)] = {
            metric: sums[(metric, index)] / counts[(metric, index)]
            for metric in ("prefix_mass", "prior_slot_mass", "self_mass")
        }
    by_group = {}
    for name, indices in groups.items():
        values = [by_layer[str(index)] for index in indices if str(index) in by_layer]
        by_group[name] = {
            metric: sum(row[metric] for row in values) / len(values)
            for metric in ("prefix_mass", "prior_slot_mass", "self_mass")
        }
    return {
        "profiles_weighted": profiles,
        "by_layer": by_layer,
        "by_layer_group": by_group,
    }


def assert_frozen_instance_unchanged(
    model: Any, *, before: str, instance: str
) -> str:
    """Verify one loaded model instance against its own pre-evaluation state."""
    after = parameter_fingerprint(model)
    if before != after:
        raise RuntimeError(f"DC1-P mutated frozen {instance} parameters")
    return after


def scale_sweep_reading(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [int(row["transition"]["net_correct_delta"]) for row in rows]
    monotone = all(left <= right for left, right in zip(deltas, deltas[1:]))
    best = max(rows, key=lambda row: int(row["transition"]["net_correct_delta"]))
    return {
        "monotone_net_utility": monotone,
        "best_label": best["label"],
        "best_net_correct_delta": int(best["transition"]["net_correct_delta"]),
        "finding_is_gate": False,
        "initialization_candidate": best["label"],
    }


def fragility_proxy(dc0_summary: dict[str, Any]) -> dict[str, Any]:
    quartiles = dc0_summary["append_arms"]["append_raw"]["transitions"]["0_to_1"][
        "by_drafter_logprob_quartile"
    ]
    return {
        "status": "proxy_only_exact_model_margin_unavailable",
        "available_proxy": "teacher log probability of the drafter token by quartile",
        "by_teacher_logprob_quartile": {
            key: {
                "hurts": int(value["hurts"]),
                "total": int(value["total"]),
                "hurt_rate": int(value["hurts"]) / int(value["total"]),
            }
            for key, value in quartiles.items()
        },
        "exact_requested_model_margin": {
            "status": "not_recoverable_from_saved_dc0_predictions",
            "reason": (
                "DC0 saved token predictions but not the baseline top-two model logits; "
                "EVAL-B text is spent and is not reopened"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--teacher_cache_summary", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected_checkpoint_sha256", required=True)
    parser.add_argument("--dc0_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--private_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--append_batch_size", type=int, default=8)
    parser.add_argument("--position_budget", type=int, default=PREFLIGHT_POSITION_BUDGET)
    args = parser.parse_args()

    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("DC1-P checkpoint SHA-256 mismatch")
    rows = read_jsonl(args.data_jsonl)
    selected_indices = select_probe_indices(rows, position_budget=args.position_budget)
    selected_rows = [rows[index] for index in selected_indices]
    teacher_summary = json.loads(Path(args.teacher_cache_summary).read_text(encoding="utf-8"))
    teacher_rows = load_partition_cache(teacher_summary, "teacher_7b", "dev_c")
    selected_teacher = [teacher_rows[index] for index in selected_indices]
    targets = torch.cat([row["teacher_greedy_token_id"].long() for row in selected_teacher])

    _tokenizer, wrapper, resize, _original_vocab = load_drafter(
        checkpoint=Path(args.checkpoint),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    wrapper.eval()
    composite = CoconutRecurrentQwen(
        wrapper, latent_token_id=int(resize.control_token_ids[2])
    ).eval()
    before = parameter_fingerprint(composite)
    private = Path(args.private_dir)
    private.mkdir(parents=True, exist_ok=True)

    square_sum = count = 0
    with torch.no_grad():
        for row in selected_rows:
            states = wrapper.qwen.embed_tokens(
                torch.tensor(row["input_ids"], device=args.device)
            ).float()
            square_sum += float(states.square().sum().cpu())
            count += states.numel()
    embedding_rms = math.sqrt(square_sum / count)
    probe_ids = torch.tensor([selected_rows[0]["input_ids"][:16]], device=args.device)
    raw_probe = composite.depth_by_append(
        input_ids=probe_ids,
        append_steps=1,
        feedback_mode="raw",
        prediction_vocab_size=resize.original_tokenizer_size,
    )
    raw_rms = float(raw_probe.diagnostics["fed_hidden_rms_mean"])
    schedule = scale_interpolation_schedule(embedding_rms=embedding_rms, raw_rms=raw_rms)

    # Every shared-path arm is anchored to the exact registered k=0 predictions.
    neutral_rows, _ = evaluate_append_arm(
        composite,
        selected_rows,
        arm="registered_anchor_cache",
        feedback_mode="neutral",
        reference_rms=None,
        neutral_token_id=int(resize.control_token_ids[2]),
        vocab_size=resize.original_tokenizer_size,
        device=args.device,
        batch_size=args.append_batch_size,
        resume_dir=private / "append_batches",
        append_steps=1,
    )
    cached_anchor = flatten_rows(neutral_rows)
    registered_rows = []
    for row in selected_rows:
        values = torch.tensor([row["input_ids"]], device=args.device)
        registered_rows.append(
            composite.depth_by_append(
                input_ids=values,
                append_steps=0,
                feedback_mode="raw",
                prediction_vocab_size=resize.original_tokenizer_size,
            ).predictions[0, :, 0]
        )
    registered = torch.cat(registered_rows)
    anchored_cache, _cached_k0, anchor_diagnostics = anchor_registered_k0(
        cached_anchor, registered
    )
    baseline = anchored_cache[:, 0]

    scale_rows = []
    for spec in schedule:
        predicted_rows, counters = evaluate_append_arm(
            composite,
            selected_rows,
            arm=f"scale_{spec['label']}",
            feedback_mode=str(spec["feedback_mode"]),
            reference_rms=embedding_rms if spec["feedback_mode"] == "scaled" else None,
            feedback_scale=spec["feedback_scale"],
            neutral_token_id=int(resize.control_token_ids[2]),
            vocab_size=resize.original_tokenizer_size,
            device=args.device,
            batch_size=args.append_batch_size,
            resume_dir=private / "append_batches",
            append_steps=1,
        )
        predicted, _cached, _path = anchor_registered_k0(
            flatten_rows(predicted_rows), registered
        )
        scale_rows.append(
            {
                **spec,
                "transition": transition_counts(baseline, predicted[:, 1], targets),
                "counters": counters,
            }
        )

    position_rows = {}
    for mode in ("advance", "superposed"):
        predicted_rows, counters = evaluate_append_arm(
            composite,
            selected_rows,
            arm=f"position_{mode}",
            feedback_mode="raw",
            reference_rms=None,
            neutral_token_id=int(resize.control_token_ids[2]),
            vocab_size=resize.original_tokenizer_size,
            device=args.device,
            batch_size=args.append_batch_size,
            resume_dir=private / "append_batches",
            append_steps=1,
            slot_position_mode=mode,
        )
        predicted, _cached, _path = anchor_registered_k0(
            flatten_rows(predicted_rows), registered
        )
        position_rows[mode] = {
            "transition": transition_counts(baseline, predicted[:, 1], targets),
            "counters": counters,
        }

    sdpa_after = assert_frozen_instance_unchanged(
        composite, before=before, instance="SDPA checkpoint or identity bridge"
    )
    del composite, wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _tokenizer, attention_wrapper, attention_resize, _original_vocab = load_drafter(
        checkpoint=Path(args.checkpoint),
        device=args.device,
        dtype=args.dtype,
        attn_implementation="eager",
    )
    for parameter in attention_wrapper.parameters():
        parameter.requires_grad_(False)
    attention_wrapper.eval()
    attention_composite = CoconutRecurrentQwen(
        attention_wrapper, latent_token_id=int(attention_resize.control_token_ids[2])
    ).eval()
    attention_before = parameter_fingerprint(attention_composite)
    attention = {}
    for stratum in ("general", "code"):
        stratum_rows = [
            row for row in selected_rows if str(row["stratum"]) == stratum
        ]
        attention_resume = private / "attention_batches" / stratum
        evaluate_append_arm(
            attention_composite,
            stratum_rows,
            arm="raw_attention",
            feedback_mode="raw",
            reference_rms=None,
            neutral_token_id=int(attention_resize.control_token_ids[2]),
            vocab_size=attention_resize.original_tokenizer_size,
            device=args.device,
            batch_size=args.append_batch_size,
            resume_dir=attention_resume,
            append_steps=1,
            capture_slot_attentions=True,
        )
        attention[stratum] = aggregate_attention_profiles(
            attention_resume / "raw_attention",
            layer_split=(
                int(attention_wrapper.layer_split.prelude_end),
                int(attention_wrapper.layer_split.recurrent_end),
            ),
        )

    attention_after = assert_frozen_instance_unchanged(
        attention_composite,
        before=attention_before,
        instance="eager-attention checkpoint or identity bridge",
    )
    dc0 = json.loads(Path(args.dc0_summary).read_text(encoding="utf-8"))
    summary = {
        "kind": "paper2_dc1_preflight",
        "status": "complete_descriptive_preflight_requires_preregistration",
        "spec": dc1_preflight_spec(),
        "training_started": False,
        "optimizer_steps": 0,
        "evaluation_c_touched": False,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "checkpoint_mutated": False,
        "parameter_integrity": {
            "sdpa_before": before,
            "sdpa_after": sdpa_after,
            "eager_before": attention_before,
            "eager_after": attention_after,
            "instances_checked_independently": True,
        },
        "dev_c": {
            "data_jsonl_sha256": sha256_file(args.data_jsonl),
            "selected_rows": len(selected_rows),
            "selected_positions": len(targets),
            "position_budget": int(args.position_budget),
            "teacher_cache_sha256": sha256_file(args.teacher_cache_summary),
        },
        "execution_path_anchor": anchor_diagnostics,
        "scale_interpolation": {
            "embedding_rms": embedding_rms,
            "raw_hidden_rms": raw_rms,
            "rows": scale_rows,
            "reading": scale_sweep_reading(scale_rows),
        },
        "slot_attention_profile": attention,
        "position_id_ablation": position_rows,
        "fragility_stratification": fragility_proxy(dc0),
        "training_authorized_by_this_receipt": False,
        "next_required_artifact": "locked DC1 preregistration after strategy review",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    del attention_wrapper, attention_composite
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
