"""Evaluate commanded-transition control for one trained oracle interface arm."""

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
    read_jsonl,
    read_resume_cache,
    sha256_file,
    target_embeddings,
)
from eval.eval_synthetic_depth_active_labels import (  # noqa: E402
    candidates_for_row,
    prompt_for_row,
    single_token_candidate_ids,
)
from training.checkpointing import load_trainable_checkpoint  # noqa: E402
from training.oracle_interface_probe_spec import (  # noqa: E402
    LOCKED_CONTROL_GROUPS,
    LOCKED_CONTROL_ROWS,
    LOCKED_CONTROL_TRANSITIONS,
    LOCKED_ROUTES,
    summarize_oracle_arm,
)
from training.phase_g_alpha_spec import phase_g_active_lineage_hash  # noqa: E402


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


def _reader(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[str], dict[str, int]]:
    prompt = prompt_for_row(
        row,
        prediction_space="full_symbols",
        prompt_style="question_only",
    )
    candidates = candidates_for_row(
        row,
        prediction_space="full_symbols",
        value_prefix="name:",
    )
    candidate_ids = single_token_candidate_ids(tokenizer, prompt, candidates)
    if candidate_ids is None:
        raise RuntimeError(
            f"Oracle interface probe requires one-token candidates on {row['id']}"
        )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)
    return encoded, list(candidate_ids), candidate_ids


def _loop_readout(
    output: Any,
    names: list[str],
    candidate_ids: dict[str, int],
) -> tuple[list[str], list[dict[str, float]]]:
    if output.loop_logits is None or output.loop_logits.dim() != 5:
        raise RuntimeError(
            "Oracle interface evaluation requires [batch,K,loop,seq,vocab] logits"
        )
    ids = torch.tensor(
        [candidate_ids[name] for name in names],
        device=output.loop_logits.device,
        dtype=torch.long,
    )
    predictions: list[str] = []
    score_rows: list[dict[str, float]] = []
    for loop_idx in range(output.loop_logits.shape[2]):
        scores = output.loop_logits[0, 0, loop_idx, -1, :].index_select(
            dim=-1,
            index=ids,
        )
        selected = int(scores.argmax().item())
        predictions.append(names[selected])
        score_rows.append(
            {
                name: float(scores[index].detach().float().cpu().item())
                for index, name in enumerate(names)
            }
        )
    return predictions, score_rows


def _forward(
    wrapper: Any,
    encoded: dict[str, torch.Tensor],
    commands: torch.Tensor,
    *,
    depth: int,
    route: str,
    enabled: bool,
    force_identity: bool,
) -> Any:
    wrapper.eval()
    with torch.no_grad():
        return wrapper(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=None,
            max_loops=depth,
            num_trajectories=1,
            particle_update_mode="none",
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
            logits_to_keep=1,
            oracle_reentry_enabled=enabled,
            oracle_reentry_mode=route,
            oracle_reentry_targets=commands if enabled else None,
            oracle_reentry_force_identity=force_identity,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--keeper", required=True)
    parser.add_argument("--expected_keeper_sha256", required=True)
    parser.add_argument("--conditioner_checkpoint", required=True)
    parser.add_argument("--expected_conditioner_sha256", required=True)
    parser.add_argument("--route", choices=LOCKED_ROUTES, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume_cache_path", required=True)
    parser.add_argument("--bottleneck_dim", type=int, default=256)
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress_every", type=int, default=16)
    args = parser.parse_args()

    if sha256_file(args.keeper) != args.expected_keeper_sha256:
        raise AssertionError("Oracle interface keeper SHA mismatch")
    checkpoint_sha = sha256_file(args.conditioner_checkpoint)
    if checkpoint_sha != args.expected_conditioner_sha256:
        raise AssertionError("Oracle interface conditioner SHA mismatch")
    rows = read_jsonl(args.data_jsonl)
    if len(rows) != LOCKED_CONTROL_ROWS:
        raise AssertionError("Oracle interface probe requires the locked 106 rows")
    if len({str(row["base_problem_id"]) for row in rows}) != LOCKED_CONTROL_GROUPS:
        raise AssertionError("Oracle interface probe requires the locked 32 groups")
    if sum(int(row["depth"]) for row in rows) != LOCKED_CONTROL_TRANSITIONS:
        raise AssertionError("Oracle interface probe requires the locked 305 transitions")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    wrapper = load_recurrent_wrapper(loader_args(args), args.keeper)
    lineage_before = phase_g_active_lineage_hash(wrapper.named_parameters())
    wrapper.enable_oracle_reentry_conditioner(
        bottleneck_dim=args.bottleneck_dim,
    )
    load_info = load_trainable_checkpoint(
        wrapper,
        args.conditioner_checkpoint,
    )
    loaded = [
        str(name)
        for name in load_info["loaded_keys"]
        if str(name).startswith("oracle_reentry_conditioner.")
    ]
    if not loaded:
        raise AssertionError("Conditioner checkpoint restored no oracle tensors")
    if any(
        str(name).startswith("oracle_reentry_conditioner.")
        for name in load_info["skipped"]
    ):
        raise AssertionError("Conditioner checkpoint skipped oracle tensors")
    checkpoint_config = dict(load_info["checkpoint"].get("config") or {})
    if checkpoint_config.get("route") != args.route:
        raise AssertionError("Conditioner checkpoint route does not match eval arm")
    lineage_loaded = phase_g_active_lineage_hash(wrapper.named_parameters())
    if lineage_loaded != lineage_before:
        raise AssertionError("Loading conditioner changed the frozen keeper lineage")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.resume_cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = read_resume_cache(cache_path)
    for index, payload in enumerate(cached):
        if index >= len(rows) or str(payload["id"]) != str(rows[index]["id"]):
            raise AssertionError("Oracle eval resume cache row mismatch")
        if payload["route"] != args.route:
            raise AssertionError("Oracle eval resume cache route mismatch")
        if payload["conditioner_sha256"] != checkpoint_sha:
            raise AssertionError("Oracle eval resume cache checkpoint mismatch")

    baseline_by_group: dict[str, dict[str, Any]] = {}
    identity_groups = 0
    for row in rows:
        group = str(row["base_problem_id"])
        if group in baseline_by_group:
            continue
        encoded, names, candidate_ids = _reader(
            tokenizer,
            row,
            device=args.device,
        )
        commands = target_embeddings(
            wrapper,
            row,
            candidate_ids,
            device=args.device,
        )
        baseline_output = _forward(
            wrapper,
            encoded,
            commands,
            depth=int(row["depth"]),
            route=args.route,
            enabled=False,
            force_identity=False,
        )
        identity_output = _forward(
            wrapper,
            encoded,
            commands,
            depth=int(row["depth"]),
            route=args.route,
            enabled=True,
            force_identity=True,
        )
        if not torch.equal(
            baseline_output.loop_logits,
            identity_output.loop_logits,
        ):
            max_delta = float(
                (
                    baseline_output.loop_logits.float()
                    - identity_output.loop_logits.float()
                )
                .abs()
                .max()
                .cpu()
                .item()
            )
            raise AssertionError(
                f"Zeroed-conditioning identity failed for {group}: {max_delta}"
            )
        identity_groups += 1
        default_predictions, default_scores = _loop_readout(
            baseline_output,
            names,
            candidate_ids,
        )
        baseline_by_group[group] = {
            "predictions": default_predictions,
            "scores": default_scores,
        }
    if identity_groups != LOCKED_CONTROL_GROUPS:
        raise AssertionError(
            f"Identity receipt covered {identity_groups} groups, expected "
            f"{LOCKED_CONTROL_GROUPS}"
        )

    with cache_path.open("a", encoding="utf-8") as cache_handle:
        for index, row in enumerate(rows, start=1):
            if index <= len(cached):
                continue
            if index == 1 or index % args.progress_every == 0 or index == len(rows):
                print(
                    f"oracle_eval_progress route={args.route} row={index}/{len(rows)} "
                    f"depth={row['depth']}",
                    flush=True,
                )
            encoded, names, candidate_ids = _reader(
                tokenizer,
                row,
                device=args.device,
            )
            commands = target_embeddings(
                wrapper,
                row,
                candidate_ids,
                device=args.device,
            )
            group = str(row["base_problem_id"])
            baseline = baseline_by_group[group]

            conditioned_output = _forward(
                wrapper,
                encoded,
                commands,
                depth=int(row["depth"]),
                route=args.route,
                enabled=True,
                force_identity=False,
            )
            predictions, score_rows = _loop_readout(
                conditioned_output,
                names,
                candidate_ids,
            )
            chain = [str(value) for value in row["sampled_chain"]]
            successors = {
                str(source): {str(target) for target in targets}
                for source, targets in row["successors"].items()
            }
            transitions: list[dict[str, Any]] = []
            for loop_offset, (target, prediction, scores) in enumerate(
                zip(chain[1:], predictions, score_rows),
                start=1,
            ):
                other_scores = [
                    float(score)
                    for name, score in scores.items()
                    if name != target
                ]
                target_margin = float(scores[target]) - max(other_scores)
                default_prediction = str(
                    baseline["predictions"][loop_offset - 1]
                )
                transitions.append(
                    {
                        "loop_index": loop_offset,
                        "target": target,
                        "prediction": prediction,
                        "default_prediction": default_prediction,
                        "command_is_default": target == default_prediction,
                        "controlled": prediction == target,
                        "legal": prediction in successors[chain[loop_offset - 1]],
                        "target_margin": target_margin,
                    }
                )
            payload = {
                "id": str(row["id"]),
                "base_problem_id": group,
                "depth": int(row["depth"]),
                "route": args.route,
                "conditioner_sha256": checkpoint_sha,
                "target": str(row["target"]),
                "reachable_symbols": list(map(str, row["reachable_symbols"])),
                "prediction": predictions[-1],
                "valid": predictions[-1]
                in {str(value) for value in row["reachable_symbols"]},
                "transitions": transitions,
                "oracle_metrics": {
                    name: float(value.detach().float().cpu().item())
                    for name, value in conditioned_output.metrics.items()
                    if name.startswith("oracle_reentry_") and value.numel() == 1
                },
            }
            cache_handle.write(json.dumps(payload, sort_keys=True) + "\n")
            cache_handle.flush()
            cached.append(payload)

    transition_rows = [
        {
            "id": row["id"],
            "base_problem_id": row["base_problem_id"],
            "depth": row["depth"],
            **transition,
        }
        for row in cached
        for transition in row["transitions"]
    ]
    terminal_rows = [
        {
            "id": row["id"],
            "base_problem_id": row["base_problem_id"],
            "valid": row["valid"],
        }
        for row in cached
    ]
    lineage_after = phase_g_active_lineage_hash(wrapper.named_parameters())
    summary = summarize_oracle_arm(
        transition_rows,
        terminal_rows,
        route=args.route,
        identity_exact=identity_groups == LOCKED_CONTROL_GROUPS,
        frozen_lineage_unchanged=lineage_after == lineage_before,
    )
    summary.update(
        {
            "status": "finished",
            "gate_status": "passed" if summary["passed"] else "blocked",
            "keeper": args.keeper,
            "keeper_sha256": args.expected_keeper_sha256,
            "conditioner_checkpoint": args.conditioner_checkpoint,
            "conditioner_sha256": checkpoint_sha,
            "identity_groups_exact": identity_groups,
            "frozen_lineage_before": lineage_before,
            "frozen_lineage_after": lineage_after,
            "loaded_oracle_keys": loaded,
        }
    )
    (output_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in cached),
        encoding="utf-8",
    )
    (output_dir / "transitions.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in transition_rows
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
