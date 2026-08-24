"""Registered W1 generative staging for margin-positive correction arms."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

from eval.eval_paper2_bicameral_w1 import (
    EVALUATOR_TAG,
    SCHEDULE,
    atomic_json,
    build_controls,
    interface_patch,
    read_jsonl,
    write_jsonl,
)
from eval.eval_paper2_phase3_p34_task_inference import (
    P34CachedPrefix,
    P34NextTokenOutput,
    current_position_mask,
)
from eval.eval_paper2_phase3_p34_task_trajectory import score_generation
from training.paper2_bicameral_w1 import (
    GAMMA,
    ORACLE_TARGET_ASSISTED,
    POPULATION_TARGET,
    build_phase_b_granularity_targets,
    deterministic_permutation,
)
from training.paper2_stage2bs_depth_study import INITIALIZATION_SEED_BASE, sha256_file
from training.run_paper2_stage2b_depth import _build_model


GENERATION_BATTERIES = frozenset({"gsm8k", "mbpp", "tier1"})
GENERATION_CAPS = {"gsm8k": 256, "mbpp": 384, "tier1": 64}
GENERATION_CONFIG = {
    "kind": "paper2_bicameral_w1_generation_config_v1",
    "evaluator": EVALUATOR_TAG,
    "schedule": SCHEDULE,
    "decoder": "greedy_incremental_cache_v1",
    "write_schedule": "all_active_prefill_then_each_new_causal_token_v1",
    "target_schedule": "one_fixed_arm_target_per_row_at_every_decode_step_v1",
    "gamma": GAMMA,
    "batch_size": 8,
    "caps": GENERATION_CAPS,
    "dtype": "bfloat16",
    "attention": "sdpa",
    "optimizer_steps": 0,
}


def _json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        (json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()


def freeze_generation_manifest(
    source: Path, output: Path, config_output: Path
) -> dict[str, Any]:
    rows = [row for row in read_jsonl(source) if str(row["battery"]) in GENERATION_BATTERIES]
    if len(rows) != 461 or Counter(str(row["battery"]) for row in rows) != {
        "gsm8k": 369,
        "mbpp": 67,
        "tier1": 25,
    }:
        raise RuntimeError("W1 frozen generation population changed")
    if len({str(row["item_id"]) for row in rows}) != len(rows):
        raise RuntimeError("W1 generation item ids are not unique")
    write_jsonl(output, rows)
    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(
        json.dumps(GENERATION_CONFIG, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )
    return {
        "kind": "paper2_bicameral_w1_generation_freeze_receipt_v1",
        "status": "frozen_before_scoring",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "manifest": str(output),
        "manifest_bytes": output.stat().st_size,
        "manifest_sha256": sha256_file(output),
        "rows": len(rows),
        "battery_counts": dict(sorted(Counter(str(row["battery"]) for row in rows).items())),
        "config": str(config_output),
        "config_bytes": config_output.stat().st_size,
        "config_sha256": sha256_file(config_output),
        "config_canonical_sha256": _json_sha256(GENERATION_CONFIG),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }


@dataclass
class W1FixedTargetGraph:
    wrapper: Any
    direction_by_item_id: Mapping[str, torch.Tensor]
    _batch_directions: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return next(self.wrapper.parameters()).device

    def prepare_probe_batch(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._batch_directions = torch.stack(
            [self.direction_by_item_id[str(row["item_id"])].float() for row in rows]
        ).to(self.device)

    def _run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        prelude_cache: Any = None,
        loop_cache: Any = None,
        current_token_only: bool = False,
        current_positions: torch.Tensor | None = None,
    ) -> P34NextTokenOutput:
        if self._batch_directions is None or self._batch_directions.shape[0] != input_ids.shape[0]:
            raise RuntimeError("W1 generation batch directions are not bound")
        with interface_patch(self.wrapper, direction=self._batch_directions) as captured:
            output = self.wrapper(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_loops=1,
                stage2b_depth_enabled=True,
                stage2b_stage="M2",
                stage2b_amplitude=0.05,
                stage2b_score_only_sparse_logits=True,
                stage2b_loop_past_key_values=(None if loop_cache is None else (loop_cache,)),
                return_loop_logits=True,
                past_key_values=prelude_cache,
                use_cache=loop_cache is not None,
                return_dict=True,
            )
        if output.loop_logits is None or "telemetry" not in captured:
            raise RuntimeError("W1 generation failed to execute the external write")
        loops = output.loop_logits[:, 0]
        positions = (
            current_position_mask(attention_mask)[1]
            if current_positions is None
            else current_positions
        )
        batch = torch.arange(input_ids.shape[0], device=input_ids.device)
        selected = loops[:, -1, -1] if current_token_only else loops[batch, -1, positions]
        top2 = selected.float().topk(2, dim=-1).values
        ratios = captured["telemetry"]["write_ratio"].to(input_ids.device)
        return P34NextTokenOutput(
            augmented_logits=selected,
            base_logits=selected,
            writeback_ratio=ratios,
            position_gate=torch.ones_like(ratios),
            current_positions=positions,
            scratch_state=torch.empty((input_ids.shape[0], 0, 0), device=input_ids.device),
            answer_token_margin=top2[:, 0] - top2[:, 1],
        )

    @torch.inference_mode()
    def prefill_cached(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[P34CachedPrefix, P34NextTokenOutput]:
        from transformers.cache_utils import DynamicCache

        prelude_cache = DynamicCache(config=self.wrapper.config)
        loop_cache = DynamicCache(config=self.wrapper.config)
        positions = current_position_mask(attention_mask)[1]
        result = self._run(
            input_ids,
            attention_mask,
            prelude_cache=prelude_cache,
            loop_cache=loop_cache,
            current_positions=positions,
        )
        state = P34CachedPrefix(
            input_ids=input_ids,
            hidden=torch.empty((input_ids.shape[0], input_ids.shape[1], 0), device=self.device),
            layer6_hidden=torch.empty((input_ids.shape[0], input_ids.shape[1], 0), device=self.device),
            attention_mask=attention_mask,
            past_key_values={"prelude": prelude_cache, "loop": loop_cache},
        )
        return state, result

    @torch.inference_mode()
    def advance_cached(
        self, *, state: P34CachedPrefix, selected_tokens: torch.Tensor
    ) -> tuple[P34CachedPrefix, P34NextTokenOutput]:
        if selected_tokens.ndim == 1:
            selected_tokens = selected_tokens[:, None]
        attention = torch.cat(
            [state.attention_mask, torch.ones_like(selected_tokens, dtype=state.attention_mask.dtype)],
            dim=1,
        )
        caches = state.past_key_values
        if not isinstance(caches, dict) or set(caches) != {"prelude", "loop"}:
            raise RuntimeError("W1 generation cache state changed")
        positions = torch.full(
            (selected_tokens.shape[0],),
            attention.shape[1] - 1,
            dtype=torch.long,
            device=selected_tokens.device,
        )
        output = self._run(
            selected_tokens,
            attention,
            prelude_cache=caches["prelude"],
            loop_cache=caches["loop"],
            current_token_only=True,
            current_positions=positions,
        )
        updated = P34CachedPrefix(
            input_ids=torch.cat([state.input_ids, selected_tokens], dim=1),
            hidden=torch.empty((selected_tokens.shape[0], attention.shape[1], 0), device=self.device),
            layer6_hidden=torch.empty((selected_tokens.shape[0], attention.shape[1], 0), device=self.device),
            attention_mask=attention,
            past_key_values=caches,
        )
        return updated, output


def _arm_summary(rows: Sequence[Mapping[str, Any]], *, arm: str, seed: int) -> dict[str, Any]:
    by_battery: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_battery[str(row["battery"])].append(row)
    return {
        "kind": "paper2_bicameral_w1_generation_cell_v1",
        "arm": arm,
        "seed": seed,
        "rows": len(rows),
        "correct": sum(bool(row["augmented_correct"]) for row in rows),
        "accuracy": sum(bool(row["augmented_correct"]) for row in rows) / len(rows),
        "by_battery": {
            name: {
                "rows": len(items),
                "correct": sum(bool(row["augmented_correct"]) for row in items),
                "accuracy": sum(bool(row["augmented_correct"]) for row in items) / len(items),
            }
            for name, items in sorted(by_battery.items())
        },
    }


def _positive_phase_b_arms(summary_paths: Sequence[Path]) -> list[str]:
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    by_seed = {
        int(summary["seed"]): {str(cell["arm"]): cell for cell in summary["cells"]}
        for summary in summaries
    }
    common = set(by_seed[0]) & set(by_seed[1])
    return sorted(
        arm for arm in common if all(float(by_seed[seed][arm]["ci_low"]) > 0 for seed in (0, 1))
    )


def _directions_for_seed(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    cache = torch.load(args.phase_a_targets, map_location="cpu", weights_only=False)
    item_to_index = {str(item): index for index, item in enumerate(cache["item_ids"])}
    controls = build_controls(cache["families"], seed=args.seed)
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("l0a", "l5_a", "l0c", "l5_c", "l4"):
        values = controls[arm]
        arms[arm] = {
            "directions": {str(row["item_id"]): values[item_to_index[str(row["item_id"])]] for row in rows},
            "target_tag": ORACLE_TARGET_ASSISTED,
            "oracle_routed": True,
        }

    assignments = torch.load(args.phase_b_assignments, map_location="cpu", weights_only=False)
    labels = torch.as_tensor(assignments["assignments"], dtype=torch.long)
    granularity = build_phase_b_granularity_targets(cache["families"]["l0c"].float(), labels)
    residual = torch.load(args.phase_b_residual, map_location="cpu", weights_only=False)
    phase_b_positive = _positive_phase_b_arms(args.phase_b_summaries)
    phase_b_values: dict[str, torch.Tensor] = {
        name: granularity[name].float() for name in ("l1", "l2", "l3")
    }
    for index, vector in enumerate(residual["directions"], start=1):
        phase_b_values[f"l6_u{index}_pos"] = vector.unsqueeze(0).expand(len(cache["item_ids"]), -1)
        phase_b_values[f"l6_u{index}_neg"] = -vector.unsqueeze(0).expand(len(cache["item_ids"]), -1)
    for arm in phase_b_positive:
        values = phase_b_values[arm]
        arms[arm] = {
            "directions": {str(row["item_id"]): values[item_to_index[str(row["item_id"])]] for row in rows},
            "target_tag": POPULATION_TARGET,
            "oracle_routed": arm in {"l1", "l3"},
        }
        if arm in {"l1", "l2", "l3"}:
            permutation = deterministic_permutation(
                values.shape[0], family=f"phase_b:{arm}:seed{args.seed}"
            )
            shuffled = values[permutation]
            arms[f"l5_{arm}"] = {
                "directions": {
                    str(row["item_id"]): shuffled[item_to_index[str(row["item_id"])]]
                    for row in rows
                },
                "target_tag": POPULATION_TARGET,
                "oracle_routed": False,
                "control_for": arm,
            }
    return arms


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.generation_manifest)
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
    arms = _directions_for_seed(args, rows)
    cells = []
    for arm, payload in arms.items():
        row_path = args.private_dir / f"seed_{args.seed}_generation_{arm}.jsonl"
        summary_path = args.output_dir / f"seed_{args.seed}_generation_{arm}.json"
        if row_path.is_file() and summary_path.is_file():
            cells.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        graph = W1FixedTargetGraph(wrapper=wrapper, direction_by_item_id=payload["directions"])
        scored = score_generation(graph, tokenizer, rows, batch_size=8)
        for row in scored:
            row["kind"] = "paper2_bicameral_w1_generation_row_v1"
            row["arm"] = arm
            row["seed"] = args.seed
            row["target_tag"] = payload["target_tag"]
            row["oracle_routed"] = payload["oracle_routed"]
            if "control_for" in payload:
                row["control_for"] = payload["control_for"]
        summary = _arm_summary(scored, arm=arm, seed=args.seed)
        summary.update(
            {
                "target_tag": payload["target_tag"],
                "oracle_routed": payload["oracle_routed"],
                "manifest_sha256": sha256_file(args.generation_manifest),
                "config_sha256": sha256_file(args.generation_config),
            }
        )
        if "control_for" in payload:
            summary["control_for"] = payload["control_for"]
        write_jsonl(row_path, scored)
        atomic_json(summary_path, summary)
        cells.append(summary)
    result = {
        "kind": "paper2_bicameral_w1_generation_seed_v1",
        "status": "complete_score_only",
        "seed": args.seed,
        "rows": len(rows),
        "cells": cells,
        "checkpoint_chain": chain,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / f"seed_{args.seed}_generation_summary.json", result)
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--seed", type=int, choices=(0, 1), required=True)
    result.add_argument("--generation_manifest", type=Path, required=True)
    result.add_argument("--generation_config", type=Path, required=True)
    result.add_argument("--phase_a_targets", type=Path, required=True)
    result.add_argument("--phase_b_assignments", type=Path, required=True)
    result.add_argument("--phase_b_residual", type=Path, required=True)
    result.add_argument("--phase_b_summaries", type=Path, nargs=2, required=True)
    result.add_argument("--output_dir", type=Path, required=True)
    result.add_argument("--private_dir", type=Path, required=True)
    result.add_argument("--model_cache", type=Path, required=True)
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
