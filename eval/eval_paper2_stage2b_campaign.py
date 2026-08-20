"""Registered DEV-1 and DEV-2 reads for the Stage 2B-D campaign."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from eval.eval_paper2_phase3_p31_references import (
    _chat_prompt,
    _generation_prompt,
    _mcq,
    _mcq_prompt,
)
from eval.eval_paper2_phase3_p34_task_inference import (
    P34CachedPrefix,
    P34NextTokenOutput,
    current_position_mask,
)
from eval.eval_paper2_phase3_p34_task_trajectory import score_generation, score_mcq


@dataclass
class Stage2BTaskInferenceGraph:
    """Registered task graph over the recurrent Stage 2B wrapper."""

    wrapper: Any
    stage: str
    amplitude: float
    flow_loops: int = 4
    diagnostic_mode: str = "standard"
    last_token_projection: bool = False
    sparse_loop_projection: bool = True
    incremental_cache: bool = False

    @property
    def device(self) -> torch.device:
        return next(self.wrapper.parameters()).device

    def _run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        prelude_cache: Any = None,
        loop_caches: tuple[Any, ...] | None = None,
        current_token_only: bool = False,
    ) -> P34NextTokenOutput:
        output = self.wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_loops=self.flow_loops,
            stage2b_depth_enabled=True,
            stage2b_stage=self.stage,
            stage2b_amplitude=self.amplitude,
            stage2b_diagnostic_mode=self.diagnostic_mode,
            stage2b_score_only_sparse_logits=self.sparse_loop_projection,
            stage2b_loop_past_key_values=loop_caches,
            return_loop_logits=True,
            logits_to_keep=1 if self.last_token_projection else 0,
            past_key_values=prelude_cache,
            use_cache=loop_caches is not None,
            return_dict=True,
        )
        if output.loop_logits is None:
            raise RuntimeError("Stage 2B task read requires loop logits")
        positions = current_position_mask(attention_mask)[1]
        batch = torch.arange(input_ids.shape[0], device=input_ids.device)
        loops = output.loop_logits[:, 0]
        if self.last_token_projection or current_token_only:
            selected = loops[:, -1, -1, :]
            base = loops[:, 0, -1, :]
        else:
            selected = loops[batch, -1, positions, :]
            base = loops[batch, 0, positions, :]
        top2 = selected.float().topk(2, dim=-1).values
        metrics = output.metrics or {}
        gate = float(metrics.get("stage2b_position_gate_mean", torch.tensor(0.0)).cpu())
        ratio = float(metrics.get("stage2b_writeback_ratio_mean", torch.tensor(0.0)).cpu())
        rows = input_ids.shape[0]
        return P34NextTokenOutput(
            augmented_logits=selected,
            base_logits=base,
            writeback_ratio=torch.full((rows,), ratio, device=input_ids.device),
            position_gate=torch.full((rows,), gate, device=input_ids.device),
            current_positions=positions,
            scratch_state=torch.empty((rows, 0, 0), device=input_ids.device),
            answer_token_margin=top2[:, 0] - top2[:, 1],
        )

    @torch.inference_mode()
    def next_token(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        flow_loops: int | None = None,
    ) -> P34NextTokenOutput:
        if flow_loops is not None and int(flow_loops) != self.flow_loops:
            raise RuntimeError("Stage 2B task read K changed")
        return self._run(input_ids, attention_mask)

    @torch.inference_mode()
    def prefill_cached(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[P34CachedPrefix, P34NextTokenOutput]:
        if self.incremental_cache:
            from transformers.cache_utils import DynamicCache

            prelude_cache = DynamicCache(config=self.wrapper.config)
            loop_caches = tuple(
                DynamicCache(config=self.wrapper.config)
                for _ in range(self.flow_loops)
            )
            output = self._run(
                input_ids,
                attention_mask,
                prelude_cache=prelude_cache,
                loop_caches=loop_caches,
            )
            state = P34CachedPrefix(
                input_ids=input_ids,
                hidden=torch.empty(
                    (input_ids.shape[0], input_ids.shape[1], 0),
                    device=input_ids.device,
                ),
                layer6_hidden=torch.empty(
                    (input_ids.shape[0], input_ids.shape[1], 0),
                    device=input_ids.device,
                ),
                attention_mask=attention_mask,
                past_key_values={"prelude": prelude_cache, "loops": loop_caches},
            )
            return state, output
        state = P34CachedPrefix(
            input_ids=input_ids,
            hidden=torch.empty((input_ids.shape[0], input_ids.shape[1], 0), device=input_ids.device),
            layer6_hidden=torch.empty((input_ids.shape[0], input_ids.shape[1], 0), device=input_ids.device),
            attention_mask=attention_mask,
            past_key_values=None,
        )
        return state, self._run(input_ids, attention_mask)

    @torch.inference_mode()
    def advance_cached(
        self, *, state: P34CachedPrefix, selected_tokens: torch.Tensor
    ) -> tuple[P34CachedPrefix, P34NextTokenOutput]:
        if selected_tokens.ndim == 1:
            selected_tokens = selected_tokens[:, None]
        input_ids = torch.cat([state.input_ids, selected_tokens], dim=1)
        attention = torch.cat(
            [
                state.attention_mask,
                torch.ones_like(selected_tokens, dtype=state.attention_mask.dtype),
            ],
            dim=1,
        )
        if self.incremental_cache:
            caches = state.past_key_values
            if not isinstance(caches, dict) or set(caches) != {"prelude", "loops"}:
                raise RuntimeError("Stage 2B incremental cache state changed")
            output = self._run(
                selected_tokens,
                attention,
                prelude_cache=caches["prelude"],
                loop_caches=tuple(caches["loops"]),
                current_token_only=True,
            )
            updated = P34CachedPrefix(
                input_ids=torch.cat([state.input_ids, selected_tokens], dim=1),
                hidden=torch.empty(
                    (selected_tokens.shape[0], attention.shape[1], 0),
                    device=selected_tokens.device,
                ),
                layer6_hidden=torch.empty(
                    (selected_tokens.shape[0], attention.shape[1], 0),
                    device=selected_tokens.device,
                ),
                attention_mask=attention,
                past_key_values=caches,
            )
            return updated, output
        return self.prefill_cached(input_ids=input_ids, attention_mask=attention)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="",
    )
    temporary.replace(destination)


@torch.inference_mode()
def score_dev1(
    *,
    graph: Stage2BTaskInferenceGraph,
    tokenizer: Any,
    panel: Sequence[Mapping[str, Any]],
    base_rows: Mapping[str, Mapping[str, Any]],
    initialization_rows: Mapping[str, Mapping[str, Any]],
    seed: int,
    look: int,
    mcq_batch_size: int,
    generation_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mcq = [row for row in panel if row["battery"] in {"arc_easy", "arc_challenge", "mmlu"}]
    generation = [row for row in panel if row["battery"] in {"gsm8k", "mbpp", "tier1"}]
    scored = score_mcq(graph, tokenizer, mcq, batch_size=mcq_batch_size)
    scored.extend(
        score_generation(graph, tokenizer, generation, batch_size=generation_batch_size)
    )
    source = {str(row["item_id"]): row for row in panel}
    enriched = []
    for row in scored:
        item_id = str(row["item_id"])
        if item_id not in base_rows or item_id not in initialization_rows:
            raise RuntimeError("DEV-1 comparator coverage is incomplete")
        enriched.append(
            {
                "kind": "paper2_stage2b_dev1_row_v1",
                "seed": seed,
                "look": look,
                "item_id": item_id,
                "battery": source[item_id]["battery"],
                "current_correct": bool(row["augmented_correct"]),
                "base_correct": bool(base_rows[item_id].get("correct", base_rows[item_id].get("augmented_correct"))),
                "initialization_correct": bool(
                    initialization_rows[item_id].get(
                        "augmented_correct", initialization_rows[item_id].get("correct")
                    )
                ),
                **row,
            }
        )
    by_battery: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_battery[str(row["battery"])].append(row)
    battery = {}
    for name, rows in sorted(by_battery.items()):
        current = sum(bool(row["current_correct"]) for row in rows)
        base = sum(bool(row["base_correct"]) for row in rows)
        initialization = sum(bool(row["initialization_correct"]) for row in rows)
        battery[name] = {
            "rows": len(rows),
            "current_correct": current,
            "base_correct": base,
            "initialization_correct": initialization,
            "delta_vs_base_rows": current - base,
            "delta_vs_initialization_rows": current - initialization,
        }
    tier1 = battery["tier1"]["current_correct"]
    gsm8k = battery["gsm8k"]["current_correct"]
    tier1_floor = 19
    gsm8k_floor = 91 if seed == 0 else 94
    summary = {
        "kind": "paper2_stage2b_dev1_summary_v1",
        "seed": seed,
        "look": look,
        "rows": len(enriched),
        "battery": battery,
        "safety": {
            "tier1_floor": tier1_floor,
            "tier1_correct": tier1,
            "tier1_pass": tier1 >= tier1_floor,
            "gsm8k_floor": gsm8k_floor,
            "gsm8k_correct": gsm8k,
            "gsm8k_pass": gsm8k >= gsm8k_floor,
        },
        "both_comparators_reported": True,
    }
    summary["safety"]["pass"] = bool(
        summary["safety"]["tier1_pass"] and summary["safety"]["gsm8k_pass"]
    )
    return enriched, summary


def _forced_target(tokenizer: Any, row: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    if row["battery"] in {"arc_easy", "arc_challenge", "mmlu"}:
        question, choices, answer = _mcq(row)
        prompt = _mcq_prompt(question, choices)
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        suffix = tokenizer(f" {answer}", add_special_tokens=False)["input_ids"]
    else:
        content, _cap = _generation_prompt(row)
        prompt = _chat_prompt(tokenizer, content)
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        if row["battery"] == "gsm8k":
            target = f"Final answer: {row['answer']}"
        elif row["battery"] == "mbpp":
            target = str(row["answer"])
        else:
            target = str(row["answer"])
        suffix = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not suffix:
        raise RuntimeError("DEV-2 target continuation is empty")
    return [int(value) for value in prompt_ids], [int(value) for value in suffix]


@torch.inference_mode()
def score_dev2_margins(
    *,
    graph: Stage2BTaskInferenceGraph,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    look: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = [(row, *_forced_target(tokenizer, row)) for row in rows]
    results = []
    activation_maxima: dict[str, float] = {}
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        widths = [len(prompt) + len(target) for _row, prompt, target in batch]
        width = max(widths)
        input_ids = torch.zeros((len(batch), width), dtype=torch.long, device=graph.device)
        attention = torch.zeros_like(input_ids)
        spans = []
        for index, (_row, prompt, target) in enumerate(batch):
            tokens = prompt + target
            input_ids[index, : len(tokens)] = torch.tensor(tokens, device=graph.device)
            attention[index, : len(tokens)] = 1
            spans.append((len(prompt) - 1, target))
        output = graph.wrapper(
            input_ids=input_ids,
            attention_mask=attention,
            max_loops=4,
            stage2b_depth_enabled=True,
            stage2b_stage=graph.stage,
            stage2b_amplitude=graph.amplitude,
            stage2b_diagnostic_mode=graph.diagnostic_mode,
            return_loop_logits=True,
            use_cache=False,
            return_dict=True,
        )
        for name, value in (output.metrics or {}).items():
            if name.startswith("stage2b_") and "_loop_" in name:
                scalar = float(value.detach().float().abs().amax().cpu())
                activation_maxima[name] = max(activation_maxima.get(name, 0.0), scalar)
        loops = output.loop_logits[:, 0]
        for local, ((row, _prompt, _target), (first, targets)) in enumerate(zip(batch, spans)):
            per_loop = []
            for loop in range(4):
                logits = loops[local, loop, first : first + len(targets)].float()
                target_ids = torch.tensor(targets, device=logits.device)
                target_logits = logits.gather(-1, target_ids[:, None]).squeeze(-1)
                masked = logits.clone()
                masked.scatter_(-1, target_ids[:, None], -torch.inf)
                margins = target_logits - masked.max(dim=-1).values
                per_loop.append(float(margins.mean().cpu()))
            results.append(
                {
                    "kind": "paper2_stage2b_dev2_margin_row_v1",
                    "seed": seed,
                    "look": look,
                    "item_id": str(row["item_id"]),
                    "battery": str(row["battery"]),
                    "source_partition": str(row["partition"]),
                    "target_tokens": len(targets),
                    "per_loop_mean_teacher_token_margin": per_loop,
                    "k1_kl_role": "reported_by_training_objective_telemetry",
                }
            )
        print(f"stage2b_dev2_progress rows={min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    means = [
        sum(row["per_loop_mean_teacher_token_margin"][loop] for row in results) / len(results)
        for loop in range(4)
    ]
    summary = {
        "kind": "paper2_stage2b_dev2_margin_summary_v1",
        "seed": seed,
        "look": look,
        "rows": len(results),
        "per_loop_mean_teacher_token_margin": means,
        "transition_means": {
            f"k{index}_to_k{index + 1}": means[index] - means[index - 1]
            for index in range(1, 4)
        },
        "battery_counts": dict(sorted(Counter(row["battery"] for row in results).items())),
        "activation_maxima": dict(sorted(activation_maxima.items())),
    }
    return results, summary
