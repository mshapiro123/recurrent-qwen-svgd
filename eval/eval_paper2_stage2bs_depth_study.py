"""Score-only write-schedule variants for the locked Stage 2B-S depth study."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import numpy as np
from transformers import AutoTokenizer

from eval.eval_paper2_phase3_p34_task_inference import (
    P34CachedPrefix,
    P34NextTokenOutput,
    current_position_mask,
)
from eval.eval_paper2_stage2b_campaign import Stage2BTaskInferenceGraph
from eval.eval_paper2_stage2b_campaign import _forced_target
from eval.eval_paper2_phase3_p31_references import MODEL_SPECS
from eval.eval_paper2_phase3_p34_task_trajectory import score_generation
from eval.eval_paper2_stage2b_autopsy import (
    _apply_state,
    _build_model,
    _checkpoint_state,
    _named_trainable_state,
    _state_digest,
)
from models.paper2_stage2b_depth import Stage2BDepthTrace, Stage2BReentryOutput
from training.paper2_stage2bs_depth_study import (
    EXPECTED_INITIALIZATION_STATE_DIGESTS,
    EXPECTED_NATIVE_COUNTS,
    INITIALIZATION_SEED_BASE,
    SCHEDULES,
    load_lock,
    schedule_amplitudes,
    sha256_file,
)


RUN_KIND = "paper2_stage2bs_depth_study_v1"


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


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


@dataclass(frozen=True)
class ScheduleProvenance:
    schedule: str
    k: int
    k_semantics: str
    amplitude: float
    identity_passes: int
    sidecar_updates: int
    bridge_writes: int
    recurrent_reentries: int
    evaluator: str = "Stage2BScheduleGraph_v1"
    substrate: str = "Stage2BTaskInferenceGraph_authoritative_wrapper_and_head"

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Stage2BScheduleGraph:
    """Authoritative Stage 2B substrate with a provenance-bound write schedule.

    Native execution delegates unchanged to ``Stage2BTaskInferenceGraph``.
    Score-only variants patch only the attachment schedule for the duration of
    one forward call and restore every method before returning.
    """

    def __init__(
        self,
        *,
        wrapper: Any,
        schedule: str,
        k: int,
        amplitude: float,
        incremental_cache: bool = False,
    ) -> None:
        if schedule not in SCHEDULES:
            raise ValueError(f"unknown Stage 2B-S schedule: {schedule}")
        if not 1 <= int(k) <= 4:
            raise ValueError("Stage 2B-S K must be in [1, 4]")
        if float(amplitude) not in {0.0, 0.02, 0.05}:
            raise ValueError("Stage 2B-S amplitude is outside the locked grid")
        if schedule not in SCHEDULES[:2] and float(amplitude) != 0.05:
            raise ValueError("only native and deferred schedules carry the amplitude cross")
        self.wrapper = wrapper
        self.schedule = schedule
        self.k = int(k)
        self.flow_loops = int(k)
        self.amplitude = float(amplitude)
        self.incremental_cache = bool(incremental_cache)
        self._base_hidden: torch.Tensor | None = None
        self._logical_updates = 0

    @property
    def device(self) -> torch.device:
        return next(self.wrapper.parameters()).device

    @property
    def runtime_recurrent_passes(self) -> int:
        if self.schedule == "native_interleaved":
            return self.k
        if self.schedule == "partial_interleave_pairs":
            return 1 + math.ceil(self.k / 2)
        return 1

    @property
    def provenance(self) -> ScheduleProvenance:
        if self.schedule == "native_interleaved":
            updates = max(self.k - 1, 0)
            writes = updates if self.amplitude > 0 else 0
            reentries = updates
            semantics = "total_recurrent_passes_including_identity_pass"
        elif self.schedule == "deferred_terminal_write_no_reentry":
            updates = self.k
            writes = 1 if self.amplitude > 0 else 0
            reentries = 0
            semantics = "sidecar_updates_after_one_identity_pass"
        elif self.schedule == "per_loop_write_no_reentry":
            updates = self.k
            writes = self.k
            reentries = 0
            semantics = "sidecar_updates_after_one_identity_pass"
        else:
            updates = self.k
            writes = math.ceil(self.k / 2)
            reentries = writes
            semantics = "sidecar_updates_after_one_identity_pass"
        return ScheduleProvenance(
            schedule=self.schedule,
            k=self.k,
            k_semantics=semantics,
            amplitude=self.amplitude,
            identity_passes=1,
            sidecar_updates=updates,
            bridge_writes=writes,
            recurrent_reentries=reentries,
        )

    def prepare_probe_batch(self, _rows: Sequence[Mapping[str, Any]]) -> None:
        return None

    def _advance(
        self,
        *,
        trace: Stage2BDepthTrace,
        context_hidden: torch.Tensor,
        previous_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        logical_index: int,
        write: bool,
    ) -> torch.Tensor:
        attachment = self.wrapper.stage2b_depth_attachment
        if trace.lane_state is None or trace.initial_lane_state is None:
            raise RuntimeError("Stage 2B-S schedule advanced before scratch initialization")
        context = attachment._masked_mean(context_hidden.float(), attention_mask)
        prompt_context = attachment._masked_mean(trace.reference.float(), attention_mask)
        step_state = trace.lane_state
        carry = step_state - trace.initial_lane_state
        step = attachment.flow.step(
            step_state,
            context,
            logical_index,
            prompt_context=prompt_context,
            dynamic_routing=False,
            constitutive_active=True,
            inherited_flow_active=True,
            forced_lane_one=False,
        )
        innovation_norm = step.flow_update.float().square().mean(dim=-1).sqrt().mean(dim=(1, 2))
        control = attachment.control(
            scratch=step.read_state,
            previous=trace.control_state,
            innovation_norm=innovation_norm,
            student_entropy=context.new_zeros((context.shape[0],)),
            top2_margin=context.new_zeros((context.shape[0],)),
            position_bucket=torch.zeros(
                context.shape[0], dtype=torch.long, device=context.device
            ),
        )
        trace.lane_state = step.state
        trace.control_state = control
        trace.routing_steps.append(step)
        trace.carry_contributions.append(carry)
        if not write or self.amplitude == 0.0:
            trace.writeback_ratios.append(
                context.new_zeros(previous_hidden.shape[:2])
            )
            trace.position_gates.append(
                context.new_zeros((*previous_hidden.shape[:2], 1))
            )
            return previous_hidden
        attachment.bridge.set_gate_ceiling(self.amplitude)
        bridge = attachment.bridge(
            h0=trace.reference,
            previous=previous_hidden,
            scratch=step.read_state,
            control_state=control,
            loop_index=logical_index,
            active=True,
            write_position_mask=attention_mask.bool().unsqueeze(-1),
        )
        trace.writeback_ratios.append(bridge.realized_writeback_ratio)
        trace.position_gates.append(bridge.position_gate)
        return bridge.hidden

    @contextmanager
    def _patched_schedule(self) -> Iterator[None]:
        if self.schedule == "native_interleaved":
            yield
            return
        attachment = self.wrapper.stage2b_depth_attachment
        original_observe = attachment.observe
        original_reenter = attachment.reenter
        original_run_layers = self.wrapper._run_layer_range
        captured: dict[str, torch.Tensor] = {}
        self._base_hidden = None
        self._logical_updates = 0

        if self.schedule in {
            "deferred_terminal_write_no_reentry",
            "per_loop_write_no_reentry",
        }:

            def run_layers(*args: Any, **kwargs: Any):
                if (
                    int(kwargs.get("start", -1)) == self.wrapper.layer_split.recurrent_end
                    and int(kwargs.get("end", -1)) == len(self.wrapper.qwen.layers)
                ):
                    captured["recurrent_hidden"] = kwargs["hidden_states"]
                return original_run_layers(*args, **kwargs)

            def observe(**kwargs: Any):
                trace = original_observe(**kwargs)
                if int(kwargs["loop_index"]) != 0:
                    return trace
                coda_hidden = kwargs["coda_hidden"]
                context_hidden = captured.get("recurrent_hidden")
                if context_hidden is None:
                    raise RuntimeError("Stage 2B-S no-reentry schedule missed recurrent context")
                self._base_hidden = coda_hidden.detach().clone()
                current = coda_hidden
                for index in range(self.k):
                    write = (
                        self.schedule == "per_loop_write_no_reentry"
                        or index == self.k - 1
                    )
                    current = self._advance(
                        trace=trace,
                        context_hidden=context_hidden,
                        previous_hidden=current,
                        attention_mask=kwargs["attention_mask"],
                        logical_index=index,
                        write=write,
                    )
                coda_hidden.copy_(current)
                self._logical_updates = self.k
                return trace

            self.wrapper._run_layer_range = run_layers
            attachment.observe = observe
        else:

            def reenter(**kwargs: Any) -> Stage2BReentryOutput:
                remaining = self.k - self._logical_updates
                if remaining <= 0:
                    raise RuntimeError("Stage 2B-S partial schedule exceeded its K")
                group = min(2, remaining)
                current = kwargs["recurrent_hidden"]
                for offset in range(group):
                    logical_index = self._logical_updates
                    current = self._advance(
                        trace=kwargs["trace"],
                        context_hidden=kwargs["recurrent_hidden"],
                        previous_hidden=current,
                        attention_mask=kwargs["attention_mask"],
                        logical_index=logical_index,
                        write=offset == group - 1,
                    )
                    self._logical_updates += 1
                return Stage2BReentryOutput(hidden=current, trace=kwargs["trace"])

            attachment.reenter = reenter
        try:
            yield
        finally:
            attachment.observe = original_observe
            attachment.reenter = original_reenter
            self.wrapper._run_layer_range = original_run_layers

    def _run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        prelude_cache: Any = None,
        loop_caches: tuple[Any, ...] | None = None,
        current_token_only: bool = False,
        current_positions: torch.Tensor | None = None,
    ) -> P34NextTokenOutput:
        if self.schedule == "native_interleaved":
            graph = Stage2BTaskInferenceGraph(
                wrapper=self.wrapper,
                stage="M2",
                amplitude=self.amplitude,
                flow_loops=self.k,
                diagnostic_mode="zero_write" if self.amplitude == 0.0 else "standard",
                sparse_loop_projection=True,
            )
            return graph._run(
                input_ids,
                attention_mask,
                prelude_cache=prelude_cache,
                loop_caches=loop_caches,
                current_token_only=current_token_only,
                current_positions=current_positions,
            )
        with self._patched_schedule():
            output = self.wrapper(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_loops=self.runtime_recurrent_passes,
                stage2b_depth_enabled=True,
                stage2b_stage="M2",
                stage2b_amplitude=0.05,
                stage2b_score_only_sparse_logits=True,
                stage2b_loop_past_key_values=loop_caches,
                return_loop_logits=True,
                past_key_values=prelude_cache,
                use_cache=loop_caches is not None,
                return_dict=True,
            )
        if self._logical_updates != self.k:
            raise RuntimeError(
                f"Stage 2B-S schedule executed {self._logical_updates} updates, expected {self.k}"
            )
        loops = output.loop_logits[:, 0]
        positions = (
            current_position_mask(attention_mask)[1]
            if current_positions is None
            else current_positions
        )
        batch = torch.arange(input_ids.shape[0], device=input_ids.device)
        if current_token_only:
            selected = loops[:, -1, -1, :]
            if self._base_hidden is not None:
                base = self.wrapper.lm_head(self._base_hidden[:, -1]).float()
            else:
                base = loops[:, 0, -1, :]
        else:
            selected = loops[batch, -1, positions, :]
            if self._base_hidden is not None:
                base_full = self.wrapper.lm_head(self._base_hidden)
                base = base_full[batch, positions, :]
            else:
                base = loops[batch, 0, positions, :]
        top2 = selected.float().topk(2, dim=-1).values
        metrics = output.metrics or {}
        rows = input_ids.shape[0]
        gate = metrics.get("stage2b_position_gate_mean")
        ratio = metrics.get("stage2b_writeback_ratio_mean")
        gate_rows = (
            gate.reshape(1).expand(rows).to(input_ids.device)
            if isinstance(gate, torch.Tensor)
            else torch.zeros(rows, device=input_ids.device)
        )
        ratio_rows = (
            ratio.reshape(1).expand(rows).to(input_ids.device)
            if isinstance(ratio, torch.Tensor)
            else torch.zeros(rows, device=input_ids.device)
        )
        return P34NextTokenOutput(
            augmented_logits=selected,
            base_logits=base,
            writeback_ratio=ratio_rows,
            position_gate=gate_rows,
            current_positions=positions,
            scratch_state=torch.empty((rows, 0, 0), device=input_ids.device),
            answer_token_margin=top2[:, 0] - top2[:, 1],
        )

    @torch.inference_mode()
    def sequence_logits(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return final-schedule logits at every sequence position."""

        if self.schedule == "native_interleaved":
            output = self.wrapper(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_loops=self.k,
                stage2b_depth_enabled=True,
                stage2b_stage="M2",
                stage2b_amplitude=(0.05 if self.amplitude == 0.0 else self.amplitude),
                stage2b_diagnostic_mode=(
                    "zero_write" if self.amplitude == 0.0 else "standard"
                ),
                stage2b_score_only_sparse_logits=True,
                return_loop_logits=True,
                use_cache=False,
                return_dict=True,
            )
        else:
            with self._patched_schedule():
                output = self.wrapper(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_loops=self.runtime_recurrent_passes,
                    stage2b_depth_enabled=True,
                    stage2b_stage="M2",
                    stage2b_amplitude=0.05,
                    stage2b_score_only_sparse_logits=True,
                    return_loop_logits=True,
                    use_cache=False,
                    return_dict=True,
                )
            if self._logical_updates != self.k:
                raise RuntimeError(
                    f"Stage 2B-S schedule executed {self._logical_updates} updates, expected {self.k}"
                )
        if output.loop_logits is None:
            raise RuntimeError("Stage 2B-S sequence margin read lacks loop logits")
        return output.loop_logits[:, 0, -1]

    @torch.inference_mode()
    def next_token(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        flow_loops: int | None = None,
    ) -> P34NextTokenOutput:
        if flow_loops is not None and int(flow_loops) != self.k:
            raise RuntimeError("Stage 2B-S per-call K changed")
        return self._run(input_ids, attention_mask)

    @torch.inference_mode()
    def prefill_cached(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[P34CachedPrefix, P34NextTokenOutput]:
        if not self.incremental_cache:
            state = P34CachedPrefix(
                input_ids=input_ids,
                hidden=torch.empty((input_ids.shape[0], input_ids.shape[1], 0), device=self.device),
                layer6_hidden=torch.empty(
                    (input_ids.shape[0], input_ids.shape[1], 0), device=self.device
                ),
                attention_mask=attention_mask,
                past_key_values=None,
            )
            return state, self._run(input_ids, attention_mask)
        from transformers.cache_utils import DynamicCache

        prelude_cache = DynamicCache(config=self.wrapper.config)
        loop_caches = tuple(
            DynamicCache(config=self.wrapper.config)
            for _ in range(self.runtime_recurrent_passes)
        )
        positions = current_position_mask(attention_mask)[1]
        result = self._run(
            input_ids,
            attention_mask,
            prelude_cache=prelude_cache,
            loop_caches=loop_caches,
            current_positions=positions,
        )
        state = P34CachedPrefix(
            input_ids=input_ids,
            hidden=torch.empty((input_ids.shape[0], input_ids.shape[1], 0), device=self.device),
            layer6_hidden=torch.empty(
                (input_ids.shape[0], input_ids.shape[1], 0), device=self.device
            ),
            attention_mask=attention_mask,
            past_key_values={"prelude": prelude_cache, "loops": loop_caches},
        )
        return state, result

    @torch.inference_mode()
    def advance_cached(
        self, *, state: P34CachedPrefix, selected_tokens: torch.Tensor
    ) -> tuple[P34CachedPrefix, P34NextTokenOutput]:
        if selected_tokens.ndim == 1:
            selected_tokens = selected_tokens[:, None]
        attention = torch.cat(
            [
                state.attention_mask,
                torch.ones_like(selected_tokens, dtype=state.attention_mask.dtype),
            ],
            dim=1,
        )
        if not self.incremental_cache:
            return self.prefill_cached(
                input_ids=torch.cat([state.input_ids, selected_tokens], dim=1),
                attention_mask=attention,
            )
        caches = state.past_key_values
        if not isinstance(caches, dict) or set(caches) != {"prelude", "loops"}:
            raise RuntimeError("Stage 2B-S incremental cache state changed")
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
            loop_caches=tuple(caches["loops"]),
            current_token_only=True,
            current_positions=positions,
        )
        updated = P34CachedPrefix(
            input_ids=torch.cat([state.input_ids, selected_tokens], dim=1),
            hidden=torch.empty((selected_tokens.shape[0], attention.shape[1], 0), device=self.device),
            layer6_hidden=torch.empty(
                (selected_tokens.shape[0], attention.shape[1], 0), device=self.device
            ),
            attention_mask=attention,
            past_key_values=caches,
        )
        return updated, output


@torch.inference_mode()
def score_margin_rows(
    *,
    graph: Stage2BScheduleGraph,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    forced_target: Any,
    seed: int,
    endpoint: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = [(row, *forced_target(tokenizer, row)) for row in rows]
    results: list[dict[str, Any]] = []
    for start in range(0, len(prepared), batch_size):
        batch_rows = prepared[start : start + batch_size]
        width = max(len(prompt) + len(target) for _row, prompt, target in batch_rows)
        input_ids = torch.zeros((len(batch_rows), width), dtype=torch.long, device=graph.device)
        attention = torch.zeros_like(input_ids)
        spans = []
        for index, (_row, prompt, target) in enumerate(batch_rows):
            tokens = prompt + target
            input_ids[index, : len(tokens)] = torch.tensor(tokens, device=graph.device)
            attention[index, : len(tokens)] = 1
            spans.append((len(prompt) - 1, target))
        logits_by_position = graph.sequence_logits(
            input_ids=input_ids, attention_mask=attention
        )
        for local, ((row, _prompt, target), (first, _targets)) in enumerate(
            zip(batch_rows, spans)
        ):
            logits = logits_by_position[local, first : first + len(target)].float()
            target_ids = torch.tensor(target, device=logits.device)
            target_logits = logits.gather(-1, target_ids[:, None]).squeeze(-1)
            masked = logits.clone()
            masked.scatter_(-1, target_ids[:, None], -torch.inf)
            margins = target_logits - masked.max(dim=-1).values
            results.append(
                {
                    "kind": "paper2_stage2bs_depth_margin_row_v1",
                    "seed": seed,
                    "endpoint": endpoint,
                    "item_id": str(row["item_id"]),
                    "battery": str(row["battery"]),
                    "target_tokens": len(target),
                    "mean_teacher_token_margin": float(margins.mean().cpu()),
                    "schedule_provenance": graph.provenance.as_dict(),
                }
            )
        print(
            f"stage2bs_depth_margin schedule={graph.schedule} k={graph.k} "
            f"rows={min(start + batch_size, len(rows))}/{len(rows)}",
            flush=True,
        )
    values = torch.tensor([row["mean_teacher_token_margin"] for row in results])
    by_battery: dict[str, list[float]] = defaultdict(list)
    for row in results:
        by_battery[str(row["battery"])].append(float(row["mean_teacher_token_margin"]))
    summary = {
        "rows": len(results),
        "mean_teacher_token_margin": float(values.mean()),
        "median_teacher_token_margin": float(values.median()),
        "positive_margin_fraction": float((values > 0).float().mean()),
        "quantiles": {
            str(q): float(torch.quantile(values, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "by_battery": {
            name: {
                "rows": len(items),
                "mean_teacher_token_margin": sum(items) / len(items),
            }
            for name, items in sorted(by_battery.items())
        },
        "battery_counts": dict(Counter(row["battery"] for row in results)),
        "schedule_provenance": graph.provenance.as_dict(),
    }
    return results, summary


def summarize_margin_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("Stage 2B-S margin summary has no rows")
    values = torch.tensor([float(row["mean_teacher_token_margin"]) for row in rows])
    by_battery: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_battery[str(row["battery"])].append(float(row["mean_teacher_token_margin"]))
    return {
        "rows": len(rows),
        "mean_teacher_token_margin": float(values.mean()),
        "median_teacher_token_margin": float(values.median()),
        "positive_margin_fraction": float((values > 0).float().mean()),
        "quantiles": {
            str(q): float(torch.quantile(values, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "by_battery": {
            name: {
                "rows": len(items),
                "mean_teacher_token_margin": sum(items) / len(items),
            }
            for name, items in sorted(by_battery.items())
        },
        "battery_counts": dict(Counter(str(row["battery"]) for row in rows)),
        "schedule_provenance": dict(rows[0]["schedule_provenance"]),
    }


def _runtime_receipt() -> dict[str, Any]:
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "weights_dtype": "bfloat16",
        "attention_backend": "sdpa",
    }


def _generation_rows(panel: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in panel
        if str(row["battery"]) in {"tier1", "gsm8k", "mbpp"}
    ]
    if len(rows) != 461:
        raise RuntimeError(f"Stage 2B-S generative slice changed: {len(rows)}")
    return rows


def _cell_slug(schedule: str, k: int, amplitude: float) -> str:
    gamma = f"{amplitude:.2f}".replace(".", "p")
    return f"{schedule}__k{k}__gamma_{gamma}"


def _score_generation_cell(
    *,
    graph: Stage2BScheduleGraph,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    endpoint: str,
    private_dir: Path,
    batch_size: int,
) -> list[dict[str, Any]]:
    slug = _cell_slug(graph.schedule, graph.k, graph.amplitude)
    final_path = private_dir / "generation" / f"{slug}.jsonl"
    partial_path = final_path.with_suffix(".partial.jsonl")
    expected = [str(row["item_id"]) for row in rows]
    provenance = graph.provenance.as_dict()

    def validate(cached: Sequence[Mapping[str, Any]], *, complete: bool) -> None:
        ids = [str(row["item_id"]) for row in cached]
        if len(ids) != len(set(ids)) or not set(ids).issubset(expected):
            raise RuntimeError(f"Stage 2B-S cached generation identity changed: {slug}")
        if complete and (len(ids) != len(expected) or set(ids) != set(expected)):
            raise RuntimeError(f"Stage 2B-S cached generation coverage changed: {slug}")
        for row in cached:
            if (
                int(row.get("seed", -1)) != seed
                or row.get("endpoint") != endpoint
                or row.get("schedule_provenance") != provenance
                or int(row.get("generation_batch_size", -1)) != batch_size
            ):
                raise RuntimeError(f"Stage 2B-S cached generation provenance changed: {slug}")

    if final_path.is_file():
        cached = read_jsonl(final_path)
        validate(cached, complete=True)
        return cached
    cached = read_jsonl(partial_path) if partial_path.is_file() else []
    validate(cached, complete=False)
    by_id = {str(row["item_id"]): dict(row) for row in cached}
    pending = [row for row in rows if str(row["item_id"]) not in by_id]

    def emit_batch(scored: list[dict[str, Any]]) -> None:
        for row in scored:
            item_id = str(row["item_id"])
            if item_id in by_id:
                raise RuntimeError(f"Stage 2B-S duplicate generation row: {item_id}")
            row.update(
                {
                    "kind": "paper2_stage2bs_depth_generation_row_v1",
                    "seed": seed,
                    "endpoint": endpoint,
                    "schedule_provenance": provenance,
                    "generation_batch_size": batch_size,
                }
            )
            by_id[item_id] = row
        ordered = [by_id[item_id] for item_id in expected if item_id in by_id]
        validate(ordered, complete=False)
        write_jsonl(partial_path, ordered)
        print(
            f"stage2bs_depth_generation cell={slug} rows={len(ordered)}/{len(expected)}",
            flush=True,
        )

    if pending:
        score_generation(
            graph,
            tokenizer,
            pending,
            batch_size=batch_size,
            emit_batch=emit_batch,
        )
    result = [by_id[item_id] for item_id in expected if item_id in by_id]
    validate(result, complete=True)
    write_jsonl(final_path, result)
    partial_path.unlink(missing_ok=True)
    return result


def _score_margin_cell(
    *,
    graph: Stage2BScheduleGraph,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    endpoint: str,
    private_dir: Path,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    slug = _cell_slug(graph.schedule, graph.k, graph.amplitude)
    final_path = private_dir / "margins" / f"{slug}.jsonl"
    partial_path = final_path.with_suffix(".partial.jsonl")
    expected = [str(row["item_id"]) for row in rows]
    provenance = graph.provenance.as_dict()
    cached = read_jsonl(final_path) if final_path.is_file() else (
        read_jsonl(partial_path) if partial_path.is_file() else []
    )
    ids = [str(row["item_id"]) for row in cached]
    if len(ids) != len(set(ids)) or not set(ids).issubset(expected):
        raise RuntimeError(f"Stage 2B-S cached margin identity changed: {slug}")
    for row in cached:
        if (
            int(row.get("seed", -1)) != seed
            or row.get("endpoint") != endpoint
            or row.get("schedule_provenance") != provenance
            or int(row.get("margin_batch_size", -1)) != batch_size
        ):
            raise RuntimeError(f"Stage 2B-S cached margin provenance changed: {slug}")
    by_id = {str(row["item_id"]): dict(row) for row in cached}
    pending = [row for row in rows if str(row["item_id"]) not in by_id]
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        scored, _summary = score_margin_rows(
            graph=graph,
            tokenizer=tokenizer,
            rows=chunk,
            forced_target=_forced_target,
            seed=seed,
            endpoint=endpoint,
            batch_size=len(chunk),
        )
        for row in scored:
            item_id = str(row["item_id"])
            if item_id in by_id:
                raise RuntimeError(f"Stage 2B-S duplicate margin row: {item_id}")
            row["margin_batch_size"] = batch_size
            by_id[item_id] = row
        ordered = [by_id[item_id] for item_id in expected if item_id in by_id]
        write_jsonl(partial_path, ordered)
    result = [by_id[item_id] for item_id in expected if item_id in by_id]
    if len(result) != len(expected):
        raise RuntimeError(f"Stage 2B-S margin coverage incomplete: {slug}")
    write_jsonl(final_path, result)
    partial_path.unlink(missing_ok=True)
    return result, summarize_margin_rows(result)


def _task_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_rows: Mapping[str, Mapping[str, Any]],
    base_rows: Mapping[str, Mapping[str, Any]],
    native_k1_rows: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    by_battery: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_battery[str(source_rows[str(row["item_id"])]["battery"])].append(row)
    current = sum(bool(row["augmented_correct"]) for row in rows)
    base = sum(
        bool(base_rows[str(row["item_id"])].get("correct", base_rows[str(row["item_id"])].get("augmented_correct")))
        for row in rows
    )
    native_k1 = sum(
        bool(native_k1_rows[str(row["item_id"])]["augmented_correct"]) for row in rows
    )
    return {
        "rows": len(rows),
        "correct": current,
        "base_correct": base,
        "native_k1_correct": native_k1,
        "delta_vs_base_rows": current - base,
        "delta_vs_native_k1_rows": current - native_k1,
        "by_battery": {
            name: {
                "rows": len(items),
                "correct": sum(bool(row["augmented_correct"]) for row in items),
            }
            for name, items in sorted(by_battery.items())
        },
        "schedule_provenance": dict(provenance),
        "both_comparators_reported": True,
    }


def _load_banked_preflight(
    *,
    args: argparse.Namespace,
    wrapper: Any,
    expected_rows: Sequence[Mapping[str, Any]],
    initialization_digest: str,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    if args.banked_preflight_receipt is None or args.banked_preflight_private is None:
        raise RuntimeError("Stage 2B-S cascade requires banked preflight inputs")
    receipt = json.loads(args.banked_preflight_receipt.read_text(encoding="utf-8"))
    expected_counts = EXPECTED_NATIVE_COUNTS[args.seed]
    if int(receipt.get("seed", -1)) != args.seed:
        raise RuntimeError("Stage 2B-S banked preflight seed changed")
    if receipt.get("observed_correct_by_k") != expected_counts:
        raise RuntimeError("Stage 2B-S banked preflight counts changed")
    if receipt.get("initialization_state_digest") != initialization_digest:
        raise RuntimeError("Stage 2B-S banked preflight state identity changed")
    expected_ids = [str(row["item_id"]) for row in expected_rows]
    by_k: dict[int, list[dict[str, Any]]] = {}
    for k in range(1, 5):
        graph = Stage2BScheduleGraph(
            wrapper=wrapper,
            schedule="native_interleaved",
            k=k,
            amplitude=0.05,
            incremental_cache=True,
        )
        path = (
            args.banked_preflight_private
            / "generation"
            / f"{_cell_slug('native_interleaved', k, 0.05)}.jsonl"
        )
        rows = read_jsonl(path)
        if [str(row["item_id"]) for row in rows] != expected_ids:
            raise RuntimeError(f"Stage 2B-S banked preflight row identity changed at K{k}")
        provenance = graph.provenance.as_dict()
        if any(
            int(row.get("seed", -1)) != args.seed
            or row.get("endpoint") != "initialization"
            or row.get("schedule_provenance") != provenance
            for row in rows
        ):
            raise RuntimeError(f"Stage 2B-S banked preflight provenance changed at K{k}")
        observed = sum(bool(row["augmented_correct"]) for row in rows)
        if observed != expected_counts[k - 1]:
            raise RuntimeError(f"Stage 2B-S banked preflight score changed at K{k}")
        by_k[k] = rows
    current = {
        "kind": "paper2_stage2bs_depth_banked_preflight_v2",
        "status": "BANKED_PASS",
        "seed": args.seed,
        "observed_correct_by_k": expected_counts,
        "expected_correct_by_k": expected_counts,
        "source_receipt": str(args.banked_preflight_receipt),
        "source_receipt_sha256": sha256_file(args.banked_preflight_receipt),
        "source_private": str(args.banked_preflight_private),
        "source_session_id": receipt.get("session_id"),
        "runtime": receipt.get("runtime"),
        "initialization_seed": receipt.get("initialization_seed"),
        "initialization_state_digest": initialization_digest,
        "lock_sha256": sha256_file(args.lock),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    return by_k, current


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_lock(args.lock)
    if any(term in str(value).casefold() for value in vars(args).values() for term in ("confirm", "eval_e")):
        raise RuntimeError("Stage 2B-S depth study attempted sealed-partition contact")
    initialization_seed = INITIALIZATION_SEED_BASE + args.seed
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    wrapper, checkpoint_chain, _groups = _build_model(args)
    initialization = _named_trainable_state(wrapper)
    initialization_digest = _state_digest(initialization)
    expected_initialization_digest = EXPECTED_INITIALIZATION_STATE_DIGESTS[args.seed]
    if initialization_digest != expected_initialization_digest:
        raise RuntimeError(
            "Stage 2B-S initialization state changed before scoring: "
            f"seed={args.seed} expected={expected_initialization_digest} "
            f"observed={initialization_digest}"
        )
    stop = _checkpoint_state(args.stop_checkpoint, expected_sha256=args.stop_sha256)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SPECS["base"]["model"], revision=MODEL_SPECS["base"]["revision"]
    )
    panel = read_jsonl(args.dev1_panel)
    generation = _generation_rows(panel)
    source_rows = {str(row["item_id"]): row for row in panel}
    base_rows = {str(row["item_id"]): row for row in read_jsonl(args.base_scores)}
    if not {str(row["item_id"]) for row in generation}.issubset(base_rows):
        raise RuntimeError("Stage 2B-S base comparator coverage is incomplete")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)

    _apply_state(wrapper, initialization)
    if args.cascade_stage == "direct":
        preflight_rows, preflight = _load_banked_preflight(
            args=args,
            wrapper=wrapper,
            expected_rows=generation,
            initialization_digest=initialization_digest,
        )
    else:
        preflight_rows = {}
        preflight_counts = []
        for k in range(1, 5):
            graph = Stage2BScheduleGraph(
                wrapper=wrapper,
                schedule="native_interleaved",
                k=k,
                amplitude=0.05,
                incremental_cache=True,
            )
            rows = _score_generation_cell(
                graph=graph,
                tokenizer=tokenizer,
                rows=generation,
                seed=args.seed,
                endpoint="initialization",
                private_dir=args.private_dir / "preflight" / args.session_id,
                batch_size=args.generation_batch_size,
            )
            preflight_rows[k] = rows
            preflight_counts.append(sum(bool(row["augmented_correct"]) for row in rows))
        expected = EXPECTED_NATIVE_COUNTS[args.seed]
        preflight = {
            "kind": "paper2_stage2bs_depth_native_preflight_v2",
            "status": "PASS" if preflight_counts == expected else "STOP_MISMATCH",
            "seed": args.seed,
            "observed_correct_by_k": preflight_counts,
            "expected_correct_by_k": expected,
            "runtime": _runtime_receipt(),
            "session_id": args.session_id,
            "initialization_seed": initialization_seed,
            "initialization_state_digest": initialization_digest,
            "lock_sha256": sha256_file(args.lock),
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "confirm_scored": False,
            "eval_e_scored": False,
        }
    atomic_json(args.output_dir / "preflight.json", preflight)
    if preflight["status"] not in {"PASS", "BANKED_PASS"}:
        raise RuntimeError(
            f"Stage 2B-S native preflight mismatch seed={args.seed}: {preflight_counts}"
        )
    if args.preflight_only:
        result = {
            "kind": RUN_KIND,
            "status": "preflight_pass_score_only",
            "seed": args.seed,
            "lock_sha256": sha256_file(args.lock),
            "checkpoint_chain": checkpoint_chain,
            "runtime": _runtime_receipt(),
            "preflight": preflight,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "confirm_scored": False,
            "eval_e_scored": False,
            "variant_cells_scored": 0,
        }
        atomic_json(args.output_dir / "summary.json", result)
        return result

    if args.cascade_stage == "direct":
        native_k1 = {str(row["item_id"]): row for row in preflight_rows[1]}
        endpoint_private = args.private_dir / "cascade_direct" / "initialization"
        cells = []
        for k in range(1, 5):
            graph = Stage2BScheduleGraph(
                wrapper=wrapper,
                schedule="deferred_terminal_write_no_reentry",
                k=k,
                amplitude=0.05,
                incremental_cache=True,
            )
            rows = _score_generation_cell(
                graph=graph,
                tokenizer=tokenizer,
                rows=generation,
                seed=args.seed,
                endpoint="initialization",
                private_dir=endpoint_private,
                batch_size=args.generation_batch_size,
            )
            cells.append(
                {
                    "seed": args.seed,
                    "endpoint": "initialization",
                    "schedule": graph.schedule,
                    "k": k,
                    "amplitude": graph.amplitude,
                    **_task_summary(
                        rows,
                        source_rows=source_rows,
                        base_rows=base_rows,
                        native_k1_rows=native_k1,
                        provenance=graph.provenance.as_dict(),
                    ),
                }
            )
        if _state_digest(_named_trainable_state(wrapper)) != initialization_digest:
            raise RuntimeError("Stage 2B-S direct discriminator mutated initialization")
        result = {
            "kind": RUN_KIND,
            "status": "cascade_direct_complete_score_only",
            "seed": args.seed,
            "lock_sha256": sha256_file(args.lock),
            "checkpoint_chain": checkpoint_chain,
            "state_digests": {"initialization": initialization_digest},
            "runtime": _runtime_receipt(),
            "preflight": preflight,
            "panels": {"generative_rows": len(generation), "dev2_margin_rows_scored": 0},
            "cells": cells,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "confirm_scored": False,
            "eval_e_scored": False,
        }
        atomic_json(args.output_dir / "summary.json", result)
        return result

    reference = {str(row["item_id"]): row for row in read_jsonl(args.reference_rows)}
    manifest = read_jsonl(args.dev2_manifest)
    dev2 = [reference[str(row["item_id"])] for row in manifest]
    if len(dev2) != 2048:
        raise RuntimeError(f"Stage 2B-S DEV-2 changed: {len(dev2)}")

    all_cells = []
    margin_references: dict[str, dict[str, Mapping[str, Any]]] = {}
    endpoint_states = (("initialization", initialization), ("ema_step_1000", stop))
    for endpoint, state in endpoint_states:
        _apply_state(wrapper, state)
        endpoint_digest = _state_digest(state)
        endpoint_private = args.private_dir / endpoint
        endpoint_cells = []
        generation_by_cell: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
        for schedule in SCHEDULES:
            for amplitude in schedule_amplitudes(schedule):
                for k in range(1, 5):
                    graph = Stage2BScheduleGraph(
                        wrapper=wrapper,
                        schedule=schedule,
                        k=k,
                        amplitude=amplitude,
                        incremental_cache=True,
                    )
                    if (
                        endpoint == "initialization"
                        and schedule == "native_interleaved"
                        and amplitude == 0.05
                    ):
                        rows = preflight_rows[k]
                        canonical = (
                            endpoint_private
                            / "generation"
                            / f"{_cell_slug(schedule, k, amplitude)}.jsonl"
                        )
                        write_jsonl(canonical, rows)
                    else:
                        rows = _score_generation_cell(
                            graph=graph,
                            tokenizer=tokenizer,
                            rows=generation,
                            seed=args.seed,
                            endpoint=endpoint,
                            private_dir=endpoint_private,
                            batch_size=args.generation_batch_size,
                        )
                    generation_by_cell[(schedule, k, amplitude)] = rows
        native_k1 = {
            str(row["item_id"]): row
            for row in generation_by_cell[("native_interleaved", 1, 0.05)]
        }
        for schedule in SCHEDULES:
            for amplitude in schedule_amplitudes(schedule):
                for k in range(1, 5):
                    graph = Stage2BScheduleGraph(
                        wrapper=wrapper,
                        schedule=schedule,
                        k=k,
                        amplitude=amplitude,
                        incremental_cache=False,
                    )
                    rows = generation_by_cell[(schedule, k, amplitude)]
                    margin_rows, margin_summary = _score_margin_cell(
                        graph=graph,
                        tokenizer=tokenizer,
                        rows=dev2,
                        seed=args.seed,
                        endpoint=endpoint,
                        private_dir=endpoint_private,
                        batch_size=args.margin_batch_size,
                    )
                    margin_by_id = {str(row["item_id"]): row for row in margin_rows}
                    if schedule == "native_interleaved" and k == 1 and amplitude == 0.05:
                        margin_references[endpoint] = margin_by_id
                    cell = {
                        "seed": args.seed,
                        "endpoint": endpoint,
                        "schedule": schedule,
                        "k": k,
                        "amplitude": amplitude,
                        **_task_summary(
                            rows,
                            source_rows=source_rows,
                            base_rows=base_rows,
                            native_k1_rows=native_k1,
                            provenance=graph.provenance.as_dict(),
                        ),
                        "dev2_margin": margin_summary,
                    }
                    endpoint_cells.append(cell)
                    all_cells.append(cell)
        reference_margins = margin_references[endpoint]
        for cell in endpoint_cells:
            slug = _cell_slug(cell["schedule"], cell["k"], cell["amplitude"])
            path = endpoint_private / "margins" / f"{slug}.jsonl"
            rows = read_jsonl(path)
            for row in rows:
                row["delta_vs_native_k1_margin"] = (
                    float(row["mean_teacher_token_margin"])
                    - float(reference_margins[str(row["item_id"])]["mean_teacher_token_margin"])
                )
            write_jsonl(path, rows)
            cell["dev2_margin"]["mean_delta_vs_native_k1_margin"] = sum(
                float(row["delta_vs_native_k1_margin"]) for row in rows
            ) / len(rows)
        if _state_digest(_named_trainable_state(wrapper)) != endpoint_digest:
            raise RuntimeError(f"Stage 2B-S score-only evaluation mutated {endpoint}")
        atomic_json(
            args.output_dir / f"{endpoint}.json",
            {
                "seed": args.seed,
                "endpoint": endpoint,
                "state_digest": endpoint_digest,
                "cells": endpoint_cells,
            },
        )

    result = {
        "kind": RUN_KIND,
        "status": "complete_score_only",
        "seed": args.seed,
        "lock_sha256": sha256_file(args.lock),
        "checkpoint_chain": checkpoint_chain,
        "state_digests": {
            "initialization": _state_digest(initialization),
            "ema_step_1000": _state_digest(stop),
        },
        "runtime": _runtime_receipt(),
        "preflight": preflight,
        "panels": {
            "generative_rows": len(generation),
            "dev2_margin_rows": len(dev2),
            "dev2_manifest_sha256": sha256_file(args.dev2_manifest),
        },
        "cells": all_cells,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    atomic_json(args.output_dir / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--dev1_panel", type=Path, required=True)
    parser.add_argument("--dev2_manifest", type=Path, required=True)
    parser.add_argument("--reference_rows", type=Path, required=True)
    parser.add_argument("--base_scores", type=Path, required=True)
    parser.add_argument("--stop_checkpoint", type=Path, required=True)
    parser.add_argument("--stop_sha256", required=True)
    parser.add_argument("--migrated", type=Path, required=True)
    parser.add_argument("--migrated_sha256", required=True)
    parser.add_argument("--p33", type=Path, required=True)
    parser.add_argument("--p33_sha256", required=True)
    parser.add_argument("--i1", type=Path, required=True)
    parser.add_argument("--i1_sha256", required=True)
    parser.add_argument("--p34", type=Path, required=True)
    parser.add_argument("--p34_sha256", required=True)
    parser.add_argument("--p35", type=Path, required=True)
    parser.add_argument("--p35_sha256", required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--margin_batch_size", type=int, default=2)
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--cascade_stage", choices=("direct",))
    parser.add_argument("--banked_preflight_receipt", type=Path)
    parser.add_argument("--banked_preflight_private", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run_study(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
