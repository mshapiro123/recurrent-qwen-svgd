"""Pure protocol helpers for Paper Two Phase T1 and its P0 pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any, Iterable

from training.internal_think_token_spec import INTERNAL_CONTROL_TOKENS


CONTINUE_CLASS = 0
STOP_CLASS = 1
PILOT_STEPS = (500, 1000, 1500)
TRAINED_DEPTHS = tuple(range(1, 9))


@dataclass(frozen=True)
class PilotCell:
    cell_id: str
    control_loss_lambda: float
    stop_to_continue_ratio: float
    reference: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateTrieContract:
    """Exact candidate sequences and their shared teacher-forced prefixes."""

    prompt_token_count: int
    candidate_values: tuple[str, ...]
    candidate_token_ids: tuple[tuple[int, ...], ...]
    scoring_prefixes: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_token_count": self.prompt_token_count,
            "candidate_values": list(self.candidate_values),
            "candidate_token_ids": [list(tokens) for tokens in self.candidate_token_ids],
            "scoring_prefixes": [list(tokens) for tokens in self.scoring_prefixes],
        }


def _number_slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def pilot_grid() -> list[PilotCell]:
    """Return the preregistered nine-cell grid plus lambda-zero reference."""

    cells = [
        PilotCell(
            cell_id="lambda0_reference",
            control_loss_lambda=0.0,
            stop_to_continue_ratio=1.0,
            reference=True,
        )
    ]
    for loss_lambda in (0.5, 1.0, 2.0):
        for ratio in (1.0, 3.5, 7.0):
            cells.append(
                PilotCell(
                    cell_id=(
                        f"lambda{_number_slug(loss_lambda)}_ratio{_number_slug(ratio)}"
                    ),
                    control_loss_lambda=loss_lambda,
                    stop_to_continue_ratio=ratio,
                )
            )
    return cells


def candidate_trie_edges(
    contract: CandidateTrieContract,
) -> dict[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Map every scored prefix to ``(candidate_index, next_token_id)`` edges."""

    edges: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for candidate_index, sequence in enumerate(contract.candidate_token_ids):
        for offset, next_token_id in enumerate(sequence):
            edges.setdefault(sequence[:offset], []).append((candidate_index, next_token_id))
    return {prefix: tuple(values) for prefix, values in edges.items()}


def build_candidate_trie_contract(
    tokenizer: Any,
    *,
    prompt: str,
    candidate_values: Iterable[str | int],
) -> CandidateTrieContract:
    """Build a prompt-boundary-safe trie for variable-length candidates."""

    values = tuple(str(value) for value in candidate_values)
    if not values:
        raise ValueError("candidate_values cannot be empty")
    if len(set(values)) != len(values):
        raise ValueError("candidate_values must be unique")
    prompt_ids = list(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    suffixes: list[tuple[int, ...]] = []
    for value in values:
        full_ids = list(
            tokenizer(prompt + f" {value}", add_special_tokens=True)["input_ids"]
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise AssertionError(
                "Candidate tokenization changed the prompt prefix; exact fast scoring "
                f"is unavailable for symbol {value}"
            )
        suffix = tuple(int(token_id) for token_id in full_ids[len(prompt_ids) :])
        if not suffix:
            raise AssertionError(f"Candidate symbol {value} has an empty token suffix")
        suffixes.append(suffix)
    if len(set(suffixes)) != len(values):
        raise AssertionError("P0 candidate token sequences are not unique")
    prefixes = sorted(
        {suffix[:offset] for suffix in suffixes for offset in range(len(suffix))},
        key=lambda value: (len(value), value),
    )
    return CandidateTrieContract(
        prompt_token_count=len(prompt_ids),
        candidate_values=values,
        candidate_token_ids=tuple(suffixes),
        scoring_prefixes=tuple(prefixes),
    )


def control_targets_for_depth(depth: int, *, max_loops: int) -> list[int]:
    """Return continue before ``depth`` and stop exactly at ``depth``."""

    depth = int(depth)
    max_loops = int(max_loops)
    if depth < 1 or depth > max_loops:
        raise ValueError(f"depth must be within [1, {max_loops}], got {depth}")
    return [CONTINUE_CLASS] * (depth - 1) + [STOP_CLASS]


def aggregate_control_label_counts(depths: Iterable[int]) -> dict[str, int]:
    continue_count = 0
    stop_count = 0
    for depth in depths:
        targets = control_targets_for_depth(int(depth), max_loops=max(TRAINED_DEPTHS))
        continue_count += targets.count(CONTINUE_CLASS)
        stop_count += targets.count(STOP_CLASS)
    return {
        "continue": continue_count,
        "stop": stop_count,
        "total": continue_count + stop_count,
    }


def class_weights_from_ratio(
    *,
    stop_to_continue_ratio: float,
    continue_count: int,
    stop_count: int,
) -> tuple[float, float]:
    """Normalize class weights to mean one over the realized labels."""

    ratio = float(stop_to_continue_ratio)
    continue_count = int(continue_count)
    stop_count = int(stop_count)
    if ratio <= 0.0:
        raise ValueError("stop_to_continue_ratio must be positive")
    if continue_count <= 0 or stop_count <= 0:
        raise ValueError("both control classes require positive counts")
    total = continue_count + stop_count
    continue_weight = total / (continue_count + ratio * stop_count)
    stop_weight = ratio * continue_weight
    return float(continue_weight), float(stop_weight)


def augment_control_row(row: dict[str, Any]) -> dict[str, Any]:
    """Insert a private readout position while preserving answer supervision."""

    augmented = dict(row)
    prompt = str(row["prompt"])
    readout = INTERNAL_CONTROL_TOKENS[2]
    if readout in prompt:
        raise ValueError("row already contains the internal readout token")
    marker = "Answer:"
    if marker not in prompt:
        raise ValueError("control rows require an Answer: marker")
    prefix, suffix = prompt.rsplit(marker, 1)
    augmented["prompt"] = f"{prefix}{readout}\n{marker}{suffix}"
    depth = int(row.get("depth", row.get("target_loop_count", 0)))
    augmented["control_targets"] = control_targets_for_depth(
        depth,
        max_loops=max(TRAINED_DEPTHS),
    )
    augmented["control_active"] = True
    return augmented


def build_pilot_mixture_rows(
    source_rows: list[dict[str, Any]],
    *,
    seed: int,
    control_rows_per_depth: int = 175,
    rehearsal_rows_per_depth: int = 75,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the exact 70/30 P0 stream with balanced depths 1-8."""

    by_depth: dict[int, list[dict[str, Any]]] = {depth: [] for depth in TRAINED_DEPTHS}
    for row in source_rows:
        depth = int(row.get("depth", row.get("target_loop_count", 0)))
        if depth in by_depth:
            by_depth[depth].append(row)
    required = max(int(control_rows_per_depth), int(rehearsal_rows_per_depth))
    short = {depth: len(rows) for depth, rows in by_depth.items() if len(rows) < required}
    if short:
        raise ValueError(f"insufficient source rows by depth: {short}")

    mixed: list[dict[str, Any]] = []
    rng = random.Random(int(seed))
    for depth in TRAINED_DEPTHS:
        ordered = list(by_depth[depth])
        rng.shuffle(ordered)
        for index, row in enumerate(ordered[: int(control_rows_per_depth)]):
            control = augment_control_row(row)
            control["pilot_stream"] = "control"
            control["pilot_source_id"] = str(
                row.get("instance_id", row.get("id", f"depth{depth}_control{index}"))
            )
            mixed.append(control)
        rehearsal_pool = ordered[-int(rehearsal_rows_per_depth) :]
        for index, row in enumerate(rehearsal_pool):
            rehearsal = dict(row)
            rehearsal["control_active"] = False
            rehearsal["control_targets"] = []
            rehearsal["pilot_stream"] = "mechanism_rehearsal"
            rehearsal["pilot_source_id"] = str(
                row.get("instance_id", row.get("id", f"depth{depth}_rehearsal{index}"))
            )
            mixed.append(rehearsal)
    rng.shuffle(mixed)
    control_count = sum(bool(row["control_active"]) for row in mixed)
    rehearsal_count = len(mixed) - control_count
    manifest = {
        "seed": int(seed),
        "total_rows": len(mixed),
        "control_rows": control_count,
        "rehearsal_rows": rehearsal_count,
        "control_fraction": control_count / len(mixed),
        "rehearsal_fraction": rehearsal_count / len(mixed),
        "by_depth": {
            str(depth): {
                "control": sum(
                    bool(row["control_active"]) and int(row["depth"]) == depth
                    for row in mixed
                ),
                "rehearsal": sum(
                    not bool(row["control_active"]) and int(row["depth"]) == depth
                    for row in mixed
                ),
            }
            for depth in TRAINED_DEPTHS
        },
    }
    return mixed, manifest


def gate3_verdict(by_depth: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = {str(depth) for depth in TRAINED_DEPTHS}
    if set(by_depth) != expected:
        raise ValueError(f"Gate 3 requires depths 1-8, got {sorted(by_depth)}")
    depth_pass: dict[str, bool] = {}
    pooled_correct = 0
    pooled_total = 0
    for depth in TRAINED_DEPTHS:
        cell = by_depth[str(depth)]
        correct = int(cell["correct"])
        total = int(cell["total"])
        if total != 128:
            raise ValueError(f"Gate 3 depth {depth} requires 128 rows, got {total}")
        depth_pass[str(depth)] = correct >= 115
        pooled_correct += correct
        pooled_total += total
    pooled_pass = pooled_correct >= 922 and pooled_total == 1024
    return {
        "passed": all(depth_pass.values()) and pooled_pass,
        "by_depth_pass": depth_pass,
        "pooled_correct": pooled_correct,
        "pooled_total": pooled_total,
        "pooled_pass": pooled_pass,
        "minimum_correct_each_depth": 115,
        "minimum_correct_pooled": 922,
    }


def locate_readout_positions(
    input_ids: Any,
    *,
    readout_token_id: int,
    control_active: Any,
) -> Any:
    """Locate one private readout token per active row; inactive rows return -1."""

    import torch

    if input_ids.dim() != 2:
        raise ValueError("input_ids must be [batch, sequence]")
    active = control_active.to(device=input_ids.device, dtype=torch.bool).view(-1)
    if active.numel() != input_ids.shape[0]:
        raise ValueError("control_active must contain one value per batch row")
    matches = input_ids.eq(int(readout_token_id))
    counts = matches.sum(dim=-1)
    bad = active & counts.ne(1)
    if bool(bad.any()):
        indices = bad.nonzero(as_tuple=False).view(-1).tolist()
        observed = counts[bad].tolist()
        raise AssertionError(
            "Every active control row requires exactly one readout token; "
            f"rows={indices}, counts={observed}"
        )
    positions = torch.full(
        (input_ids.shape[0],),
        -1,
        device=input_ids.device,
        dtype=torch.long,
    )
    if bool(active.any()):
        positions[active] = matches[active].to(torch.long).argmax(dim=-1)
    return positions


def gather_control_examples(
    loop_logits: Any,
    *,
    readout_positions: Any,
    required_depths: Any,
    control_active: Any,
    continue_token_id: int,
    stop_token_id: int,
) -> tuple[Any, Any, Any, Any]:
    """Gather two-class logits for every active transition through exact stop."""

    import torch

    if loop_logits.dim() != 5 or loop_logits.shape[1] != 1:
        raise ValueError(
            "loop_logits must be [batch, one trajectory, loops, sequence, vocab]"
        )
    batch_size, _, max_loops, sequence_length, _ = loop_logits.shape
    positions = readout_positions.to(device=loop_logits.device, dtype=torch.long).view(-1)
    depths = required_depths.to(device=loop_logits.device, dtype=torch.long).view(-1)
    active = control_active.to(device=loop_logits.device, dtype=torch.bool).view(-1)
    if any(value.numel() != batch_size for value in (positions, depths, active)):
        raise ValueError("control tensors must contain one value per batch row")
    if bool((active & ((depths < 1) | (depths > max_loops))).any()):
        raise ValueError("active required depths must be within available loops")
    if bool((active & ((positions < 0) | (positions >= sequence_length))).any()):
        raise ValueError("active readout positions must be within the sequence")

    logit_rows: list[Any] = []
    target_rows: list[int] = []
    source_rows: list[int] = []
    source_loops: list[int] = []
    token_ids = torch.tensor(
        [int(continue_token_id), int(stop_token_id)],
        device=loop_logits.device,
        dtype=torch.long,
    )
    for row_index in range(batch_size):
        if not bool(active[row_index]):
            continue
        depth = int(depths[row_index].item())
        position = int(positions[row_index].item())
        for loop_number in range(1, depth + 1):
            logit_rows.append(
                loop_logits[row_index, 0, loop_number - 1, position].index_select(
                    dim=-1,
                    index=token_ids,
                )
            )
            target_rows.append(STOP_CLASS if loop_number == depth else CONTINUE_CLASS)
            source_rows.append(row_index)
            source_loops.append(loop_number)
    if not logit_rows:
        raise ValueError("batch contains no active control transitions")
    return (
        torch.stack(logit_rows),
        torch.tensor(target_rows, device=loop_logits.device, dtype=torch.long),
        torch.tensor(source_rows, device=loop_logits.device, dtype=torch.long),
        torch.tensor(source_loops, device=loop_logits.device, dtype=torch.long),
    )


def score_control_predictions(
    rows: list[dict[str, Any]],
    *,
    max_loops: int,
) -> dict[str, Any]:
    """Score transition recalls and first-stop exact-depth selection."""

    counts = {
        "continue_correct": 0,
        "continue_total": 0,
        "stop_correct": 0,
        "stop_total": 0,
    }
    exact = 0
    exhausted = 0
    by_depth: dict[str, dict[str, int | float]] = {}
    for row in rows:
        depth = int(row["depth"])
        predictions = [int(value) for value in row["predictions"]][: int(max_loops)]
        if len(predictions) < depth:
            raise ValueError(f"row {row.get('row_id')} lacks predictions through depth {depth}")
        for loop_number in range(1, depth + 1):
            target = STOP_CLASS if loop_number == depth else CONTINUE_CLASS
            prediction = predictions[loop_number - 1]
            key = "stop" if target == STOP_CLASS else "continue"
            counts[f"{key}_total"] += 1
            counts[f"{key}_correct"] += int(prediction == target)
        selected = next(
            (index + 1 for index, value in enumerate(predictions) if value == STOP_CLASS),
            None,
        )
        if selected is None:
            exhausted += 1
        hit = selected == depth
        exact += int(hit)
        cell = by_depth.setdefault(str(depth), {"correct": 0, "total": 0})
        cell["correct"] = int(cell["correct"]) + int(hit)
        cell["total"] = int(cell["total"]) + 1
    for cell in by_depth.values():
        cell["accuracy"] = int(cell["correct"]) / max(1, int(cell["total"]))
    return {
        **counts,
        "continue_recall": counts["continue_correct"] / max(1, counts["continue_total"]),
        "stop_recall": counts["stop_correct"] / max(1, counts["stop_total"]),
        "exact_selected_depth_correct": exact,
        "exact_selected_depth_total": len(rows),
        "exact_selected_depth_accuracy": exact / max(1, len(rows)),
        "exhausted_without_stop": exhausted,
        "by_depth": by_depth,
    }


def select_pilot_cell(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the P0 selection rule without permitting an implicit extension."""

    def cell_field(row: dict[str, Any], name: str) -> Any:
        """Read both persisted nested receipts and legacy flat test fixtures."""

        cell = row.get("cell")
        if isinstance(cell, dict) and name in cell:
            return cell[name]
        return row[name]

    references = [
        row for row in results if float(cell_field(row, "control_loss_lambda")) == 0.0
    ]
    if len(references) != 1:
        raise ValueError("P0 selection requires exactly one lambda-zero reference")
    reference_accuracy = float(references[0]["step_1500"]["answer_accuracy"])
    qualifying: list[dict[str, Any]] = []
    for row in results:
        if row is references[0]:
            continue
        metrics = row["step_1500"]
        if float(metrics["stop_recall"]) < 0.60 or float(metrics["continue_recall"]) < 0.60:
            continue
        candidate = dict(row)
        candidate["answer_accuracy_drop"] = reference_accuracy - float(
            metrics["answer_accuracy"]
        )
        qualifying.append(candidate)
    if not qualifying:
        return {
            "status": "no_qualifying_cell_reassess_before_lock",
            "selected_cell_id": None,
            "reference_answer_accuracy": reference_accuracy,
            "qualifying_cells": [],
        }

    qualifying.sort(
        key=lambda row: (
            float(row["answer_accuracy_drop"]),
            abs(float(cell_field(row, "control_loss_lambda")) - 1.0),
            abs(float(cell_field(row, "stop_to_continue_ratio")) - 3.5),
            str(cell_field(row, "cell_id")),
        )
    )
    selected = qualifying[0]
    return {
        "status": "selected",
        "selected_cell_id": str(cell_field(selected, "cell_id")),
        "control_loss_lambda": float(cell_field(selected, "control_loss_lambda")),
        "stop_to_continue_ratio": float(cell_field(selected, "stop_to_continue_ratio")),
        "answer_accuracy_drop": float(selected["answer_accuracy_drop"]),
        "reference_answer_accuracy": reference_accuracy,
        "qualifying_cells": [str(cell_field(row, "cell_id")) for row in qualifying],
    }
