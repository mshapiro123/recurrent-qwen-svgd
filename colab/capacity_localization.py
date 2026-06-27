"""Utilities for Stage 5 recurrent-operator capacity localization.

The capacity arm keeps the architecture and curriculum fixed while changing
only recurrent LoRA rank. These helpers make the resulting summaries comparable
without requiring a GPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


QWEN25_05B_STORED_PARAMS = 494_000_000
QWEN25_05B_RECURRENT_BLOCK_PARAMS = 179_000_000

# Qwen2.5-0.5B recurrent split 6:18 has 12 recurrent layers. Each layer wraps
# q/k/v/o plus gate/up/down projections. Sum(in + out) per layer is 24,448.
QWEN25_05B_RECURRENT_LORA_PARAMS_PER_RANK = 293_376


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def parse_int_csv(value: str | None, *, default: list[int]) -> list[int]:
    if not value:
        return list(default)
    ranks: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        rank = int(item)
        if rank <= 0:
            raise ValueError(f"capacity rank must be positive, got {rank}")
        ranks.append(rank)
    if not ranks:
        raise ValueError("capacity rank list is empty")
    return ranks


def lora_trainable_params_for_rank(rank: int) -> int:
    if rank <= 0:
        raise ValueError("rank must be positive")
    return int(rank) * QWEN25_05B_RECURRENT_LORA_PARAMS_PER_RANK


def trainable_parameter_ledger(rank: int, *, alpha: int | None = None) -> dict[str, Any]:
    """Return the accounting ledger used to interpret the capacity arm."""

    lora_params = lora_trainable_params_for_rank(rank)
    return {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "stored_model_params_estimate": QWEN25_05B_STORED_PARAMS,
        "recurrent_block_params_estimate": QWEN25_05B_RECURRENT_BLOCK_PARAMS,
        "lora_rank": int(rank),
        "lora_alpha": int(alpha if alpha is not None else 2 * rank),
        "lora_trainable_params_estimate": lora_params,
        "lora_trainable_params_millions": round(lora_params / 1_000_000, 3),
        "recurrent_block_unfreeze_trainable_params_estimate": QWEN25_05B_RECURRENT_BLOCK_PARAMS,
        "stored_size_changes_with_rank": False,
        "per_loop_compute_changes_with_rank": False,
        "capacity_cost_type": "training_params_not_stored_params",
    }


def _repo_path(root: str | Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else Path(root) / path


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - requirements include PyYAML.
        raise RuntimeError("PyYAML is required to read capacity run configs") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a YAML object")
    return payload


def lora_config_for_recovery_summary(
    root: str | Path,
    summary_path: str | Path,
    payload: dict[str, Any],
    *,
    fallback_rank: int | None = None,
    fallback_alpha: int | None = None,
) -> dict[str, int]:
    """Read LoRA rank/alpha from the child phase1 YAML when possible."""

    child_summary_value = str(payload.get("child_summary") or "")
    candidates: list[Path] = []
    if child_summary_value:
        child_summary = _repo_path(root, child_summary_value)
        candidates.append(child_summary.parent / "phase1_curriculum_sft.yaml")
    candidates.append(Path(summary_path).parent.parent / (Path(summary_path).parent.name + "_curriculum_sft") / "phase1_curriculum_sft.yaml")

    for candidate in candidates:
        if not candidate.exists():
            continue
        config = _load_yaml(candidate)
        lora = config.get("lora") if isinstance(config.get("lora"), dict) else {}
        rank = lora.get("rank")
        alpha = lora.get("alpha")
        if rank is not None:
            return {
                "rank": int(rank),
                "alpha": int(alpha if alpha is not None else 2 * int(rank)),
            }

    if fallback_rank is None:
        raise KeyError(f"Could not infer LoRA rank for {summary_path}")
    return {
        "rank": int(fallback_rank),
        "alpha": int(fallback_alpha if fallback_alpha is not None else 2 * fallback_rank),
    }


def strength_summary(payload: dict[str, Any], *, strength: float = 1.0) -> dict[str, Any]:
    readout = payload.get("fixed_tail_damper_depth_readout")
    if not isinstance(readout, dict):
        return {}
    summaries = readout.get("strength_summaries")
    if not isinstance(summaries, list):
        return {}
    for item in summaries:
        if not isinstance(item, dict):
            continue
        try:
            item_strength = float(item.get("strength"))
        except (TypeError, ValueError):
            continue
        if math.isclose(item_strength, strength, rel_tol=1e-6, abs_tol=1e-6):
            return item
    return {}


def extract_capacity_row(
    root: str | Path,
    summary_path: str | Path,
    *,
    rank: int | None = None,
    alpha: int | None = None,
    strength: float = 1.0,
) -> dict[str, Any]:
    summary_path = Path(summary_path)
    payload = read_json(summary_path)
    lora = lora_config_for_recovery_summary(
        root,
        summary_path,
        payload,
        fallback_rank=rank,
        fallback_alpha=alpha,
    )
    strength_payload = strength_summary(payload, strength=strength)
    score = strength_payload.get("score_summary") if isinstance(strength_payload.get("score_summary"), dict) else {}
    loop_results = score.get("loop_results") if isinstance(score.get("loop_results"), dict) else {}
    tail_trace = strength_payload.get("tail_trace") if isinstance(strength_payload.get("tail_trace"), dict) else {}

    def loop_correct(loop: int) -> int | None:
        item = loop_results.get(str(loop))
        return int(item["correct"]) if isinstance(item, dict) and item.get("correct") is not None else None

    def loop_accuracy(loop: int) -> float | None:
        item = loop_results.get(str(loop))
        return float(item["accuracy"]) if isinstance(item, dict) and item.get("accuracy") is not None else None

    def tail_ratio(loop: int) -> float | None:
        item = tail_trace.get(f"loop{loop}")
        return float(item["ratio_vs_entry"]) if isinstance(item, dict) and item.get("ratio_vs_entry") is not None else None

    loop_corrects = {str(loop): loop_correct(loop) for loop in (1, 2, 3)}
    loop_accuracies = {str(loop): loop_accuracy(loop) for loop in (1, 2, 3)}
    loop1 = loop_corrects["1"]
    loops_to_benefit = None
    if loop1 is not None:
        for loop in (2, 3):
            value = loop_corrects[str(loop)]
            if value is not None and value > loop1:
                loops_to_benefit = loop
                break

    return {
        "summary_path": summary_path.as_posix(),
        "run_id": payload.get("run_id") or summary_path.parent.name,
        "status": payload.get("status"),
        "passed": bool(payload.get("passed")),
        "checkpoint": payload.get("checkpoint") or payload.get("phase1_checkpoint"),
        "child_summary": payload.get("child_summary"),
        "lora": lora,
        "parameter_ledger": trainable_parameter_ledger(lora["rank"], alpha=lora["alpha"]),
        "fixed_tail_damper_strength": strength,
        "post_reentry_health_status": (
            payload.get("post_reentry_health_checks", {}).get("status")
            if isinstance(payload.get("post_reentry_health_checks"), dict)
            else None
        ),
        "loop_correct": loop_corrects,
        "loop_accuracy": loop_accuracies,
        "oracle_correct": score.get("oracle_correct"),
        "oracle_accuracy": score.get("oracle_accuracy"),
        "oracle_gap_vs_loop1": score.get("oracle_gap_vs_loop1"),
        "rescued_vs_loop1": score.get("rescued_vs_loop1"),
        "harmed_vs_loop1": score.get("harmed_vs_loop1"),
        "stable_correct": score.get("stable_correct"),
        "stable_wrong": score.get("stable_wrong"),
        "pattern_counts": score.get("pattern_counts", {}),
        "tail_ratio_vs_entry": {
            "loop1": tail_ratio(1),
            "loop2": tail_ratio(2),
            "loop3": tail_ratio(3),
            "loop4": tail_ratio(4),
            "loop8": tail_ratio(8),
        },
        "loops_to_benefit_vs_loop1": loops_to_benefit,
        "rescued_harmed_ratio": (
            float(score["rescued_vs_loop1"]) / float(score["harmed_vs_loop1"])
            if score.get("harmed_vs_loop1")
            else None
        ),
    }


def add_baseline_deltas(rows: list[dict[str, Any]], *, baseline_rank: int = 32) -> list[dict[str, Any]]:
    baseline = next((row for row in rows if int(row["lora"]["rank"]) == baseline_rank), None)
    if baseline is None:
        return rows

    metrics = [
        "oracle_correct",
        "oracle_gap_vs_loop1",
        "rescued_vs_loop1",
        "harmed_vs_loop1",
        "stable_correct",
        "stable_wrong",
    ]
    loop_metrics = ["1", "2", "3"]
    for row in rows:
        deltas: dict[str, Any] = {}
        for metric in metrics:
            left = row.get(metric)
            right = baseline.get(metric)
            if left is not None and right is not None:
                deltas[metric] = left - right
        for loop in loop_metrics:
            left = row["loop_correct"].get(loop)
            right = baseline["loop_correct"].get(loop)
            if left is not None and right is not None:
                deltas[f"loop{loop}_correct"] = left - right
        row["delta_vs_rank32"] = deltas
    return rows


def capacity_decision(rows: list[dict[str, Any]], *, baseline_rank: int = 32) -> dict[str, Any]:
    candidates = [row for row in rows if int(row["lora"]["rank"]) != baseline_rank]
    if not candidates:
        return {
            "status": "needs_capacity_run",
            "recommendation": "Run at least one higher-rank capacity arm.",
        }
    best_oracle = max(candidates, key=lambda row: int(row.get("oracle_correct") or -1))
    best_loop1 = max(candidates, key=lambda row: int(row["loop_correct"].get("1") or -1))
    any_depth_pays = any(row.get("loops_to_benefit_vs_loop1") for row in candidates)
    any_oracle_improves = any((row.get("delta_vs_rank32") or {}).get("oracle_correct", 0) > 0 for row in candidates)
    any_rescue_improves = any((row.get("delta_vs_rank32") or {}).get("rescued_vs_loop1", 0) > 0 for row in candidates)
    any_harm_worsens = any((row.get("delta_vs_rank32") or {}).get("harmed_vs_loop1", 0) > 0 for row in candidates)
    any_loop2_or_loop3_regresses = any(
        (row.get("delta_vs_rank32") or {}).get("loop2_correct", 0) < 0
        or (row.get("delta_vs_rank32") or {}).get("loop3_correct", 0) < 0
        for row in candidates
    )
    useful_signal = any_depth_pays or (
        any_oracle_improves
        and any_rescue_improves
        and not any_harm_worsens
        and not any_loop2_or_loop3_regresses
    )
    weak_or_mixed_signal = (any_oracle_improves or any_rescue_improves) and not useful_signal
    return {
        "status": (
            "capacity_signal_present"
            if useful_signal
            else "capacity_signal_mixed_or_negative"
            if weak_or_mixed_signal
            else "capacity_signal_not_yet_seen"
        ),
        "best_oracle_rank": int(best_oracle["lora"]["rank"]),
        "best_oracle_correct": best_oracle.get("oracle_correct"),
        "best_loop1_rank": int(best_loop1["lora"]["rank"]),
        "best_loop1_correct": best_loop1["loop_correct"].get("1"),
        "any_depth_loop_beats_loop1": any_depth_pays,
        "any_oracle_count_improves_vs_rank32": any_oracle_improves,
        "any_rescue_count_improves_vs_rank32": any_rescue_improves,
        "any_harm_count_worsens_vs_rank32": any_harm_worsens,
        "any_loop2_or_loop3_regresses_vs_rank32": any_loop2_or_loop3_regresses,
        "recommendation": (
            "Capacity rank produced cleaner deeper-loop benefit; replicate or expand before unfreezing."
            if useful_signal
            else "Higher LoRA rank produced only mixed/noisy oracle or rescue movement while harm/deeper-loop regression remained; stop rank-only escalation and review the unfreeze+Muon bundle."
            if weak_or_mixed_signal
            else "Rank-only capacity is flat; the next meaningful test is the unfreeze+Muon bundle."
        ),
    }


def write_capacity_localization_summary(
    *,
    root: str | Path,
    run_id: str,
    output_dir: str | Path,
    baseline_summaries: list[str | Path],
    result_summaries: list[str | Path],
    target_ranks: list[int],
    baseline_rank: int = 32,
) -> Path:
    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [_repo_path(root, path) for path in [*baseline_summaries, *result_summaries]]
    rows = [extract_capacity_row(root, path) for path in paths]
    rows = add_baseline_deltas(rows, baseline_rank=baseline_rank)
    summary = {
        "kind": "stage5_capacity_localization_sweep",
        "run_id": run_id,
        "status": "completed",
        "target_ranks": target_ranks,
        "baseline_rank": baseline_rank,
        "baseline_summaries": [str(path).replace("\\", "/") for path in baseline_summaries],
        "result_summaries": [str(path).replace("\\", "/") for path in result_summaries],
        "fixed_tail_damper_strength": 1.0,
        "rows": rows,
        "decision": capacity_decision(rows, baseline_rank=baseline_rank),
        "next_step": next_capacity_step(target_ranks),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Stage 5 Capacity Localization - {run_id}",
        "",
        f"- Baseline rank: `{baseline_rank}`",
        f"- Target ranks: `{', '.join(map(str, target_ranks))}`",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']['status']}`",
        "",
        "## Capacity Ledger",
        "",
        "| rank | alpha | LoRA trainable M | stored params M | loop1 | loop2 | loop3 | oracle | rescued | harmed | loop8 tail ratio |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        ledger = row["parameter_ledger"]
        lines.append(
            "| {rank} | {alpha} | {train_m:.3f} | {stored_m:.1f} | {l1} | {l2} | {l3} | {oracle} | {rescued} | {harmed} | {tail8} |".format(
                rank=row["lora"]["rank"],
                alpha=row["lora"]["alpha"],
                train_m=ledger["lora_trainable_params_millions"],
                stored_m=ledger["stored_model_params_estimate"] / 1_000_000,
                l1=row["loop_correct"].get("1"),
                l2=row["loop_correct"].get("2"),
                l3=row["loop_correct"].get("3"),
                oracle=row.get("oracle_correct"),
                rescued=row.get("rescued_vs_loop1"),
                harmed=row.get("harmed_vs_loop1"),
                tail8=row["tail_ratio_vs_entry"].get("loop8"),
            )
        )
    lines.extend(
        [
            "",
            "## Deltas Vs Rank 32",
            "",
            "```json",
            json.dumps(
                {
                    row["lora"]["rank"]: row.get("delta_vs_rank32", {})
                    for row in rows
                    if row.get("delta_vs_rank32") is not None
                },
                indent=2,
            ),
            "```",
            "",
            "## Decision",
            "```json",
            json.dumps(summary["decision"], indent=2),
            "```",
            "",
            summary["next_step"],
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def next_capacity_step(target_ranks: list[int]) -> str:
    max_rank = max(target_ranks) if target_ranks else 0
    if max_rank < 128:
        return (
            "Review rank-localization deltas before escalating. Rank128 is justified only if rank64 "
            "shows rescued/oracle/depth movement; otherwise the next meaningful test is the unfreeze+Muon bundle."
        )
    return (
        "Rank128 completes the clean rank-only capacity sweep. If depth still does not beat loop1 and "
        "rescued gains come with added harm, stop rank-only escalation. The next experimental fork is "
        "either close this line or run the unfreeze+Muon recurrence-curriculum bundle as a deliberately "
        "larger-capacity substrate test."
    )
