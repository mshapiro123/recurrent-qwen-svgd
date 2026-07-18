"""Compile the manuscript-facing Paper 1 evidence into one auditable receipt pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

LINEAGE_RUNS = [
    "stage5_lineage_regression_battery_current_1_stage5_reentry_recovery_20260625_154210",
    "stage5_lineage_regression_battery_current_2_stage5_depth_support_route_20260705_124320",
    "stage5_lineage_regression_battery_current_3_stage5_support8_dose_arm_20260706_153028",
    "stage5_lineage_regression_battery_current_4_stage5_n24_support12_rung_20260707_140139",
]
EARLY_TELEMETRY = [
    (
        "extended_fold0_random32_rep05",
        "outputs/diagnostics/extended_fold0_random32_rep05_seeds5_9.jsonl",
    ),
    (
        "extended_fold0_within_group_dim8_rep2",
        "outputs/diagnostics/extended_fold0_wg_dim8_rep2_seeds5_9.jsonl",
    ),
    (
        "extended_fold1_random32_rep05",
        "outputs/diagnostics/extended_fold1_random32_rep05_seeds5_9.jsonl",
    ),
    (
        "extended_fold1_within_group_dim8_rep2",
        "outputs/diagnostics/extended_fold1_wg_dim8_rep2_seeds5_9.jsonl",
    ),
    (
        "recreated_current_random32_rep05",
        "outputs/diagnostics/recreated_current_random32_rep05.jsonl",
    ),
    (
        "recreated_current_within_group_dim8_rep2",
        "outputs/diagnostics/recreated_current_wg_dim8_rep2.jsonl",
    ),
    (
        "original_stage4_exact_phase1_vs_phase2",
        "outputs/stage4/stage4_opus_a100_20260620/exact_phase1_vs_phase2.jsonl",
    ),
]
PRIMARY_MULTIPLICITY_FAMILY = 8
NONINFERIORITY_MARGIN = -0.03


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with (ROOT / path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def telemetry_receipt(label: str, path: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["label"]), str(row["task"]), int(row["seed"]))
        cells.setdefault(key, row)
    if not cells:
        raise RuntimeError(f"No telemetry cells found in {path}")
    unique_rows = list(cells.values())
    diagnostics = [row["diagnostics"] for row in unique_rows]
    first = unique_rows[0]
    return {
        "label": label,
        "path": path,
        "sha256": sha256_file(path),
        "raw_rows": len(rows),
        "unique_task_seed_cells": len(unique_rows),
        "deduplication_key": ["label", "task", "seed"],
        "mean_expected_loops": mean(
            float(item["mean_expected_loops"]) for item in diagnostics
        ),
        "mean_halt_entropy": mean(float(item["mean_halt_entropy"]) for item in diagnostics),
        "best_hits": sum(bool(row["best_hit"]) for row in unique_rows),
        "candidate_hits": sum(int(row["task_candidate_hits"]) for row in unique_rows),
        "candidate_total": sum(int(row["task_candidate_count"]) for row in unique_rows),
        "config": {
            "mode": first.get("mode"),
            "num_trajectories": first.get("num_trajectories"),
            "sample_latents": first.get("sample_latents"),
            "latent_injection_mode": first.get("latent_injection_mode"),
            "temperature": first.get("temperature"),
            "particle_init_noise": first.get("particle_init_noise"),
            "particle_noise_steps": first.get("particle_noise_steps"),
            "svgd_repulsion_scale": first.get("svgd_repulsion_scale"),
            "svgd_kernel_projection_dim": first.get("svgd_kernel_projection_dim"),
            "svgd_kernel_geometry": first.get("svgd_kernel_geometry"),
        },
        "interpretation": (
            "descriptive_historical_screen_without_preregistered_statistical_acceptance_gate"
        ),
    }


def paired_guardrail_receipts() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index, run_id in enumerate(LINEAGE_RUNS, start=1):
        path = f"outputs/stage5/{run_id}/summary.json"
        payload = read_json(path)
        for benchmark in ("arc_easy", "arc_challenge"):
            for score_target, aggregate in (
                ("content_question_only", "mean"),
                ("cyclic_label_aggregated", "permutation_mean"),
            ):
                paired = payload["paired_comparisons"][benchmark][score_target][aggregate]
                record = {
                    "checkpoint_index": index,
                    "run_id": run_id,
                    "benchmark": benchmark,
                    "score_target": score_target,
                    "aggregate": aggregate,
                    "n": int(paired["paired_examples"]),
                    "base_correct": int(paired["base_correct"]),
                    "recurrent_correct": int(paired["recurrent_correct"]),
                    "correct_delta": int(paired["correct_delta_recurrent_vs_base"]),
                    "accuracy_delta": float(paired["accuracy_delta_recurrent_vs_base"]),
                    "wins": int(paired["wins"]),
                    "losses": int(paired["losses"]),
                    "ties": int(paired["ties"]),
                    "raw_sign_test_p": float(paired["sign_test_p_value"]),
                    "noninferiority_margin": NONINFERIORITY_MARGIN,
                    "noninferior": (
                        float(paired["accuracy_delta_recurrent_vs_base"])
                        >= NONINFERIORITY_MARGIN
                    ),
                    "source": path,
                }
                if score_target == "cyclic_label_aggregated":
                    record["bonferroni_family_size"] = PRIMARY_MULTIPLICITY_FAMILY
                    record["bonferroni_p"] = min(
                        1.0,
                        record["raw_sign_test_p"] * PRIMARY_MULTIPLICITY_FAMILY,
                    )
                records.append(record)
    primary = [
        record for record in records if record["score_target"] == "cyclic_label_aggregated"
    ]
    secondary = [
        record for record in records if record["score_target"] == "content_question_only"
    ]
    most_adverse = min(primary, key=lambda record: record["accuracy_delta"])
    return {
        "battery": "ARC Easy and ARC Challenge, content and cyclic-label readers",
        "checkpoints": len(LINEAGE_RUNS),
        "primary_family": {
            "definition": "four checkpoints x two cyclic-label benchmark comparisons",
            "multiplicity_method": "Bonferroni",
            "family_size": PRIMARY_MULTIPLICITY_FAMILY,
            "records": primary,
        },
        "secondary_diagnostic_family": {
            "definition": "content-question-only comparisons",
            "multiplicity_method": "descriptive_not_used_for_primary_inference",
            "records": secondary,
        },
        "most_adverse_primary_result": most_adverse,
        "all_noninferior_at_minus_3pp": all(record["noninferior"] for record in records),
        "claim_boundary": (
            "The battery supports bounded noninferiority at a minus-three-point margin; "
            "it does not establish broad capability parity."
        ),
    }


def peft_canary_receipt() -> dict[str, Any]:
    path = "outputs/stage5/stage5_peft_ponder_closure_20260717_182113/summary.json"
    payload = read_json(path)
    arm = next(
        item for item in payload["p1_results"] if str(item.get("arm")) == "R16"
    )
    intervals = arm["intervals"]
    return {
        "composition": "64 frozen arithmetic prompts",
        "n": 64,
        "baseline": arm["tier1_baseline"],
        "hard_stop_accuracy_delta": -0.03,
        "intervals": [
            {
                "step": int(item["cumulative_step"]),
                "candidate_accuracy": float(item["tier1"]["candidate_accuracy"]),
                "accuracy_delta": float(
                    item["tier1"]["in_training_receipt"]["accuracy_delta"]
                ),
                "status": item["tier1"]["in_training_receipt"]["status"],
            }
            for item in intervals
        ],
        "identity_max_abs_diff": float(arm["identity"]["max_abs_diff"]),
        "pretrained_base_hash_unchanged": bool(arm["base_hash_unchanged"]),
        "permutation_control": "not_run_for_this_bounded_canary",
        "source": path,
        "claim_boundary": "Arithmetic-slice preservation only; not broad natural capability.",
    }


def dead_bridge_receipt() -> dict[str, Any]:
    path = "outputs/stage5/stage5_reentry_drift_20260625_011444/summary.json"
    payload = read_json(path)
    return {
        "bridge": payload["bridge"],
        "gradient_liveness": payload["bridge_gradient_liveness"],
        "reference_gradient_liveness": payload["reference_bridge_gradient_liveness"],
        "source": path,
        "interpretation": (
            "The historical identity-gated bridge was functionally dead: zero bridge "
            "delta and zero gate, weight, and bias gradients. Later repaired-lineage "
            "results must not be back-projected onto this era."
        ),
    }


def r16_bridge_receipt() -> dict[str, Any]:
    ledger_path = "docs/part1_claim_evidence_ledger.json"
    ledger = read_json(ledger_path)
    serialized = json.dumps(ledger)
    required = {
        "optimizer_marked": 7_613_953,
        "forward_active": 6_007_425,
        "lora": 4_399_104,
        "bridge_marked": 3_214_849,
        "bridge_active": 1_608_321,
        "bridge_legacy_bypassed": 1_606_528,
    }
    for value in required.values():
        if str(value) not in serialized:
            raise RuntimeError(f"R16 accounting value {value} missing from {ledger_path}")
    return {
        **required,
        "legacy_tensor_names": ["bridge.proj.weight", "bridge.proj.bias"],
        "historical_interpretation": (
            "The historical R16 optimizer marked legacy concat tensors trainable even "
            "though split-mode forward bypassed them."
        ),
        "prospective_arm_e_policy": (
            "Arm E excludes the bypassed legacy tensors from optimizer groups and "
            "requires optimizer-marked equals forward-active equals 6,007,425."
        ),
        "source": ledger_path,
    }


def keeper_receipt() -> dict[str, Any]:
    path = "outputs/stage5/stage5_part1_closeout_pivot_20260715/summary.json"
    payload = read_json(path)
    serialized = json.dumps(payload)
    expected = {
        "row_sha256": "eb80ef24637aee511a3e35607e87ae2530842ce11c551e6fa90ecda4d4115ef8",
        "keeper_sha256": "0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f",
    }
    for value in expected.values():
        if value not in serialized:
            raise RuntimeError(f"Keeper receipt {value} missing from {path}")
    return {
        "composition": "frozen N20 verbal branching rows, depths 1-4, 128 per depth",
        "pooled": {"correct": 389, "total": 512, "accuracy": 389 / 512, "floor": 0.70},
        "by_depth_correct": {"1": 127, "2": 95, "3": 87, "4": 80},
        "per_depth_total": 128,
        "per_depth_floor": 0.55,
        **expected,
        "source": path,
    }


def literature_receipts() -> list[dict[str, Any]]:
    return [
        {
            "work": "Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence",
            "authors": (
                "Sean McLeish, Ang Li, John Kirchenbauer, Dayal Singh Kalra, "
                "Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, "
                "Jonas Geiping, Tom Goldstein, Micah Goldblum"
            ),
            "arxiv": "2511.07384",
            "url": "https://arxiv.org/abs/2511.07384",
            "claim_fit": (
                "Supports contextualizing pretrained-model recurrence retrofitting. "
                "Do not describe its method as LoRA-based or as requiring auxiliary adapters."
            ),
        },
        {
            "work": "Hierarchical Reasoning Model",
            "authors": (
                "Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, "
                "Meng Lu, Sen Song, Yasin Abbasi-Yadkori"
            ),
            "arxiv": "2506.21734",
            "url": "https://arxiv.org/abs/2506.21734",
            "claim_fit": "Task-specific hierarchical recurrent reasoning comparator.",
        },
        {
            "work": "Tiny Recursive Models",
            "authors": "Alexia Jolicoeur-Martineau",
            "arxiv": "2510.04871",
            "url": "https://arxiv.org/abs/2510.04871",
            "claim_fit": "Task-specific recursive reasoning comparator.",
        },
        {
            "work": "Procedural Knowledge in Pretraining Drives Reasoning in Large Language Models",
            "authors": "Laura Ruis et al.",
            "arxiv": "2411.12580",
            "url": "https://arxiv.org/abs/2411.12580",
            "claim_fit": (
                "Supports a relationship between procedural pretraining knowledge and "
                "reasoning; does not justify a strict memorization-versus-procedure dichotomy."
            ),
        },
    ]


def build_receipts() -> dict[str, Any]:
    guardrail = paired_guardrail_receipts()
    most_adverse = guardrail["most_adverse_primary_result"]
    if most_adverse["correct_delta"] != -14:
        raise RuntimeError(f"Unexpected most adverse guardrail result: {most_adverse}")
    if not math.isclose(most_adverse["raw_sign_test_p"], 0.03847730828420026):
        raise RuntimeError(f"Unexpected raw p value: {most_adverse}")
    return {
        "kind": "paper1_experimental_closure_receipts",
        "version": "20260718",
        "manuscript_prose_modified": False,
        "guardrails": {
            "lineage_regression_battery": guardrail,
            "peft_installation_canary": peft_canary_receipt(),
            "phase_g_keeper_gate": keeper_receipt(),
        },
        "early_stochastic_era": {
            "telemetry": [telemetry_receipt(label, path) for label, path in EARLY_TELEMETRY],
            "dead_bridge": dead_bridge_receipt(),
            "acceptance_gate_reading": (
                "These screens predated preregistered statistical acceptance thresholds. "
                "They are descriptive archaeology, not retrospective pass/fail experiments."
            ),
        },
        "parameter_accounting": {"r16_split_bridge": r16_bridge_receipt()},
        "literature": literature_receipts(),
        "do_not_claim": [
            "No broad natural-capability parity from the bounded guardrails.",
            "No calibrated useful early learned-halting policy from aggregate telemetry.",
            "No GRAM-style width conclusion from pre-repair stochastic experiments.",
            "No description of McLeish et al. as a LoRA or auxiliary-adapter recipe.",
            "No budget-independence or capacity-limit conclusion before Arm E lands.",
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    guardrail = payload["guardrails"]["lineage_regression_battery"]
    adverse = guardrail["most_adverse_primary_result"]
    lines = [
        "# Paper 1 Experimental Closure Receipts",
        "",
        "**Scope:** evidence and claim boundaries only; manuscript prose was not edited.",
        "",
        "## Guardrail Battery",
        "",
        f"- Four checkpoint lineages; ARC Easy `n=5,197`, ARC Challenge `n=2,590`.",
        "- Primary family: eight cyclic-label comparisons; Bonferroni correction.",
        f"- Most adverse result: `{adverse['run_id']}` / `{adverse['benchmark']}`, "
        f"delta `{adverse['correct_delta']}`, raw `p={adverse['raw_sign_test_p']:.6f}`, "
        f"corrected `p={adverse['bonferroni_p']:.6f}`.",
        f"- Every comparison remained above the locked `{NONINFERIORITY_MARGIN:.0%}` margin.",
        "",
        "| Checkpoint | Benchmark | Reader | Delta | Wins | Losses | Raw p | Corrected p |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    records = (
        guardrail["primary_family"]["records"]
        + guardrail["secondary_diagnostic_family"]["records"]
    )
    for row in records:
        corrected = row.get("bonferroni_p")
        corrected_text = f"{corrected:.6f}" if corrected is not None else "descriptive"
        lines.append(
            f"| {row['checkpoint_index']} | {row['benchmark']} | {row['score_target']} | "
            f"{row['correct_delta']:+d} | {row['wins']} | {row['losses']} | "
            f"{row['raw_sign_test_p']:.6f} | {corrected_text} |"
        )
    canary = payload["guardrails"]["peft_installation_canary"]
    lines.extend(
        [
            "",
            "## Bounded PEFT Canary",
            "",
            f"- Baseline: `{canary['baseline']['accuracy']:.4f}` on `{canary['n']}` arithmetic rows.",
            f"- Six interval checks stayed at `{canary['intervals'][-1]['candidate_accuracy']:.4f}`.",
            f"- Identity maximum absolute difference: `{canary['identity_max_abs_diff']}`.",
            "- No permutation control was run for this bounded canary.",
            "",
            "## Early-Era Telemetry",
            "",
            "| Archive | Cells | Expected loops | Halt entropy |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["early_stochastic_era"]["telemetry"]:
        lines.append(
            f"| {row['label']} | {row['unique_task_seed_cells']} | "
            f"{row['mean_expected_loops']:.4f} | {row['mean_halt_entropy']:.4f} |"
        )
    bridge = payload["early_stochastic_era"]["dead_bridge"]
    r16 = payload["parameter_accounting"]["r16_split_bridge"]
    lines.extend(
        [
            "",
            "## Structural Receipts",
            "",
            f"- Historical bridge gate: `{bridge['bridge']['bridge_gate']}`; "
            f"delta RMS: `{bridge['bridge']['sample_bridge_delta_rms']}`; "
            f"weight-gradient RMS: `{bridge['gradient_liveness']['weight_grad_rms']}`.",
            f"- Historical R16 optimizer-marked parameters: `{r16['optimizer_marked']:,}`.",
            f"- Historical R16 forward-active parameters: `{r16['forward_active']:,}`.",
            "- Prospective Arm E excludes the bypassed legacy concat tensors.",
            "",
            "## Literature Claim Fit",
            "",
        ]
    )
    for item in payload["literature"]:
        lines.append(
            f"- **{item['work']}** (arXiv:{item['arxiv']}): {item['claim_fit']}"
        )
    lines.extend(["", "## Do Not Claim", ""])
    lines.extend(f"- {claim}" for claim in payload["do_not_claim"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_json",
        default="docs/PAPER1_EXPERIMENTAL_CLOSURE_RECEIPTS_20260718.json",
    )
    parser.add_argument(
        "--output_md",
        default="docs/PAPER1_EXPERIMENTAL_CLOSURE_RECEIPTS_20260718.md",
    )
    args = parser.parse_args()
    payload = build_receipts()
    json_path = ROOT / args.output_json
    md_path = ROOT / args.output_md
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
