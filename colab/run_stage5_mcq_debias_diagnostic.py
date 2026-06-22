"""Run MCQ option-ID debias diagnostics before more recurrent training.

This runner asks whether the current ARC-Easy regression is real content
degradation or an amplified A/B/C/D selection-bias artifact. It re-scores the
same bounded ARC slice three ways:

1. bare option-label scoring, the historical metric;
2. option-content scoring with labels removed from the prompt;
3. cyclic option permutations, aggregated back to original option content.

It intentionally does not train.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_recovered_phase1_arc_gate import restore_checkpoint_if_needed  # noqa: E402
from eval.analyze_mcq_regressions import paired_rows, rows_by_id, summarize  # noqa: E402
from eval.mcq_debias import (  # noqa: E402
    aggregate_permutation_scores,
    cyclic_permutation_rows,
    read_jsonl,
    summarize_rows,
    write_jsonl,
)


RUN_ID = os.environ.get("STAGE5_MCQ_DEBIAS_RUN_ID") or time.strftime(
    "stage5_mcq_debias_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_SUMMARY = os.environ.get(
    "STAGE5_MCQ_DEBIAS_SOURCE_SUMMARY",
    "outputs/stage5/stage5_arc_agi_next_action_20260622_181850_plan_conservative_direct_preservation/answer_prior_diagnosis.json",
)
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
ARC_CONFIG = os.environ.get("STAGE5_MCQ_DEBIAS_ARC_CONFIG", "ARC-Easy")
ARC_LIMIT = int(os.environ.get("STAGE5_MCQ_DEBIAS_ARC_LIMIT", "128"))
ARC_SEED = int(os.environ.get("STAGE5_MCQ_DEBIAS_ARC_SEED", "17"))
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE5_MCQ_DEBIAS_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
MAX_DEBIASED_GAP = int(os.environ.get("STAGE5_MCQ_DEBIAS_MAX_DEBIASED_GAP", "2"))
MIN_CLOSURE = int(os.environ.get("STAGE5_MCQ_DEBIAS_MIN_CLOSURE", "3"))
MIN_LABEL_GAP = int(os.environ.get("STAGE5_MCQ_DEBIAS_MIN_LABEL_GAP", "3"))
QUIET_EVAL = os.environ.get("STAGE5_MCQ_DEBIAS_QUIET_EVAL", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
RESUME_EXISTING = os.environ.get("STAGE5_MCQ_DEBIAS_RESUME_EXISTING", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

DATA_DIR = ROOT / "data" / RUN_ID
ARC_JSONL = DATA_DIR / f"{ARC_CONFIG.lower().replace('-', '_')}_{ARC_LIMIT}.jsonl"
PERMUTED_JSONL = DATA_DIR / f"{ARC_CONFIG.lower().replace('-', '_')}_{ARC_LIMIT}_cyclic_permuted.jsonl"


@dataclass(frozen=True)
class Arm:
    name: str
    mode: str
    checkpoint: Path | None = None
    max_loops: int = 4


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(cmd: list[str], *, check: bool = True, log_name: str | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
    if QUIET_EVAL and log_name:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout = proc.stdout or ""
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
        summary_lines = [
            line
            for line in stdout.splitlines()
            if line.startswith(("aggregate=", "dataset_id=", "config=", "split=", "rows=", "output_jsonl="))
            or "loaded_checkpoint=" in line
            or line.startswith("lora_recurrent_modules=")
        ]
        for line in summary_lines[-20:]:
            print(line, flush=True)
        if check and proc.returncode:
            print("FAILED_TAIL_START", flush=True)
            print("\n".join(stdout.splitlines()[-80:]), flush=True)
            print("FAILED_TAIL_END", flush=True)
            raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
        return proc

    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    stdout = "".join(chunks)
    if log_name:
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    proc = subprocess.CompletedProcess(cmd, process.wait(), stdout, None)
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def existing_complete_jsonl(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_jsonl(path)
    except Exception as exc:
        print(f"discarding invalid existing output: {path_for_cli(path)} ({exc})", flush=True)
        path.unlink(missing_ok=True)
        return False
    if len(rows) != expected_rows:
        print(
            f"discarding incomplete existing output: {path_for_cli(path)} rows={len(rows)} expected={expected_rows}",
            flush=True,
        )
        path.unlink(missing_ok=True)
        return False
    return True


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def update_current_source_summary(summary_path: Path) -> None:
    pointer = current_source_summary_file()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")


def source_payloads() -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    source_path = resolve_path(SOURCE_SUMMARY)
    source = read_json(source_path)
    if source.get("kind") == "stage5_arc_mix_answer_prior_diagnosis":
        nested = resolve_path(str(source.get("source_summary") or ""))
        return source_path, source, nested, read_json(nested)
    if source.get("kind") == "stage5_mcq_debias_diagnostic":
        nested_ref = str(source.get("nested_source_summary") or source.get("source_summary") or "").strip()
        if nested_ref:
            nested = resolve_path(nested_ref)
            return source_path, source, nested, read_json(nested)
    return source_path, source, source_path, source


def prepare_arc_data() -> None:
    if ARC_JSONL.exists() and PERMUTED_JSONL.exists():
        return
    run(
        [
            sys.executable,
            "eval/prepare_arc_mcq.py",
            "--config",
            ARC_CONFIG,
            "--split",
            "validation",
            "--seed",
            str(ARC_SEED),
            "--limit",
            str(ARC_LIMIT),
            "--output_jsonl",
            path_for_cli(ARC_JSONL),
        ],
        log_name="prepare_arc.log",
    )
    write_jsonl(PERMUTED_JSONL, cyclic_permutation_rows(read_jsonl(ARC_JSONL)))


def best_checkpoint_from_summary(summary: dict[str, Any]) -> Path | None:
    best = ((summary.get("best_arm") or {}).get("best_checkpoint") or {}).get("checkpoint")
    return resolve_path(best) if best else None


def arms_from_summary(summary: dict[str, Any]) -> list[Arm]:
    resume = resolve_path(str(summary.get("resume_checkpoint") or ""))
    best = best_checkpoint_from_summary(summary)
    arms = [
        Arm(name="base", mode="base"),
        Arm(name="start_loop1", mode="phase1", checkpoint=resume, max_loops=1),
        Arm(name="start_loop4", mode="phase1", checkpoint=resume, max_loops=4),
    ]
    if best and best != resume:
        arms.append(Arm(name="best_loop4", mode="phase1", checkpoint=best, max_loops=4))
    return arms


def restore_arm_checkpoints(summary: dict[str, Any], arms: list[Arm]) -> None:
    run_id = str(summary.get("resume_run_id") or summary.get("run_id") or "")
    for arm in arms:
        if arm.checkpoint is None:
            continue
        checkpoint_run_id = run_id
        checkpoint_text = path_for_cli(arm.checkpoint)
        if "outputs/stage5/" in checkpoint_text:
            checkpoint_run_id = checkpoint_text.split("outputs/stage5/", 1)[1].split("/", 1)[0]
        restore_checkpoint_if_needed(arm.checkpoint, run_id=checkpoint_run_id)


def eval_mcq(arm: Arm, *, data_jsonl: Path, prompt_style: str, score_target: str, suffix: str) -> Path:
    output = RUN_DIR / f"{arm.name}_{suffix}.jsonl"
    expected_rows = len(read_jsonl(data_jsonl))
    if RESUME_EXISTING and existing_complete_jsonl(output, expected_rows):
        print(f"reuse_existing={path_for_cli(output)} rows={expected_rows}", flush=True)
        return output
    output.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        "eval/eval_mcq.py",
        "--model_name",
        MODEL_NAME,
        "--data_jsonl",
        path_for_cli(data_jsonl),
        "--prompt_style",
        prompt_style,
        "--score_target",
        score_target,
        "--mode",
        arm.mode,
        "--aggregate",
        "mean",
        "--dtype",
        DTYPE,
        "--adapter_dtype",
        ADAPTER_DTYPE,
        "--device",
        DEVICE,
        "--seed",
        "0",
        "--output_jsonl",
        path_for_cli(output),
    ]
    if arm.mode != "base":
        assert arm.checkpoint is not None
        cmd.extend(
            [
                "--checkpoint",
                path_for_cli(arm.checkpoint),
                "--max_loops",
                str(arm.max_loops),
                "--num_trajectories",
                "1",
                "--include_loop_diagnostics",
            ]
        )
    run(cmd, log_name=f"{arm.name}_{suffix}.log")
    return output


def evaluate_arm(arm: Arm) -> dict[str, Any]:
    label_path = eval_mcq(arm, data_jsonl=ARC_JSONL, prompt_style="with_options", score_target="label", suffix="label")
    content_path = eval_mcq(
        arm,
        data_jsonl=ARC_JSONL,
        prompt_style="question_only",
        score_target="option_text",
        suffix="content_question_only",
    )
    perm_label_path = eval_mcq(
        arm,
        data_jsonl=PERMUTED_JSONL,
        prompt_style="with_options",
        score_target="label",
        suffix="cyclic_label",
    )
    perm_rows = aggregate_permutation_scores(read_jsonl(perm_label_path), read_jsonl(PERMUTED_JSONL))
    perm_output = RUN_DIR / f"{arm.name}_cyclic_label_aggregated.jsonl"
    write_jsonl(perm_output, perm_rows)
    return {
        "arm": arm.name,
        "mode": arm.mode,
        "checkpoint": None if arm.checkpoint is None else path_for_cli(arm.checkpoint),
        "max_loops": arm.max_loops if arm.mode != "base" else None,
        "paths": {
            "label": path_for_cli(label_path),
            "content_question_only": path_for_cli(content_path),
            "cyclic_label": path_for_cli(perm_label_path),
            "cyclic_label_aggregated": path_for_cli(perm_output),
        },
        "metrics": {
            "label": summarize_rows(read_jsonl(label_path)),
            "content_question_only": summarize_rows(read_jsonl(content_path)),
            "cyclic_label_aggregated": summarize_rows(perm_rows),
        },
    }


def compare_to_base(arm_payload: dict[str, Any], base_payload: dict[str, Any]) -> dict[str, Any]:
    data = rows_by_id(read_jsonl(ARC_JSONL))
    comparisons = {}
    method_to_path_key = {
        "label": "label",
        "content_question_only": "content_question_only",
        "cyclic_label_aggregated": "cyclic_label_aggregated",
    }
    for method, path_key in method_to_path_key.items():
        base_rows = rows_by_id(read_jsonl(resolve_path(base_payload["paths"][path_key])))
        arm_rows = rows_by_id(read_jsonl(resolve_path(arm_payload["paths"][path_key])))
        comparisons[method] = summarize(
            paired_rows(base_rows, arm_rows, data),
            benchmark=f"{ARC_CONFIG}:{method}",
        )
    return comparisons


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    start = summary["comparisons"].get("start_loop4")
    if not start:
        return {
            "status": "measurement_inconclusive",
            "passed": False,
            "reason": "Missing start_loop4 comparison.",
        }
    label_delta = int(start["label"]["delta"])
    content_delta = int(start["content_question_only"]["delta"])
    cyclic_delta = int(start["cyclic_label_aggregated"]["delta"])
    best_debiased_delta = max(content_delta, cyclic_delta)
    closure = best_debiased_delta - label_delta
    if label_delta <= -MIN_LABEL_GAP and best_debiased_delta >= -MAX_DEBIASED_GAP and closure >= MIN_CLOSURE:
        status = "selection_bias_likely"
        next_step = (
            "Do not train direct preservation yet. Regenerate MCQ benchmark claims with content/permutation "
            "scoring and then decide whether any residual degradation remains."
        )
        passed = True
    elif best_debiased_delta < -MAX_DEBIASED_GAP:
        status = "content_degradation_persists"
        next_step = (
            "Debiased scoring still trails base. The bounded max_loops=1 direct-preservation probe remains justified."
        )
        passed = False
    else:
        status = "measurement_inconclusive"
        next_step = "Inspect per-example debias rows before spending on training."
        passed = False
    return {
        "status": status,
        "passed": passed,
        "label_delta": label_delta,
        "content_delta": content_delta,
        "cyclic_delta": cyclic_delta,
        "best_debiased_delta": best_debiased_delta,
        "closure_vs_label": closure,
        "thresholds": {
            "min_label_gap": MIN_LABEL_GAP,
            "max_debiased_gap": MAX_DEBIASED_GAP,
            "min_closure": MIN_CLOSURE,
        },
        "next_step": next_step,
    }


def write_report(payload: dict[str, Any]) -> None:
    summary_path = RUN_DIR / "summary.json"
    write_json(summary_path, payload)
    lines = [
        f"# MCQ Debias Diagnostic - {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source summary: `{payload['source_summary']}`",
        f"- ARC config: `{payload['arc_config']}`",
        f"- ARC limit: `{payload['arc_limit']}`",
        f"- Decision: {payload['decision']['next_step']}",
        "",
        "| arm | method | correct | total | edge-minus-middle | delta vs base |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm_name, arm_payload in payload["arms"].items():
        comparisons = payload["comparisons"].get(arm_name, {})
        for method, metrics in arm_payload["metrics"].items():
            delta = "" if arm_name == "base" else str(comparisons.get(method, {}).get("delta"))
            lines.append(
                f"| `{arm_name}` | `{method}` | {metrics['correct']} | {metrics['total']} | "
                f"{metrics['edge_minus_middle']} | {delta} |"
            )
    lines.extend(["", "## Decision Payload", "", "```json", json.dumps(payload["decision"], indent=2), "```"])
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"), flush=True)
    update_current_source_summary(summary_path)


def commit_outputs() -> None:
    if not PUSH_RESULTS:
        return
    paths = [
        path_for_cli(RUN_DIR),
        path_for_cli(current_source_summary_file()),
    ]
    run(["git", "add", "-f", *paths], check=False)
    diff = run(["git", "diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        print("No MCQ debias outputs to commit.", flush=True)
        return
    run(["git", "commit", "-m", f"Record MCQ debias diagnostic {RUN_ID}"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    source_path, source, nested_path, nested = source_payloads()
    prepare_arc_data()
    arms = arms_from_summary(nested)
    restore_arm_checkpoints(nested, arms)

    arm_payloads = {arm.name: evaluate_arm(arm) for arm in arms}
    base_payload = arm_payloads["base"]
    comparisons = {
        name: compare_to_base(payload, base_payload)
        for name, payload in arm_payloads.items()
        if name != "base"
    }
    payload: dict[str, Any] = {
        "run_id": RUN_ID,
        "kind": "stage5_mcq_debias_diagnostic",
        "source_summary": path_for_cli(source_path),
        "nested_source_summary": path_for_cli(nested_path),
        "input_source_kind": source.get("kind"),
        "arc_config": ARC_CONFIG,
        "arc_limit": ARC_LIMIT,
        "arc_seed": ARC_SEED,
        "arms": arm_payloads,
        "comparisons": comparisons,
    }
    decision = classify(payload)
    payload.update(
        {
            "status": decision["status"],
            "passed": decision["passed"],
            "decision": decision,
        }
    )
    write_report(payload)
    commit_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
