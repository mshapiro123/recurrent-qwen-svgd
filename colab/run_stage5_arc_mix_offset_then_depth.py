"""Run ARC-mix offset confirmation, then optionally launch depth-routing SFT.

This is a GPU-session efficiency wrapper. It first runs the independent
offset-256 ARC-Easy/ARC-Challenge confirmation for the current recovered
ARC-mix checkpoint. It launches the bounded learned-loop-control ARC-mix
depth-routing probe only when content competence replicates and the cyclic
debiased surface is non-negative by default. Content recovery is a leading
indicator; cyclic-debiased scoring is the steering gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.colab_auth import ensure_hf_token_from_colab  # noqa: E402


RUN_ID = os.environ.get("STAGE5_ARC_MIX_CHAIN_RUN_ID") or time.strftime(
    "stage5_arc_mix_offset_then_depth_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
SOURCE_SUMMARY = os.environ.get(
    "STAGE5_ARC_MIX_CHAIN_SOURCE_SUMMARY",
    "outputs/stage5/stage5_content_arcmix_qonly_optiontext_arc256_check_20260623_123424/summary.json",
)
OFFSET_RUN_ID = os.environ.get(
    "STAGE5_ARC_MIX_CHAIN_OFFSET_RUN_ID",
    f"{RUN_ID}_offset256_confirm",
)
OFFSET_SUMMARY_OVERRIDE = os.environ.get("STAGE5_ARC_MIX_CHAIN_OFFSET_SUMMARY", "").strip()
DEPTH_RUN_ID = os.environ.get(
    "STAGE5_ARC_MIX_CHAIN_DEPTH_RUN_ID",
    f"{RUN_ID}_depth_routing_probe",
)
EXECUTE_DEPTH = os.environ.get("STAGE5_ARC_MIX_CHAIN_EXECUTE_DEPTH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
RUN_POST_DEPTH_DEBIASED_GATE = os.environ.get(
    "STAGE5_ARC_MIX_CHAIN_RUN_POST_DEPTH_DEBIASED_GATE", "1"
).strip().lower() in {"1", "true", "yes", "y"}
ALLOWED_NEGATIVE_DELTA = int(os.environ.get("STAGE5_ARC_MIX_CHAIN_ALLOWED_NEGATIVE_DELTA", "0"))
MIN_EXAMPLES = int(os.environ.get("STAGE5_ARC_MIX_CHAIN_MIN_EXAMPLES", "256"))
MIN_EXAMPLES_ARC_EASY = int(os.environ.get("STAGE5_ARC_MIX_CHAIN_MIN_EXAMPLES_ARC_EASY", str(MIN_EXAMPLES)))
MIN_EXAMPLES_ARC_CHALLENGE = int(os.environ.get("STAGE5_ARC_MIX_CHAIN_MIN_EXAMPLES_ARC_CHALLENGE", "32"))
CONTENT_ALLOWED_NEGATIVE_DELTA = int(
    os.environ.get("STAGE5_ARC_MIX_CHAIN_CONTENT_ALLOWED_NEGATIVE_DELTA", str(ALLOWED_NEGATIVE_DELTA))
)
DEBIASED_ALLOWED_NEGATIVE_DELTA = int(
    os.environ.get("STAGE5_ARC_MIX_CHAIN_DEBIASED_ALLOWED_NEGATIVE_DELTA", "0")
)
POST_DEPTH_MIN_EXAMPLES = int(os.environ.get("STAGE5_ARC_MIX_CHAIN_POST_DEPTH_MIN_EXAMPLES", "128"))
POST_DEPTH_DEBIASED_ALLOWED_NEGATIVE_DELTA = int(
    os.environ.get("STAGE5_ARC_MIX_CHAIN_POST_DEPTH_DEBIASED_ALLOWED_NEGATIVE_DELTA", "0")
)
POST_DEPTH_LIMIT = os.environ.get("STAGE5_ARC_MIX_CHAIN_POST_DEPTH_LIMIT", "128")
POST_DEPTH_OFFSET = os.environ.get("STAGE5_ARC_MIX_CHAIN_POST_DEPTH_OFFSET", "256")
PUSH_RESULTS = os.environ.get("STAGE5_ARC_MIX_CHAIN_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

REQUIRED_READOUTS = (
    ("arc_easy", "content_question_only", "mean"),
    ("arc_easy", "cyclic_label_aggregated", "permutation_mean"),
    ("arc_challenge", "content_question_only", "mean"),
    ("arc_challenge", "cyclic_label_aggregated", "permutation_mean"),
)


DEFAULT_MIN_EXAMPLES_BY_BENCHMARK = {
    "arc_easy": MIN_EXAMPLES_ARC_EASY,
    "arc_challenge": MIN_EXAMPLES_ARC_CHALLENGE,
}


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    log_name: str | None = None,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
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
    proc = subprocess.CompletedProcess(cmd, process.wait(), stdout, None)
    if log_name:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def paired_row(payload: dict[str, Any], benchmark: str, score_target: str, aggregate: str) -> dict[str, Any] | None:
    row = (
        (payload.get("paired_comparisons") or {})
        .get(benchmark, {})
        .get(score_target, {})
        .get(aggregate)
    )
    return row if isinstance(row, dict) else None


def evidence_row(
    payload: dict[str, Any],
    benchmark: str,
    score_target: str,
    aggregate: str,
    *,
    min_examples: int = MIN_EXAMPLES,
    min_examples_by_benchmark: dict[str, int] | None = None,
    content_allowed_negative_delta: int = CONTENT_ALLOWED_NEGATIVE_DELTA,
    debiased_allowed_negative_delta: int = DEBIASED_ALLOWED_NEGATIVE_DELTA,
) -> dict[str, Any]:
    required_examples = (min_examples_by_benchmark or {}).get(benchmark, min_examples)
    allowed_negative_delta = (
        debiased_allowed_negative_delta
        if score_target == "cyclic_label_aggregated"
        else content_allowed_negative_delta
    )
    row = paired_row(payload, benchmark, score_target, aggregate)
    if not row:
        return {
            "benchmark": benchmark,
            "score_target": score_target,
            "aggregate": aggregate,
            "present": False,
            "paired_examples": 0,
            "required_examples": required_examples,
            "allowed_negative_delta": allowed_negative_delta,
            "correct_delta_recurrent_vs_base": None,
            "passed": False,
        }
    delta = int(row.get("correct_delta_recurrent_vs_base", 0) or 0)
    paired = int(row.get("paired_examples", 0) or 0)
    return {
        "benchmark": benchmark,
        "score_target": score_target,
        "aggregate": aggregate,
        "present": True,
        "paired_examples": paired,
        "required_examples": required_examples,
        "allowed_negative_delta": allowed_negative_delta,
        "base_correct": int(row.get("base_correct", 0) or 0),
        "recurrent_correct": int(row.get("recurrent_correct", 0) or 0),
        "correct_delta_recurrent_vs_base": delta,
        "wins": int(row.get("wins", 0) or 0),
        "losses": int(row.get("losses", 0) or 0),
        "ties": int(row.get("ties", 0) or 0),
        "sign_test_p_value": row.get("sign_test_p_value"),
        "passed": paired >= required_examples and delta >= -allowed_negative_delta,
    }


def assess_offset_confirmation(
    payload: dict[str, Any],
    *,
    min_examples: int = MIN_EXAMPLES,
    min_examples_by_benchmark: dict[str, int] | None = None,
    content_allowed_negative_delta: int = CONTENT_ALLOWED_NEGATIVE_DELTA,
    debiased_allowed_negative_delta: int = DEBIASED_ALLOWED_NEGATIVE_DELTA,
) -> dict[str, Any]:
    failures = payload.get("failures") or []
    completed = payload.get("status") == "completed" and not failures
    min_by_benchmark = (
        dict(DEFAULT_MIN_EXAMPLES_BY_BENCHMARK)
        if min_examples_by_benchmark is None and min_examples == MIN_EXAMPLES
        else (min_examples_by_benchmark or {"arc_easy": min_examples, "arc_challenge": min(min_examples, MIN_EXAMPLES_ARC_CHALLENGE)})
    )
    evidence = [
        evidence_row(
            payload,
            *readout,
            min_examples=min_examples,
            min_examples_by_benchmark=min_by_benchmark,
            content_allowed_negative_delta=content_allowed_negative_delta,
            debiased_allowed_negative_delta=debiased_allowed_negative_delta,
        )
        for readout in REQUIRED_READOUTS
    ]
    evidence_passed = all(row["passed"] for row in evidence)
    passed = completed and evidence_passed
    if passed:
        content_deltas = [
            int(row["correct_delta_recurrent_vs_base"] or 0)
            for row in evidence
            if row["score_target"] == "content_question_only"
        ]
        cyclic_deltas = [
            int(row["correct_delta_recurrent_vs_base"] or 0)
            for row in evidence
            if row["score_target"] == "cyclic_label_aggregated"
        ]
        debiased_positive = any(delta > 0 for delta in cyclic_deltas)
        debiased_flat = cyclic_deltas and all(delta == 0 for delta in cyclic_deltas)
        content_replicated = any(delta > 0 for delta in content_deltas)
        if debiased_positive:
            status = "offset_confirmed_debiased_positive"
        elif debiased_flat:
            status = "offset_confirmed_debiased_flat"
        else:
            status = "offset_confirmed_debiased_tolerated_negative"
        next_step = (
            "Launch the bounded learned-loop ARC-mix depth-routing probe; "
            "treat cyclic-debiased scoring as the post-depth survival gate."
        )
    elif not completed:
        status = "offset_incomplete"
        next_step = "Inspect offset-confirmation benchmark logs and rerun failed slices."
    else:
        status = "offset_regressed"
        next_step = "Do not launch depth-routing SFT yet; diagnose the regressed offset readouts."
    return {
        "status": status,
        "passed": passed,
        "next_step": next_step,
        "content_allowed_negative_delta": content_allowed_negative_delta,
        "debiased_allowed_negative_delta": debiased_allowed_negative_delta,
        "min_examples": min_examples,
        "min_examples_by_benchmark": min_by_benchmark,
        "failures": failures,
        "evidence": evidence,
        "content_replicated": any(
            row["score_target"] == "content_question_only"
            and int(row["correct_delta_recurrent_vs_base"] or 0) > 0
            for row in evidence
        ),
        "debiased_positive": any(
            row["score_target"] == "cyclic_label_aggregated"
            and int(row["correct_delta_recurrent_vs_base"] or 0) > 0
            for row in evidence
        ),
    }


def run_offset_confirmation() -> Path:
    if OFFSET_SUMMARY_OVERRIDE:
        summary = resolve_path(OFFSET_SUMMARY_OVERRIDE)
        if not summary.exists():
            raise FileNotFoundError(f"Requested offset summary does not exist: {summary}")
        print(f"Reusing offset summary: {path_for_cli(summary)}", flush=True)
        return summary
    env = os.environ.copy()
    env["STAGE5_BENCHMARK_SUITE_RUN_ID"] = OFFSET_RUN_ID
    env["STAGE5_BENCHMARK_SOURCE_SUMMARY"] = SOURCE_SUMMARY
    env["STAGE5_BENCHMARKS"] = "arc_easy,arc_challenge"
    env["STAGE5_BENCHMARK_ARC_EASY_LIMIT"] = "256"
    env["STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT"] = "256"
    env["STAGE5_BENCHMARK_ARC_EASY_OFFSET"] = "256"
    env["STAGE5_BENCHMARK_ARC_CHALLENGE_OFFSET"] = "256"
    env["STAGE5_BENCHMARK_SCORE_TARGETS"] = "content_question_only,cyclic_label_aggregated"
    env["STAGE5_BENCHMARK_AGGREGATES"] = "mean"
    env["STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL"] = "1"
    env["STAGE5_BENCHMARK_PUSH"] = "1"
    run([sys.executable, "colab/run_stage5_benchmark_suite.py"], env=env, log_name="offset_confirmation.log")
    summary = ROOT / "outputs" / "stage5" / OFFSET_RUN_ID / "summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"Offset confirmation did not write summary: {summary}")
    return summary


def run_depth_routing(source_summary: Path) -> tuple[int, Path | None]:
    env = os.environ.copy()
    env["STAGE5_ARC_MIX_RUN_ID"] = DEPTH_RUN_ID
    env["STAGE5_ARC_MIX_SOURCE_SUMMARY"] = path_for_cli(source_summary)
    env.setdefault("STAGE5_ARC_MIX_OPUS_LIMIT", "0")
    env.setdefault("STAGE5_ARC_MIX_ARC_TRAIN_LIMIT", "0")
    env.setdefault("STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT", "2")
    env.setdefault("STAGE5_ARC_MIX_ARC_EASY_REPEAT", "6")
    env.setdefault("STAGE5_ARC_MIX_ARC_CHALLENGE_TARGET_LOOP", "3")
    env.setdefault("STAGE5_ARC_MIX_ARC_EASY_TARGET_LOOP", "1")
    env.setdefault("STAGE5_ARC_MIX_ARC_CHALLENGE_ROUTING_TYPE", "deep_narrow")
    env.setdefault("STAGE5_ARC_MIX_ARC_EASY_ROUTING_TYPE", "direct")
    env.setdefault("STAGE5_ARC_MIX_PROMPT_STYLE", "question_only")
    env.setdefault("STAGE5_ARC_MIX_SCORE_TARGET", "option_text")
    env.setdefault("STAGE5_ARC_MIX_ARC_EVAL_CONFIG", "ARC-Challenge")
    env.setdefault("STAGE5_ARC_MIX_ARC_EVAL_LIMIT", "128")
    env.setdefault("STAGE5_ARC_MIX_ARMS", "arc_mix_response_w02_lr2e6")
    env.setdefault("STAGE5_ARC_MIX_USE_LEARNED_LOOP_CONTROL", "1")
    env.setdefault("STAGE5_ARC_MIX_EVAL_USE_LEARNED_LOOP_CONTROL", "1")
    env.setdefault("STAGE5_ARC_MIX_LOOP_CONTROL_CE_WEIGHT", "0.05")
    env.setdefault("STAGE5_ARC_MIX_HALT_TARGET_NLL_WEIGHT", "0.03")
    env.setdefault("STAGE5_ARC_MIX_MIN_MARGIN_DELTA", "-0.05")
    env.setdefault("STAGE5_ARC_MIX_MAX_PREDICTION_SHIFT", "16")
    env.setdefault("STAGE5_ARC_MIX_PUSH", "1")
    proc = run(
        [sys.executable, "colab/run_stage5_balanced_arc_mix_gate.py"],
        env=env,
        check=False,
        log_name="depth_routing.log",
    )
    summary = ROOT / "outputs" / "stage5" / DEPTH_RUN_ID / "summary.json"
    return proc.returncode, summary if summary.exists() else None


def run_post_depth_debiased_gate(depth_summary: Path, checkpoint: str | None) -> Path:
    env = os.environ.copy()
    env["STAGE5_BENCHMARK_SUITE_RUN_ID"] = f"{DEPTH_RUN_ID}_debiased_gate"
    env["STAGE5_BENCHMARK_SOURCE_SUMMARY"] = path_for_cli(depth_summary)
    if checkpoint:
        env["STAGE5_BENCHMARK_CHECKPOINT"] = checkpoint
    env["STAGE5_BENCHMARKS"] = "arc_easy,arc_challenge"
    env["STAGE5_BENCHMARK_ARC_EASY_LIMIT"] = POST_DEPTH_LIMIT
    env["STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT"] = POST_DEPTH_LIMIT
    env["STAGE5_BENCHMARK_ARC_EASY_OFFSET"] = POST_DEPTH_OFFSET
    env["STAGE5_BENCHMARK_ARC_CHALLENGE_OFFSET"] = POST_DEPTH_OFFSET
    env["STAGE5_BENCHMARK_SCORE_TARGETS"] = "content_question_only,cyclic_label_aggregated"
    env["STAGE5_BENCHMARK_AGGREGATES"] = "mean"
    env["STAGE5_BENCHMARK_PUSH"] = "1"
    run([sys.executable, "colab/run_stage5_benchmark_suite.py"], env=env, log_name="post_depth_debiased_gate.log")
    summary = ROOT / "outputs" / "stage5" / f"{DEPTH_RUN_ID}_debiased_gate" / "summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"Post-depth debiased gate did not write summary: {summary}")
    return summary


def selected_checkpoint_from_payload(payload: dict[str, Any]) -> str | None:
    best = payload.get("best_arm") if isinstance(payload.get("best_arm"), dict) else None
    if best:
        checkpoint_row = best.get("best_checkpoint") if isinstance(best.get("best_checkpoint"), dict) else {}
        checkpoint = checkpoint_row.get("checkpoint")
        if checkpoint:
            return str(checkpoint)
    checkpoint = payload.get("checkpoint")
    return str(checkpoint) if checkpoint else None


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def update_current_source_summary(summary_path: Path) -> None:
    pointer = current_source_summary_file()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")


def write_report(payload: dict[str, Any]) -> None:
    summary = RUN_DIR / "summary.json"
    report = RUN_DIR / "summary.md"
    write_json(summary, payload)
    update_current_source_summary(summary)
    lines = [
        f"# Stage 5 ARC-Mix Offset-Then-Depth Chain - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Passed offset: `{payload['offset_assessment']['passed']}`",
        f"- Launched depth routing: `{payload['depth_launched']}`",
        f"- Offset summary: `{payload['offset_summary']}`",
        f"- Depth summary: `{payload.get('depth_summary') or 'not_run'}`",
        f"- Post-depth debiased summary: `{payload.get('post_depth_debiased_summary') or 'not_run'}`",
        f"- Checkpoint: `{payload.get('checkpoint') or 'none'}`",
        f"- Next step: {payload['next_step']}",
        "",
        "## Offset Evidence",
        "",
    ]
    for row in payload["offset_assessment"]["evidence"]:
        lines.append(
            f"- `{row['benchmark']}` `{row['score_target']}` / `{row['aggregate']}`: "
            f"paired `{row['paired_examples']}`, delta `{row['correct_delta_recurrent_vs_base']}`, "
            f"W/L/T `{row.get('wins', 0)}/{row.get('losses', 0)}/{row.get('ties', 0)}`, "
            f"passed `{row['passed']}`"
        )
    if payload.get("post_depth_debiased_assessment"):
        lines.extend(
            [
                "",
                "## Post-Depth Debiased Evidence",
                "",
                "Cyclic-debiased scoring is the primary survival gate here; content-question scoring is a leading indicator.",
                "",
            ]
        )
        for row in payload["post_depth_debiased_assessment"]["evidence"]:
            lines.append(
                f"- `{row['benchmark']}` `{row['score_target']}` / `{row['aggregate']}`: "
                f"paired `{row['paired_examples']}`, delta `{row['correct_delta_recurrent_vs_base']}`, "
                f"W/L/T `{row.get('wins', 0)}/{row.get('losses', 0)}/{row.get('ties', 0)}`, "
                f"passed `{row['passed']}`"
            )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"), flush=True)


def committable_files() -> list[str]:
    files = [
        path_for_cli(path)
        for path in RUN_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".md", ".log", ".txt"}
    ]
    pointer = current_source_summary_file()
    if pointer.exists():
        files.append(path_for_cli(pointer))
    return sorted(set(files))


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    files = committable_files()
    if not files:
        return
    run(["git", "add", "-f", *files], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No offset-then-depth chain artifacts changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record ARC-mix offset-depth chain {RUN_ID} [skip ci]"])
    run(["git", "push", "origin", "main"], check=False)


def main() -> int:
    ensure_hf_token_from_colab()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    offset_summary = run_offset_confirmation()
    offset_payload = read_json(offset_summary)
    offset_assessment = assess_offset_confirmation(offset_payload)
    depth_launched = False
    depth_returncode = None
    depth_summary = None
    depth_payload: dict[str, Any] | None = None
    post_depth_debiased_summary = None
    post_depth_debiased_assessment = None
    if offset_assessment["passed"] and EXECUTE_DEPTH:
        depth_launched = True
        depth_returncode, depth_summary_path = run_depth_routing(offset_summary)
        if depth_summary_path:
            depth_summary = path_for_cli(depth_summary_path)
            depth_payload = read_json(depth_summary_path)

    checkpoint = selected_checkpoint_from_payload(depth_payload or offset_payload)
    if (
        depth_launched
        and depth_returncode == 0
        and depth_summary
        and RUN_POST_DEPTH_DEBIASED_GATE
    ):
        post_depth_summary_path = run_post_depth_debiased_gate(resolve_path(depth_summary), checkpoint)
        post_depth_debiased_summary = path_for_cli(post_depth_summary_path)
        post_depth_debiased_assessment = assess_offset_confirmation(
            read_json(post_depth_summary_path),
            min_examples=POST_DEPTH_MIN_EXAMPLES,
            debiased_allowed_negative_delta=POST_DEPTH_DEBIASED_ALLOWED_NEGATIVE_DELTA,
        )
    if depth_launched and depth_returncode not in {0, None}:
        status = "depth_failed"
        next_step = "Inspect depth-routing logs before proceeding."
    elif post_depth_debiased_assessment and not post_depth_debiased_assessment["passed"]:
        status = "depth_completed_debiased_warning"
        next_step = (
            "Depth training completed, but the post-depth debiased gate did not clear; "
            "treat content gains as provisional and inspect cyclic scoring."
        )
    elif offset_assessment["passed"] and depth_launched:
        status = "depth_completed"
        next_step = "Review post-depth content and cyclic-debiased gate before extending training."
    elif offset_assessment["passed"]:
        status = "offset_confirmed_depth_skipped"
        next_step = "Launch depth-routing probe manually or set STAGE5_ARC_MIX_CHAIN_EXECUTE_DEPTH=1."
    else:
        status = "offset_not_confirmed"
        next_step = offset_assessment["next_step"]
    payload = {
        "run_id": RUN_ID,
        "kind": "stage5_arc_mix_offset_then_depth_chain",
        "status": status,
        "source_summary": SOURCE_SUMMARY,
        "offset_summary": path_for_cli(offset_summary),
        "offset_assessment": offset_assessment,
        "depth_launched": depth_launched,
        "depth_returncode": depth_returncode,
        "depth_summary": depth_summary,
        "post_depth_debiased_summary": post_depth_debiased_summary,
        "post_depth_debiased_assessment": post_depth_debiased_assessment,
        "checkpoint": checkpoint,
        "next_step": next_step,
    }
    write_report(payload)
    commit_results()
    if depth_returncode:
        return int(depth_returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
