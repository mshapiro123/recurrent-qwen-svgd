"""Run the Stage 5 loop-1 regression battery before narrow depth training.

This is measurement-only. It evaluates one or more recurrent checkpoints
against base Qwen on frozen AI2 ARC item sets, using the existing debiased MCQ
benchmark runner and the paired non-inferiority assessor.
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

from eval.assess_regression_battery import build_assessment, write_markdown


RUN_ID = os.environ.get("STAGE5_REGRESSION_BATTERY_RUN_ID") or time.strftime(
    "stage5_regression_battery_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SOURCE = "outputs/stage5/stage5_chain_continuation_attribution_20260704_163056/summary.json"
BENCHMARKS = os.environ.get("STAGE5_REGRESSION_BENCHMARKS", "arc_easy,arc_challenge")
SCORE_TARGETS = os.environ.get(
    "STAGE5_REGRESSION_SCORE_TARGETS",
    "content_question_only,cyclic_label_aggregated",
)
TARGET_SPECS = os.environ.get(
    "STAGE5_REGRESSION_TARGET_SPECS",
    "content_question_only:mean,cyclic_label_aggregated:permutation_mean",
)
ARC_SPLIT = os.environ.get("STAGE5_REGRESSION_ARC_SPLIT", "all")
ARC_EASY_LIMIT = os.environ.get("STAGE5_REGRESSION_ARC_EASY_LIMIT", "all")
ARC_CHALLENGE_LIMIT = os.environ.get("STAGE5_REGRESSION_ARC_CHALLENGE_LIMIT", "all")
MARGIN = float(os.environ.get("STAGE5_REGRESSION_MARGIN", "0.03"))
YELLOW_MARGIN = float(os.environ.get("STAGE5_REGRESSION_YELLOW_MARGIN", "0.015"))
PUSH_RESULTS = os.environ.get("STAGE5_REGRESSION_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
RESUME_EXISTING = os.environ.get("STAGE5_REGRESSION_RESUME_EXISTING", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
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


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_target_specs(value: str) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for item in parse_csv(value):
        if ":" in item:
            score_target, aggregate = item.split(":", 1)
            specs.append((score_target.strip(), aggregate.strip()))
        else:
            specs.append((item.strip(), "mean"))
    return specs


def safe_label(source_summary: str, payload: dict[str, Any]) -> str:
    run_id = str(payload.get("run_id") or "").strip()
    if run_id:
        return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in run_id)[-80:]
    return resolve_path(source_summary).parent.name[-80:]


def source_summaries() -> list[str]:
    raw = os.environ.get("STAGE5_REGRESSION_SOURCE_SUMMARIES", "").strip()
    if raw:
        return parse_csv(raw)
    current = os.environ.get("STAGE5_REGRESSION_CURRENT_SOURCE_SUMMARY", DEFAULT_SOURCE).strip()
    return [current]


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def restore_current_source_pointer(value: str) -> None:
    pointer = current_source_summary_file()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(value.strip() + "\n", encoding="utf-8")


def run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
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
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def benchmark_env(
    *,
    child_run_id: str,
    source_summary: str,
    base_reuse_run_id: str | None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "STAGE5_BENCHMARK_SUITE_RUN_ID": child_run_id,
            "STAGE5_BENCHMARK_SOURCE_SUMMARY": source_summary,
            "STAGE5_BENCHMARKS": BENCHMARKS,
            "STAGE5_BENCHMARK_SCORE_TARGETS": SCORE_TARGETS,
            "STAGE5_BENCHMARK_AGGREGATES": "mean",
            "STAGE5_BENCHMARK_ARC_EASY_SPLIT": ARC_SPLIT,
            "STAGE5_BENCHMARK_ARC_CHALLENGE_SPLIT": ARC_SPLIT,
            "STAGE5_BENCHMARK_ARC_EASY_LIMIT": ARC_EASY_LIMIT,
            "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT": ARC_CHALLENGE_LIMIT,
            "STAGE5_BENCHMARK_ARC_EASY_OFFSET": "0",
            "STAGE5_BENCHMARK_ARC_CHALLENGE_OFFSET": "0",
            "STAGE5_BENCHMARK_MAX_LOOPS": "1",
            "STAGE5_BENCHMARK_FORCED_LOOP_COUNT": "1",
            "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL": "0",
            "STAGE5_BENCHMARK_INCLUDE_LOOP_DIAGNOSTICS": "1",
            "STAGE5_BENCHMARK_CONTINUE_ON_FAILURE": "1",
            "STAGE5_BENCHMARK_PUSH": "1" if PUSH_RESULTS else "0",
        }
    )
    if base_reuse_run_id:
        env["STAGE5_BENCHMARK_BASE_REUSE_RUN_ID"] = base_reuse_run_id
    return env


def assess_child(*, child_summary: Path, assessment_run_id: str) -> dict[str, Any]:
    payload = build_assessment(
        suite_summary=child_summary,
        suite_payload=read_json(child_summary),
        required_benchmarks=parse_csv(BENCHMARKS),
        target_specs=parse_target_specs(TARGET_SPECS),
        margin=MARGIN,
        yellow_margin=YELLOW_MARGIN,
        run_id=assessment_run_id,
    )
    assessment_dir = RUN_DIR / assessment_run_id
    assessment_dir.mkdir(parents=True, exist_ok=True)
    (assessment_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, assessment_dir / "summary.md")
    return payload


def combined_status(assessments: list[dict[str, Any]]) -> str:
    statuses = [str(item.get("status")) for item in assessments]
    if any(status.startswith("red") for status in statuses):
        return "red_regression_established"
    if any(status.startswith("yellow") for status in statuses):
        return "yellow_drift_watch"
    if any(status.startswith("incomplete") for status in statuses):
        return "incomplete_missing_rows"
    if any(status.startswith("grey") for status in statuses):
        return "grey_underpowered"
    return "green_noninferior"


def write_summary(payload: dict[str, Any]) -> None:
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Regression Battery: {RUN_ID}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Benchmarks: `{payload['benchmarks']}`",
        f"- ARC split: `{payload['arc_split']}`",
        f"- Loop: `1 forced`",
        "",
        "## Assessments",
    ]
    for item in payload["assessments"]:
        lines.append(
            f"- `{item['label']}`: status=`{item['status']}`, "
            f"suite=`{item['suite_summary']}`, assessment=`{item['assessment_summary']}`"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- This is AI2 ARC, not ARC-AGI.",
            "- ARC is used here only as an evaluation/regression instrument.",
            "- HellaSwag, Winogrande, LAMBADA, and natural-text NLL canaries are recorded as pending extensions.",
        ]
    )
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def commit_results() -> None:
    if not PUSH_RESULTS:
        return
    run(["git", "add", "-f", path_for_cli(RUN_DIR), path_for_cli(current_source_summary_file())])
    diff = run(["git", "diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        print("No regression battery changes to commit.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 regression battery {RUN_ID} [skip ci]"])
    pushed = run(["git", "push", "origin", "main"], check=False)
    if pushed.returncode == 0:
        return
    print("Regression battery push failed; attempting one autostash rebase and retry.", flush=True)
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
    run(["git", "push", "origin", "main"])


def main() -> int:
    sources = source_summaries()
    if not sources:
        raise RuntimeError("No STAGE5_REGRESSION_SOURCE_SUMMARIES provided.")
    pointer_before = current_source_summary_file().read_text(encoding="utf-8").strip() if current_source_summary_file().exists() else sources[0]
    assessments: list[dict[str, Any]] = []
    first_child_run_id: str | None = None
    try:
        for idx, source in enumerate(sources):
            payload = read_json(source)
            label = safe_label(source, payload)
            child_run_id = f"{RUN_ID}_{idx + 1}_{label}"
            child_summary = ROOT / "outputs" / "stage5" / child_run_id / "summary.json"
            print(f"regression_source={source} child_run_id={child_run_id}", flush=True)
            if RESUME_EXISTING and child_summary.exists():
                print(f"resume_existing_benchmark_suite={path_for_cli(child_summary)}", flush=True)
            else:
                run(
                    [sys.executable, "colab/run_stage5_benchmark_suite.py"],
                    env=benchmark_env(
                        child_run_id=child_run_id,
                        source_summary=source,
                        base_reuse_run_id=first_child_run_id if idx else None,
                    ),
                )
            if first_child_run_id is None:
                first_child_run_id = child_run_id
            assessment_payload = assess_child(child_summary=child_summary, assessment_run_id=f"{idx + 1}_{label}")
            assessments.append(
                {
                    "label": label,
                    "source_summary": source,
                    "suite_summary": path_for_cli(child_summary),
                    "assessment_summary": path_for_cli(RUN_DIR / f"{idx + 1}_{label}" / "summary.json"),
                    "status": assessment_payload["status"],
                    "pooled": assessment_payload["pooled"],
                }
            )
    finally:
        restore_current_source_pointer(pointer_before)

    summary = {
        "kind": "stage5_regression_battery",
        "run_id": RUN_ID,
        "status": combined_status([read_json(ROOT / item["assessment_summary"]) for item in assessments]),
        "benchmarks": BENCHMARKS,
        "score_targets": SCORE_TARGETS,
        "target_specs": TARGET_SPECS,
        "arc_split": ARC_SPLIT,
        "accuracy_margin": MARGIN,
        "yellow_margin": YELLOW_MARGIN,
        "source_summaries": sources,
        "assessments": assessments,
        "policy": {
            "ai2_arc_not_arc_agi": True,
            "arc_eval_only": True,
            "if_red_or_yellow": "Review before route comparison; add uniform generic LM rehearsal to future training arms only if confirmed damage is real.",
            "tier1_canary_status": "pending",
            "hellaswag_winogrande_lambada_status": "pending",
        },
    }
    write_summary(summary)
    commit_results()
    print(json.dumps({"run_id": RUN_ID, "status": summary["status"], "summary": path_for_cli(RUN_DIR / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
