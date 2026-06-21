"""Compare base, recurrent-start, and recovered recurrent checkpoints on ARC.

This runner is intentionally evaluation-only. It is meant to follow a
curriculum run and answer the benchmark-facing question:

Did the recovered recurrent checkpoint close the exact-grid gap to the original
base Qwen model on the same ARC-AGI split?
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from colab.stage5_limits import difficulty_args, limit_args, limit_label, parse_optional_limit
    from colab.stage5_model_metadata import model_metadata
except ModuleNotFoundError:  # pragma: no cover - direct ``python colab/script.py`` execution
    from stage5_limits import difficulty_args, limit_args, limit_label, parse_optional_limit
    from stage5_model_metadata import model_metadata


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_ARC_AGI_RECOVERED_BENCHMARK_RUN_ID") or time.strftime(
    "stage5_arc_agi_recovered_benchmark_%Y%m%d_%H%M%S"
)
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

BASE_RUN_ID = os.environ.get("STAGE5_BASE_RUN_ID", "stage4_opus_a100_20260620")
BASE_RUN_DIR = ROOT / "outputs" / "stage4" / BASE_RUN_ID
DEFAULT_PHASE1_CKPT = BASE_RUN_DIR / "phase1" / "phase1_step_500.pt"

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
CURRICULUM_SUMMARY = os.environ.get("STAGE5_ARC_AGI_CURRICULUM_SUMMARY", "")
RECOVERED_CKPT = os.environ.get("STAGE5_ARC_AGI_RECOVERED_CKPT", "")
PHASE1_START_CKPT = os.environ.get("STAGE5_PHASE1_CKPT", "")
DATA_ROOT = ROOT / "data" / "arc_agi"
ARC_AGI_1_REPO = os.environ.get("ARC_AGI_1_REPO", "https://github.com/fchollet/ARC-AGI.git")
ARC_AGI_2_REPO = os.environ.get("ARC_AGI_2_REPO", "https://github.com/arcprize/ARC-AGI-2.git")
ARC_VERSION = os.environ.get("STAGE5_ARC_AGI_VERSION", "1")
ARC_SPLIT = os.environ.get("STAGE5_ARC_AGI_SPLIT", "evaluation")
LIMIT = parse_optional_limit(os.environ.get("STAGE5_ARC_AGI_LIMIT", "20"))
DIFFICULTY_BUCKETS = os.environ.get("STAGE5_ARC_AGI_DIFFICULTY_BUCKETS", "")
EXAMPLES_PER_DIFFICULTY = parse_optional_limit(os.environ.get("STAGE5_ARC_AGI_EXAMPLES_PER_DIFFICULTY"))
MAX_NEW_TOKENS = int(os.environ.get("STAGE5_ARC_AGI_MAX_NEW_TOKENS", "512"))
GRID_FORMAT = os.environ.get("STAGE5_ARC_AGI_GRID_FORMAT", "compact")
GEOMETRY_TTA = os.environ.get("STAGE5_ARC_AGI_GEOMETRY_TTA", "none")
PROGRAM_PARSE_MODE = os.environ.get("STAGE5_ARC_AGI_PROGRAM_PARSE_MODE", "fallback")
SELECTION_STRATEGY = os.environ.get("STAGE5_ARC_AGI_SELECTION_STRATEGY", "heuristic")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_RECOVERED_BENCHMARK_PUSH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def run(cmd: list[str], *, check: bool = True, log_name: str | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)))
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
    proc = subprocess.CompletedProcess(cmd, process.wait(), stdout=stdout, stderr=None)
    if log_name:
        (RUN_DIR / log_name).write_text(stdout, encoding="utf-8")
    if check and proc.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return proc


def path_for_cli(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latest_curriculum_summary() -> Path:
    candidates = sorted(
        ROOT.glob("outputs/stage5/*curriculum*/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No curriculum summary found. Set STAGE5_ARC_AGI_CURRICULUM_SUMMARY or STAGE5_ARC_AGI_RECOVERED_CKPT.")
    return candidates[0]


def maybe_curriculum_summary() -> dict[str, Any] | None:
    if CURRICULUM_SUMMARY:
        return read_json(resolve_path(CURRICULUM_SUMMARY))
    if RECOVERED_CKPT:
        return None
    return read_json(latest_curriculum_summary())


def final_stage_row(curriculum: dict[str, Any]) -> dict[str, Any]:
    stages = curriculum.get("stages") or []
    if not stages:
        raise ValueError("Curriculum summary has no stages.")
    return stages[-1]


def recovered_checkpoint_from_curriculum(curriculum: dict[str, Any]) -> Path:
    final_stage = final_stage_row(curriculum)
    checkpoint = final_stage.get("selected_checkpoint", {}).get("checkpoint") or curriculum.get("final_checkpoint")
    if not checkpoint:
        raise ValueError("Curriculum summary has no selected checkpoint.")
    return resolve_path(checkpoint)


def phase1_start_checkpoint_from_curriculum(curriculum: dict[str, Any] | None) -> Path:
    if PHASE1_START_CKPT:
        return resolve_path(PHASE1_START_CKPT)
    if curriculum:
        stages = curriculum.get("stages") or []
        if stages and stages[0].get("resume_checkpoint"):
            return resolve_path(stages[0]["resume_checkpoint"])
    return resolve_path(DEFAULT_PHASE1_CKPT)


def recovered_checkpoint(curriculum: dict[str, Any] | None) -> Path:
    if RECOVERED_CKPT:
        return resolve_path(RECOVERED_CKPT)
    if curriculum is None:
        raise ValueError("Recovered checkpoint requires STAGE5_ARC_AGI_RECOVERED_CKPT or a curriculum summary.")
    return recovered_checkpoint_from_curriculum(curriculum)


def clone_or_update(repo_url: str, target: Path) -> None:
    if target.exists() and (target / ".git").exists():
        run(["git", "-C", str(target), "pull", "--ff-only"], check=False)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", repo_url, str(target)])


def resolve_tasks_path() -> Path:
    if user_path := os.environ.get("STAGE5_ARC_AGI_TASKS_PATH"):
        return resolve_path(user_path)
    if ARC_VERSION == "2":
        repo_dir = DATA_ROOT / "ARC-AGI-2"
        clone_or_update(ARC_AGI_2_REPO, repo_dir)
    else:
        repo_dir = DATA_ROOT / "ARC-AGI"
        clone_or_update(ARC_AGI_1_REPO, repo_dir)
    candidates = [
        repo_dir / "data" / ARC_SPLIT,
        repo_dir / ARC_SPLIT,
        repo_dir / "data" / f"{ARC_SPLIT}_challenges",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find ARC-AGI split {ARC_SPLIT!r} under {repo_dir}")


def eval_arc(label: str, *, mode: str, tasks_path: Path, checkpoint: Path | None = None) -> dict[str, Any]:
    summary_json = RUN_DIR / f"{label}_summary.json"
    cmd = [
        sys.executable,
        "eval/eval_arc_agi.py",
        "--tasks_path",
        str(tasks_path),
        "--mode",
        mode,
        "--max_new_tokens",
        str(MAX_NEW_TOKENS),
        "--grid_format",
        GRID_FORMAT,
        "--geometry_tta",
        GEOMETRY_TTA,
        "--program_parse_mode",
        PROGRAM_PARSE_MODE,
        "--selection_strategy",
        SELECTION_STRATEGY,
        "--dtype",
        DTYPE,
        "--adapter_dtype",
        ADAPTER_DTYPE,
        "--device",
        DEVICE,
        "--output_jsonl",
        path_for_cli(RUN_DIR / f"{label}_candidates.jsonl"),
        "--summary_json",
        path_for_cli(summary_json),
        "--summary_md",
        path_for_cli(RUN_DIR / f"{label}_summary.md"),
    ]
    if EXAMPLES_PER_DIFFICULTY is None:
        cmd += limit_args(LIMIT)
    cmd += difficulty_args(DIFFICULTY_BUCKETS, EXAMPLES_PER_DIFFICULTY)
    if mode != "base":
        if checkpoint is None:
            raise ValueError(f"checkpoint required for mode={mode}")
        cmd += [
            "--checkpoint",
            path_for_cli(checkpoint),
            "--max_loops",
            "4",
            "--num_candidates",
            "1",
        ]
    run(cmd, log_name=f"{label}.log")
    return read_json(summary_json)


def comparison_specs() -> list[tuple[str, str, str]]:
    return [
        ("phase1_start_vs_base", "base", "phase1_start"),
        ("recovered_vs_start", "phase1_start", "recovered"),
        ("recovered_vs_base", "base", "recovered"),
    ]


def compare_eval_summaries() -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for label, reference, candidate in comparison_specs():
        output_json = RUN_DIR / f"{label}_paired_comparison.json"
        output_md = RUN_DIR / f"{label}_paired_comparison.md"
        run(
            [
                sys.executable,
                "eval/compare_arc_agi_runs.py",
                "--reference_summary_json",
                path_for_cli(RUN_DIR / f"{reference}_summary.json"),
                "--candidate_summary_json",
                path_for_cli(RUN_DIR / f"{candidate}_summary.json"),
                "--reference_label",
                reference,
                "--candidate_label",
                candidate,
                "--output_json",
                path_for_cli(output_json),
                "--output_md",
                path_for_cli(output_md),
            ],
            log_name=f"{label}_paired_comparison.log",
        )
        comparisons[label] = read_json(output_json)
    return comparisons


def analyze_recovery_summaries() -> dict[str, Any]:
    output_json = RUN_DIR / "recovery_analysis.json"
    output_md = RUN_DIR / "recovery_analysis.md"
    run(
        [
            sys.executable,
            "eval/analyze_arc_agi_recovery.py",
            "--base_summary_json",
            path_for_cli(RUN_DIR / "base_summary.json"),
            "--start_summary_json",
            path_for_cli(RUN_DIR / "phase1_start_summary.json"),
            "--recovered_summary_json",
            path_for_cli(RUN_DIR / "recovered_summary.json"),
            "--output_json",
            path_for_cli(output_json),
            "--output_md",
            path_for_cli(output_md),
        ],
        log_name="recovery_analysis.log",
    )
    return read_json(output_json)


def metric_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    keys = ("first_exact", "selected_exact", "best_of_k_exact", "tasks_solved_best_of_k")
    return {f"{key}_delta": int(candidate.get(key, 0)) - int(reference.get(key, 0)) for key in keys}


def gap_closure(base: dict[str, Any], start: dict[str, Any], recovered: dict[str, Any]) -> dict[str, Any]:
    """Measure how much later training recovers from recurrent surgery.

    Positive ``initial_gap_to_base`` means the recurrent wrapper started below
    the dense base. ``closure_fraction`` is recovered-vs-start gain divided by
    that initial deficit. Values above 1.0 mean recovered recurrent surpassed
    the dense base on that metric.
    """

    keys = ("first_exact", "selected_exact", "best_of_k_exact", "tasks_solved_best_of_k")
    rows: dict[str, Any] = {}
    for key in keys:
        base_value = int(base.get(key, 0))
        start_value = int(start.get(key, 0))
        recovered_value = int(recovered.get(key, 0))
        initial_gap = base_value - start_value
        recovered_gain = recovered_value - start_value
        remaining_gap = base_value - recovered_value
        if initial_gap > 0:
            closure_fraction: float | None = recovered_gain / initial_gap
            if remaining_gap <= 0:
                status = "closed_or_surpassed"
            elif recovered_gain > 0:
                status = "partially_closed"
            elif recovered_gain == 0:
                status = "unchanged"
            else:
                status = "widened"
        else:
            closure_fraction = None
            status = "start_at_or_above_base"
        rows[key] = {
            "base": base_value,
            "phase1_start": start_value,
            "recovered": recovered_value,
            "initial_gap_to_base": initial_gap,
            "recovered_gain_from_start": recovered_gain,
            "remaining_gap_to_base": remaining_gap,
            "closure_fraction": closure_fraction,
            "status": status,
        }
    return rows


def backup_to_drive() -> None:
    if not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
        except Exception as exc:  # pragma: no cover - Colab only
            print(f"Drive mount skipped/failed: {exc}")
            return
    drive_root = Path(os.environ.get("DRIVE_BACKUP_DIR", "/content/drive/MyDrive/recurrent-qwen-svgd-artifacts"))
    if not drive_root.exists():
        print(f"Drive backup skipped; missing {drive_root}")
        return
    backup = drive_root / RUN_ID
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_DIR, backup / "run_dir", dirs_exist_ok=True)
    print(f"backed_up_to={backup}")


def git_commit_results() -> None:
    for pattern in ["*.json", "*.md", "*.log", "*.jsonl"]:
        run(["git", "add", "-f", str((RUN_DIR / pattern).relative_to(ROOT))], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No recovered benchmark outputs changed.")
        return
    run(["git", "commit", "-m", "Record recovered ARC-AGI benchmark comparison"])
    run(["git", "push", "origin", "main"], check=False)


def write_report(payload: dict[str, Any]) -> None:
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Recovered ARC-AGI Benchmark - {RUN_ID}",
        "",
        f"- ARC version/split: `{ARC_VERSION}` / `{ARC_SPLIT}`",
        f"- Limit: `{limit_label(LIMIT)}`",
        f"- Difficulty buckets: `{DIFFICULTY_BUCKETS or 'all'}`",
        f"- Examples per difficulty: `{EXAMPLES_PER_DIFFICULTY}`",
        f"- Tasks path: `{payload['metadata']['tasks_path']}`",
        f"- Phase1 start checkpoint: `{payload['metadata']['phase1_start_checkpoint']}`",
        f"- Recovered checkpoint: `{payload['metadata']['recovered_checkpoint']}`",
        f"- Geometry TTA: `{GEOMETRY_TTA}`",
        f"- Program parse mode: `{PROGRAM_PARSE_MODE}`",
        f"- Selection strategy: `{SELECTION_STRATEGY}`",
        "",
        "## Results",
        "",
        f"- Base: `{payload['base']['summary']}`",
        f"- Phase1 start: `{payload['phase1_start']['summary']}`",
        f"- Recovered: `{payload['recovered']['summary']}`",
        "",
        "## Deltas",
        "",
        f"- Start vs base: `{payload['deltas']['phase1_start_vs_base']}`",
        f"- Recovered vs start: `{payload['deltas']['recovered_vs_start']}`",
        f"- Recovered vs base: `{payload['deltas']['recovered_vs_base']}`",
        "",
        "## Surgical Gap Closure",
        "",
    ]
    for metric, row in payload["gap_closure"].items():
        fraction = row["closure_fraction"]
        fraction_text = "n/a" if fraction is None else f"{fraction:.2%}"
        lines.append(
            f"- `{metric}`: initial gap `{row['initial_gap_to_base']}`, "
            f"gain `{row['recovered_gain_from_start']}`, remaining gap `{row['remaining_gap_to_base']}`, "
            f"closure `{fraction_text}`, status `{row['status']}`"
        )
    lines += [
        "",
        "Interpretation: gap closure is the share of the dense-base regression introduced by the recurrent surgery "
        "that was recovered by later training. Values above 100% mean recovered recurrent surpassed the dense base.",
        "",
        "## Paired Evidence",
        "",
    ]
    for name, comparison in payload.get("paired_comparisons", {}).items():
        selected = comparison["metrics"]["selected_exact"]
        best = comparison["metrics"]["best_of_k_exact"]
        lines.append(
            f"- `{name}` selected delta `{selected['delta_exact']}` "
            f"({selected['wins']}/{selected['losses']}/{selected['ties']} W/L/T, p `{selected['sign_test_p_value']}`); "
            f"best-of-K delta `{best['delta_exact']}` "
            f"({best['wins']}/{best['losses']}/{best['ties']} W/L/T, p `{best['sign_test_p_value']}`)"
        )
    if payload.get("recovery_analysis"):
        lines += ["", "## Recovery Diagnosis", ""]
        for recommendation in payload["recovery_analysis"].get("recommendations", []):
            lines.append(f"- `{recommendation['area']}`: {recommendation['reason']}")
        lines.append("")
        lines.append("See `recovery_analysis.md` for family gaps and regression examples.")
    lines += [
        "",
        "This is a true ARC exact-grid comparison. Use larger limits only after the smoke limit is stable.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Run after curriculum recovery to compare base/start/recovered checkpoints on an ARC split.")
        return 0

    curriculum = maybe_curriculum_summary()
    tasks_path = resolve_tasks_path()
    start_ckpt = phase1_start_checkpoint_from_curriculum(curriculum)
    recovered_ckpt = recovered_checkpoint(curriculum)
    if not start_ckpt.exists():
        raise FileNotFoundError(start_ckpt)
    if not recovered_ckpt.exists():
        raise FileNotFoundError(recovered_ckpt)

    base = eval_arc("base", mode="base", tasks_path=tasks_path)
    phase1_start = eval_arc("phase1_start", mode="phase1", tasks_path=tasks_path, checkpoint=start_ckpt)
    recovered = eval_arc("recovered", mode="phase1", tasks_path=tasks_path, checkpoint=recovered_ckpt)
    paired_comparisons = compare_eval_summaries()
    recovery_analysis = analyze_recovery_summaries()

    payload = {
        "run_id": RUN_ID,
        "metadata": {
            **model_metadata(MODEL_NAME),
            "arc_version": ARC_VERSION,
            "arc_split": ARC_SPLIT,
            "limit": limit_label(LIMIT),
            "difficulty_buckets": DIFFICULTY_BUCKETS or None,
            "examples_per_difficulty": EXAMPLES_PER_DIFFICULTY,
            "tasks_path": str(tasks_path),
            "curriculum_summary": CURRICULUM_SUMMARY or None,
            "phase1_start_checkpoint": path_for_cli(start_ckpt),
            "recovered_checkpoint": path_for_cli(recovered_ckpt),
            "grid_format": GRID_FORMAT,
            "geometry_tta": GEOMETRY_TTA,
            "program_parse_mode": PROGRAM_PARSE_MODE,
            "selection_strategy": SELECTION_STRATEGY,
        },
        "base": base,
        "phase1_start": phase1_start,
        "recovered": recovered,
        "deltas": {
            "phase1_start_vs_base": metric_delta(phase1_start["summary"], base["summary"]),
            "recovered_vs_start": metric_delta(recovered["summary"], phase1_start["summary"]),
            "recovered_vs_base": metric_delta(recovered["summary"], base["summary"]),
        },
        "gap_closure": gap_closure(base["summary"], phase1_start["summary"], recovered["summary"]),
        "paired_comparisons": paired_comparisons,
        "recovery_analysis": recovery_analysis,
    }
    write_report(payload)
    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
