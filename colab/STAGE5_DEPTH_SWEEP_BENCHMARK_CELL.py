"""Colab cell: run a low-cost loop-depth sweep on ARC MCQ slices.

This is meant for T4/L4. It runs the existing benchmark suite repeatedly with
``max_loops`` set to 1, 2, 3, and 4, using one final GitHub publish step after
all runs finish. Each run is also copied to Drive before publish, so a GitHub
push failure does not lose GPU work.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, runtime, userdata

STAGE5_DEPTH_SWEEP_BENCHMARK_CELL_VERSION = "depth_sweep_arc_v1"

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")
SOURCE_SUMMARY = "outputs/stage5/stage5_direct_preservation_loop1_20260622_232720/summary.json"
CHECKPOINT = "outputs/stage5/stage5_arc_agi_next_action_20260622_170927_plan_curriculum_sft/phase1/phase1_step_150.pt"
DRIVE_BACKUP = os.environ.get("STAGE5_DEPTH_SWEEP_DRIVE_BACKUP", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DISCONNECT_ON_FINISH = os.environ.get("STAGE5_DEPTH_SWEEP_DISCONNECT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}


def secret(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None


GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
HF_TOKEN = secret("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


def run(cmd, *, cwd=None, env=None, check=True):
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print("$", printable, flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc


def sync_repo():
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url], cwd=ROOT)
        run(["git", "fetch", "origin", "main"], cwd=ROOT)
        run(["git", "checkout", "main"], cwd=ROOT)
        run(["git", "reset", "--hard", "origin/main"], cwd=ROOT)
    else:
        run(["git", "clone", clone_url, str(ROOT)])
    run(["git", "config", "user.email", "colab-runner@local"], cwd=ROOT)
    run(["git", "config", "user.name", "Colab Runner"], cwd=ROOT)


def parse_csv_ints(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def copy_run_to_drive(run_id):
    if not DRIVE_BACKUP:
        print(f"drive_backup_skipped={run_id}", flush=True)
        return None
    run_dir = ROOT / "outputs" / "stage5" / run_id
    if not run_dir.exists():
        return None
    drive_dst = DRIVE_ARTIFACT_ROOT / "stage5" / run_id
    drive_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, drive_dst, dirs_exist_ok=True)
    print(f"backed_up_run_dir={run_dir} -> {drive_dst}", flush=True)
    return drive_dst


def compact_summary(run_id):
    path = ROOT / "outputs" / "stage5" / run_id / "summary.json"
    if not path.exists():
        return {"run_id": run_id, "missing_summary": True}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for benchmark, score_targets in payload.get("comparisons", {}).items():
        for score_target, aggregates in score_targets.items():
            for aggregate, row in aggregates.items():
                paired = (
                    payload.get("paired_comparisons", {})
                    .get(benchmark, {})
                    .get(score_target, {})
                    .get(aggregate, {})
                )
                rows.append(
                    {
                        "benchmark": benchmark,
                        "score_target": score_target,
                        "aggregate": aggregate,
                        "base_correct": row.get("base", {}).get("correct"),
                        "base_total": row.get("base", {}).get("total"),
                        "recurrent_correct": row.get("recurrent", {}).get("correct"),
                        "recurrent_total": row.get("recurrent", {}).get("total"),
                        "delta": row.get("correct_delta_recurrent_vs_base"),
                        "paired_wins": paired.get("wins"),
                        "paired_losses": paired.get("losses"),
                        "paired_ties": paired.get("ties"),
                        "sign_test_p": paired.get("sign_test_p_value"),
                    }
                )
    return {
        "run_id": run_id,
        "status": payload.get("status"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "max_loops": payload.get("recurrent_max_loops")
        or payload.get("config", {}).get("max_loops"),
        "rows": rows,
        "failures": payload.get("failures", []),
    }


def write_sweep_summary(sweep_id, run_ids):
    out_dir = ROOT / "outputs" / "stage5" / sweep_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "stage5_depth_sweep_benchmark",
        "run_id": sweep_id,
        "source_summary": SOURCE_SUMMARY,
        "checkpoint": CHECKPOINT,
        "loop_run_ids": run_ids,
        "summaries": [compact_summary(run_id) for run_id in run_ids],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [f"# Stage 5 Depth Sweep - {sweep_id}", ""]
    for summary in payload["summaries"]:
        lines.append(f"## {summary['run_id']}")
        for row in summary.get("rows", []):
            lines.append(
                "- "
                f"{row['benchmark']} `{row['score_target']}/{row['aggregate']}`: "
                f"base {row['base_correct']}/{row['base_total']}, "
                f"recurrent {row['recurrent_correct']}/{row['recurrent_total']}, "
                f"delta {row['delta']}, W/L/T "
                f"{row['paired_wins']}/{row['paired_losses']}/{row['paired_ties']}, "
                f"p {row['sign_test_p']}"
            )
        if summary.get("failures"):
            lines.append(f"- failures: `{summary['failures']}`")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((out_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    copy_run_to_drive(sweep_id)
    return out_dir


def run_depth_analysis(sweep_dir):
    output_dir = sweep_dir / "depth_analysis"
    run(
        [
            sys.executable,
            "eval/analyze_depth_sweep.py",
            "--sweep_summary",
            (sweep_dir / "summary.json").relative_to(ROOT).as_posix(),
            "--output_dir",
            output_dir.relative_to(ROOT).as_posix(),
        ],
        cwd=ROOT,
    )
    copy_run_to_drive(sweep_dir.name)
    return output_dir


def publish_results(paths):
    run(["git", "pull", "--rebase", "origin", "main"], cwd=ROOT, check=False)
    for path in paths:
        if path.exists():
            run(["git", "add", "-f", path.relative_to(ROOT).as_posix()], cwd=ROOT, check=False)
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if pointer.exists():
        run(["git", "add", "-f", pointer.relative_to(ROOT).as_posix()], cwd=ROOT, check=False)
    status = run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False)
    if status.returncode == 0:
        print("No depth sweep outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 loop depth sweep {paths[-1].name}"], cwd=ROOT)
    run(["git", "push", "origin", "main"], cwd=ROOT, check=False)


def disconnect(reason):
    if not DISCONNECT_ON_FINISH:
        print(f"Leaving Colab runtime connected: {reason}", flush=True)
        return
    try:
        print(f"Disconnecting Colab runtime to conserve credits: {reason}", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"Runtime disconnect skipped: {exc}", flush=True)


try:
    if DRIVE_BACKUP:
        drive.mount("/content/drive", force_remount=False)
    else:
        print("Drive backup disabled; using GitHub as primary artifact store.", flush=True)
    sync_repo()
    os.chdir(ROOT)
    run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)

    loops = parse_csv_ints(os.environ.get("STAGE5_DEPTH_SWEEP_LOOPS", "1,2,3,4"))
    sweep_id = os.environ.get("STAGE5_DEPTH_SWEEP_RUN_ID") or time.strftime(
        "stage5_depth_sweep_arc_loop1234_%Y%m%d_%H%M%S"
    )
    run_ids = []
    result_paths = []
    print("depth_sweep_id:", sweep_id, flush=True)
    print("loops:", loops, flush=True)

    for max_loops in loops:
        run_id = f"{sweep_id}_loop{max_loops}"
        run_ids.append(run_id)
        env = os.environ.copy()
        env.update(
            {
                "STAGE5_BENCHMARK_SUITE_RUN_ID": run_id,
                "STAGE5_BENCHMARK_SOURCE_SUMMARY": SOURCE_SUMMARY,
                "STAGE5_BENCHMARK_CHECKPOINT": CHECKPOINT,
                "STAGE5_BENCHMARKS": os.environ.get("STAGE5_DEPTH_SWEEP_BENCHMARKS", "arc_easy,arc_challenge"),
                "STAGE5_BENCHMARK_ARC_EASY_LIMIT": os.environ.get("STAGE5_DEPTH_SWEEP_ARC_EASY_LIMIT", "256"),
                "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT": os.environ.get(
                    "STAGE5_DEPTH_SWEEP_ARC_CHALLENGE_LIMIT", "256"
                ),
                "STAGE5_BENCHMARK_ARC_EASY_OFFSET": os.environ.get("STAGE5_DEPTH_SWEEP_ARC_EASY_OFFSET", "0"),
                "STAGE5_BENCHMARK_ARC_CHALLENGE_OFFSET": os.environ.get(
                    "STAGE5_DEPTH_SWEEP_ARC_CHALLENGE_OFFSET", "0"
                ),
                "STAGE5_BENCHMARK_MAX_LOOPS": str(max_loops),
                "STAGE5_BENCHMARK_NUM_TRAJECTORIES": "1",
                "STAGE5_BENCHMARK_SCORE_TARGETS": os.environ.get("STAGE5_DEPTH_SWEEP_SCORE_TARGETS", "label"),
                "STAGE5_BENCHMARK_AGGREGATES": "mean",
                "STAGE5_BENCHMARK_INCLUDE_LOOP_DIAGNOSTICS": "1",
                "STAGE5_BENCHMARK_PUSH": "0",
                "DTYPE": os.environ.get("DTYPE", "bfloat16"),
                "ADAPTER_DTYPE": os.environ.get("ADAPTER_DTYPE", "float32"),
                "DEVICE": os.environ.get("DEVICE", "cuda"),
            }
        )
        print(f"\n===== max_loops={max_loops} =====", flush=True)
        run([sys.executable, "colab/run_stage5_benchmark_suite.py"], cwd=ROOT, env=env)
        copy_run_to_drive(run_id)
        result_paths.append(ROOT / "outputs" / "stage5" / run_id)

    sweep_dir = write_sweep_summary(sweep_id, run_ids)
    analysis_dir = run_depth_analysis(sweep_dir)
    result_paths.append(sweep_dir)
    result_paths.append(analysis_dir)
    publish_results(result_paths)
    disconnect("depth sweep finished")
except Exception:
    disconnect("depth sweep errored")
    raise
