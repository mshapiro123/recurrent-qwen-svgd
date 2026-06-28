"""Colab cell: forced-depth ARC-Challenge diagnostic.

This is the next diagnostic after the depth-signal recovery benchmark. It does
not train. It forces recurrent MCQ scoring to use loop 1, loop 2, and loop 3
logits directly, so we can distinguish:

* router bottleneck: forced deeper loops help, but learned routing stays shallow;
* recurrence-quality bottleneck: forced deeper loops do not help.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from google.colab import drive, runtime, userdata

STAGE5_FORCED_DEPTH_DIAGNOSTIC_CELL_VERSION = "forced_depth_arc_v1"

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")
DISCONNECT_ON_FINISH = os.environ.get("STAGE5_FORCED_DEPTH_DISCONNECT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DRIVE_BACKUP = os.environ.get("STAGE5_FORCED_DEPTH_DRIVE_BACKUP", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
FORCE_DRIVE_REMOUNT = os.environ.get("FORCE_DRIVE_REMOUNT", "0").strip().lower() in {
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
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    proc = subprocess.CompletedProcess(cmd, process.wait())
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc


def require_cuda_runtime() -> None:
    requested_device = os.environ.get("DEVICE", "cuda").strip().lower()
    if not requested_device.startswith("cuda"):
        return
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError(
            "Forced-depth diagnostic requires an attached GPU runtime because DEVICE=cuda. "
            "Reconnect Colab with an L4/T4/A100/H100 runtime, or explicitly set DEVICE=cpu for a very slow CPU debug run."
        )
    run(["nvidia-smi"], check=False)


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


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def current_source_summary() -> str:
    explicit = os.environ.get("STAGE5_FORCED_DEPTH_SOURCE_SUMMARY", "").strip()
    if explicit:
        return explicit
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    assert pointer.exists(), "Missing config/stage5_current_source_summary.txt"
    value = pointer.read_text(encoding="utf-8").strip()
    assert value, "Current source summary pointer is empty."
    return value


def resolve_path(path: str | Path) -> Path:
    candidate = Path(str(path).replace("\\", "/"))
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def checkpoint_bearing_source_summary(source_summary: str) -> tuple[str, dict[str, Any]]:
    current_summary = source_summary
    seen: set[Path] = set()
    for _depth in range(8):
        current_path = resolve_path(current_summary)
        if current_path in seen:
            raise RuntimeError(f"Cycle while resolving forced-depth source summary: {current_summary}")
        seen.add(current_path)
        payload = read_json(current_path)
        kind = str(payload.get("kind") or "")
        next_summary = None
        if kind == "stage5_forced_depth_diagnostic":
            next_summary = payload.get("source_summary")
        elif kind == "stage5_mcq_scoring_policy":
            next_summary = payload.get("source_summary")
        elif kind == "stage5_mcq_debias_pair_assessment":
            source_summaries = payload.get("source_summaries") or {}
            if isinstance(source_summaries, dict):
                next_summary = source_summaries.get("arc_challenge") or source_summaries.get("arc_easy")
        elif kind == "stage5_mcq_debias_diagnostic":
            next_summary = payload.get("nested_source_summary") or payload.get("source_summary")

        if not next_summary:
            return path_for_cli(current_path), payload
        current_summary = str(next_summary)
    raise RuntimeError(f"Could not resolve checkpoint-bearing forced-depth source summary from {source_summary}")


def forced_depth_lora_rank(source_payload: dict[str, Any]) -> str:
    override = os.environ.get("STAGE5_FORCED_DEPTH_LORA_RANK", "").strip()
    if override:
        return override
    if source_payload.get("kind") == "stage5_unfreeze_recurrent_curriculum":
        return "0"
    return "8"


def forced_depth_lora_alpha(rank: str) -> str:
    override = os.environ.get("STAGE5_FORCED_DEPTH_LORA_ALPHA", "").strip()
    if override:
        return override
    return "16"


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def copy_run_to_drive(run_id: str):
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


def ensure_drive_for_checkpoint_restore() -> None:
    """Mount Drive in the notebook process so child runners can restore checkpoints.

    The benchmark runner executes in a subprocess. Calling ``drive.mount`` from
    that child can fail because the Colab kernel context is not attached there.
    Mount once here, then the child sees ``/content/drive/MyDrive`` as a normal
    filesystem path.
    """

    if Path("/content/drive/MyDrive").exists() and not FORCE_DRIVE_REMOUNT:
        print("Drive already mounted for checkpoint restore.", flush=True)
        return
    print("Mounting Drive for checkpoint restore.", flush=True)
    drive.mount("/content/drive", force_remount=FORCE_DRIVE_REMOUNT)


def compact_summary(run_id: str) -> dict:
    path = ROOT / "outputs" / "stage5" / run_id / "summary.json"
    if not path.exists():
        return {"run_id": run_id, "missing_summary": True}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for benchmark, score_targets in payload.get("paired_comparisons", {}).items():
        for score_target, aggregates in score_targets.items():
            for aggregate, row in aggregates.items():
                rows.append(
                    {
                        "benchmark": benchmark,
                        "score_target": score_target,
                        "aggregate": aggregate,
                        "base_correct": row.get("base_correct"),
                        "recurrent_correct": row.get("recurrent_correct"),
                        "paired_examples": row.get("paired_examples"),
                        "delta": row.get("correct_delta_recurrent_vs_base"),
                        "wins": row.get("wins"),
                        "losses": row.get("losses"),
                        "ties": row.get("ties"),
                        "sign_test_p": row.get("sign_test_p_value"),
                    }
                )
    return {
        "run_id": run_id,
        "status": payload.get("status"),
        "forced_loop_count": payload.get("recurrent_forced_loop_count"),
        "max_loops": payload.get("recurrent_max_loops"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "rows": rows,
        "hard_content_signal": payload.get("hard_content_signal", {}),
        "failures": payload.get("failures", []),
    }


def write_pointer(summary_path: Path) -> None:
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")


def write_sweep_summary(sweep_id: str, source_summary: str, run_ids: list[str]) -> Path:
    out_dir = ROOT / "outputs" / "stage5" / sweep_id
    out_dir.mkdir(parents=True, exist_ok=True)
    loops = parse_csv_ints(os.environ.get("STAGE5_FORCED_DEPTH_LOOPS", "1,2,3"))
    payload = {
        "kind": "stage5_forced_depth_diagnostic",
        "run_id": sweep_id,
        "cell_version": STAGE5_FORCED_DEPTH_DIAGNOSTIC_CELL_VERSION,
        "source_summary": source_summary,
        "loop_run_ids": run_ids,
        "loops": loops,
        "forward_max_loops": max(loops),
        "benchmarks": os.environ.get("STAGE5_FORCED_DEPTH_BENCHMARKS", "arc_challenge"),
        "score_targets": os.environ.get(
            "STAGE5_FORCED_DEPTH_SCORE_TARGETS",
            "content_question_only,cyclic_label_aggregated",
        ),
        "summaries": [compact_summary(run_id) for run_id in run_ids],
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Forced Depth Diagnostic - {sweep_id}",
        "",
        f"- Cell version: `{STAGE5_FORCED_DEPTH_DIAGNOSTIC_CELL_VERSION}`",
        f"- Source summary: `{source_summary}`",
        f"- Loops: `{payload['loops']}`",
        f"- Forward max loops: `{payload['forward_max_loops']}`",
        f"- Benchmarks: `{payload['benchmarks']}`",
        f"- Score targets: `{payload['score_targets']}`",
        "",
        "## Loop Runs",
        "",
    ]
    for summary in payload["summaries"]:
        lines.append(f"### {summary['run_id']}")
        lines.append(f"- Forced loop count: `{summary.get('forced_loop_count')}`")
        lines.append(f"- Status: `{summary.get('status')}`")
        for row in summary.get("rows", []):
            lines.append(
                "- "
                f"{row['benchmark']} `{row['score_target']}/{row['aggregate']}`: "
                f"base `{row['base_correct']}/{row['paired_examples']}`, "
                f"recurrent `{row['recurrent_correct']}/{row['paired_examples']}`, "
                f"delta `{row['delta']}`, W/L/T "
                f"`{row['wins']}/{row['losses']}/{row['ties']}`, p `{row['sign_test_p']}`"
            )
        if summary.get("failures"):
            lines.append(f"- failures: `{summary['failures']}`")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_pointer(summary_json)
    print((out_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    copy_run_to_drive(sweep_id)
    return out_dir


def run_depth_analysis(sweep_dir: Path, *, score_target: str, aggregate: str, suffix: str) -> Path:
    output_dir = sweep_dir / f"depth_analysis_{suffix}"
    run(
        [
            sys.executable,
            "eval/analyze_depth_sweep.py",
            "--sweep_summary",
            path_for_cli(sweep_dir / "summary.json"),
            "--score_target",
            score_target,
            "--aggregate",
            aggregate,
            "--output_dir",
            path_for_cli(output_dir),
        ],
        cwd=ROOT,
    )
    return output_dir


def publish_results(paths: list[Path]) -> None:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in paths:
        if path.exists():
            run(["git", "add", "-f", path_for_cli(path)], cwd=ROOT, check=False)
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if pointer.exists():
        run(["git", "add", "-f", path_for_cli(pointer)], cwd=ROOT, check=False)
    status = run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False)
    if status.returncode == 0:
        print("No forced-depth outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 forced depth diagnostic {paths[0].name} [skip ci]"], cwd=ROOT)
    push = run(["git", "push", "origin", "main"], cwd=ROOT, check=False)
    if push.returncode == 0:
        return
    print("Initial push failed; rebasing once and retrying.", flush=True)
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT)
    run(["git", "push", "origin", "main"], cwd=ROOT)


def disconnect(reason: str) -> None:
    if not DISCONNECT_ON_FINISH:
        print(f"Leaving Colab runtime connected: {reason}", flush=True)
        return
    try:
        print(f"Disconnecting Colab runtime to conserve credits: {reason}", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"Runtime disconnect skipped: {exc}", flush=True)


try:
    require_cuda_runtime()
    ensure_drive_for_checkpoint_restore()
    if not DRIVE_BACKUP:
        print("Drive backup disabled; using GitHub as primary artifact store for outputs.", flush=True)
    sync_repo()
    os.chdir(ROOT)
    print(f"STAGE5_FORCED_DEPTH_DIAGNOSTIC_CELL_VERSION={STAGE5_FORCED_DEPTH_DIAGNOSTIC_CELL_VERSION}", flush=True)
    run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_eval_mcq_loop_diagnostics.py",
            "tests/test_stage5_benchmark_suite.py",
            "tests/test_analyze_depth_sweep.py",
            "tests/test_stage5_benchmark_assessment.py",
        ],
        cwd=ROOT,
    )

    requested_source_summary = current_source_summary()
    source_summary, source_payload = checkpoint_bearing_source_summary(requested_source_summary)
    lora_rank = forced_depth_lora_rank(source_payload)
    lora_alpha = forced_depth_lora_alpha(lora_rank)
    loops = parse_csv_ints(os.environ.get("STAGE5_FORCED_DEPTH_LOOPS", "1,2,3"))
    forward_max_loops = max(loops)
    sweep_id = os.environ.get("STAGE5_FORCED_DEPTH_RUN_ID") or time.strftime(
        "stage5_forced_depth_arc_challenge_loop123_%Y%m%d_%H%M%S"
    )
    run_ids = []
    result_paths: list[Path] = []
    print("forced_depth_requested_source_summary:", requested_source_summary, flush=True)
    print("forced_depth_source_summary:", source_summary, flush=True)
    print("forced_depth_sweep_id:", sweep_id, flush=True)
    print("forced_depth_loops:", loops, flush=True)
    print("forced_depth_forward_max_loops:", forward_max_loops, flush=True)
    print(f"forced_depth_lora_rank={lora_rank} forced_depth_lora_alpha={lora_alpha}", flush=True)

    for forced_loop in loops:
        run_id = f"{sweep_id}_loop{forced_loop}"
        run_ids.append(run_id)
        env = os.environ.copy()
        env.update(
            {
                "STAGE5_BENCHMARK_SUITE_RUN_ID": run_id,
                "STAGE5_BENCHMARK_SOURCE_SUMMARY": source_summary,
                "STAGE5_BENCHMARKS": os.environ.get("STAGE5_FORCED_DEPTH_BENCHMARKS", "arc_challenge"),
                "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT": os.environ.get("STAGE5_FORCED_DEPTH_ARC_CHALLENGE_LIMIT", "256"),
                "STAGE5_BENCHMARK_ARC_CHALLENGE_OFFSET": os.environ.get("STAGE5_FORCED_DEPTH_ARC_CHALLENGE_OFFSET", "0"),
                "STAGE5_BENCHMARK_ARC_EASY_LIMIT": os.environ.get("STAGE5_FORCED_DEPTH_ARC_EASY_LIMIT", "128"),
                "STAGE5_BENCHMARK_ARC_EASY_OFFSET": os.environ.get("STAGE5_FORCED_DEPTH_ARC_EASY_OFFSET", "0"),
                "STAGE5_BENCHMARK_MAX_LOOPS": str(forward_max_loops),
                "STAGE5_BENCHMARK_FORCED_LOOP_COUNT": str(forced_loop),
                "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL": "0",
                "STAGE5_BENCHMARK_NUM_TRAJECTORIES": "1",
                "STAGE5_BENCHMARK_SCORE_TARGETS": os.environ.get(
                    "STAGE5_FORCED_DEPTH_SCORE_TARGETS",
                    "content_question_only,cyclic_label_aggregated",
                ),
                "STAGE5_BENCHMARK_AGGREGATES": "mean",
                "STAGE5_BENCHMARK_INCLUDE_LOOP_DIAGNOSTICS": "1",
                "STAGE5_BENCHMARK_LORA_RANK": lora_rank,
                "STAGE5_BENCHMARK_LORA_ALPHA": lora_alpha,
                "STAGE5_BENCHMARK_PUSH": "0",
                "DTYPE": os.environ.get("DTYPE", "bfloat16"),
                "ADAPTER_DTYPE": os.environ.get("ADAPTER_DTYPE", "float32"),
                "DEVICE": os.environ.get("DEVICE", "cuda"),
            }
        )
        print(f"\n===== forced_loop_count={forced_loop} =====", flush=True)
        run([sys.executable, "colab/run_stage5_benchmark_suite.py"], cwd=ROOT, env=env)
        copy_run_to_drive(run_id)
        result_paths.append(ROOT / "outputs" / "stage5" / run_id)

    sweep_dir = write_sweep_summary(sweep_id, source_summary, run_ids)
    content_analysis = run_depth_analysis(
        sweep_dir,
        score_target="content_question_only",
        aggregate="mean",
        suffix="content_question_only",
    )
    cyclic_analysis = run_depth_analysis(
        sweep_dir,
        score_target="cyclic_label_aggregated",
        aggregate="permutation_mean",
        suffix="cyclic_label_aggregated",
    )
    result_paths.extend([sweep_dir, content_analysis, cyclic_analysis])
    copy_run_to_drive(sweep_id)
    publish_results(result_paths)
    disconnect("forced-depth diagnostic finished")
except Exception:
    disconnect("forced-depth diagnostic errored")
    raise
