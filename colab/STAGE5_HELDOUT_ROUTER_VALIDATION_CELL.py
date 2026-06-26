"""Colab cell: held-out validation for forced-depth router transfer.

This is a no-training reality test. It uses the previous forced-depth
ARC-Challenge run as the discovery set, runs forced loop 1/2/3 on held-out
benchmark slices, then evaluates whether the selector chosen on discovery
transfers to held-out data.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, runtime, userdata

STAGE5_HELDOUT_ROUTER_VALIDATION_CELL_VERSION = "heldout_router_validation_v1"
# Marker for bootstrap/tests: router_transfer_content_question_only.

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")
DRIVE_BACKUP = os.environ.get("STAGE5_HELDOUT_ROUTER_DRIVE_BACKUP", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DISCONNECT_ON_FINISH = os.environ.get("STAGE5_HELDOUT_ROUTER_DISCONNECT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
RUN_LATENT_CRITICALITY = os.environ.get("STAGE5_HELDOUT_ROUTER_RUN_LATENT_CRITICALITY", "1").strip().lower() in {
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


def ensure_drive_for_checkpoint_restore():
    if Path("/content/drive/MyDrive").exists() and not FORCE_DRIVE_REMOUNT:
        print("Drive already mounted for checkpoint restore.", flush=True)
        return
    print("Mounting Drive for checkpoint restore.", flush=True)
    drive.mount("/content/drive", force_remount=FORCE_DRIVE_REMOUNT)


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def current_source_summary() -> str:
    explicit = os.environ.get("STAGE5_HELDOUT_ROUTER_DISCOVERY_SUMMARY", "").strip()
    if explicit:
        return explicit
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    assert pointer.exists(), "Missing config/stage5_current_source_summary.txt"
    value = pointer.read_text(encoding="utf-8").strip()
    assert value, "Current source summary pointer is empty."
    return value


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
        "failures": payload.get("failures", []),
    }


def write_pointer(summary_path: Path) -> None:
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(path_for_cli(summary_path) + "\n", encoding="utf-8")


def write_sweep_summary(sweep_id: str, discovery_summary: str, run_ids: list[str], loops: list[int]) -> Path:
    out_dir = ROOT / "outputs" / "stage5" / sweep_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "stage5_heldout_router_validation_sweep",
        "run_id": sweep_id,
        "cell_version": STAGE5_HELDOUT_ROUTER_VALIDATION_CELL_VERSION,
        "discovery_sweep_summary": discovery_summary,
        "loop_run_ids": run_ids,
        "loops": loops,
        "forward_max_loops": max(loops),
        "benchmarks": os.environ.get(
            "STAGE5_HELDOUT_ROUTER_BENCHMARKS",
            "arc_easy,arc_challenge,open_hard_arc_challenge",
        ),
        "score_targets": os.environ.get(
            "STAGE5_HELDOUT_ROUTER_SCORE_TARGETS",
            "content_question_only,cyclic_label_aggregated",
        ),
        "summaries": [compact_summary(run_id) for run_id in run_ids],
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Held-Out Router Validation Sweep - {sweep_id}",
        "",
        f"- Cell version: `{STAGE5_HELDOUT_ROUTER_VALIDATION_CELL_VERSION}`",
        f"- Discovery sweep: `{discovery_summary}`",
        f"- Loops: `{loops}`",
        f"- Forward max loops: `{max(loops)}`",
        f"- Benchmarks: `{payload['benchmarks']}`",
        f"- Score targets: `{payload['score_targets']}`",
        "",
    ]
    for summary in payload["summaries"]:
        lines.append(f"## {summary['run_id']}")
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


def run_transfer(
    sweep_dir: Path,
    *,
    discovery_summary: str,
    score_target: str,
    aggregate: str,
    suffix: str,
) -> Path:
    output_dir = sweep_dir / f"router_transfer_{suffix}"
    run(
        [
            sys.executable,
            "eval/evaluate_depth_router_transfer.py",
            "--discovery_sweep_summary",
            discovery_summary,
            "--heldout_sweep_summary",
            path_for_cli(sweep_dir / "summary.json"),
            "--score_target",
            score_target,
            "--aggregate",
            aggregate,
            "--output_dir",
            path_for_cli(output_dir),
            "--min_oracle_gap_capture",
            os.environ.get("STAGE5_HELDOUT_ROUTER_MIN_ORACLE_CAPTURE", "0.2"),
        ],
        cwd=ROOT,
    )
    return output_dir


def write_final_review(sweep_dir: Path, transfer_dirs: list[Path]) -> Path:
    payload = {
        "kind": "stage5_heldout_router_validation",
        "run_id": sweep_dir.name,
        "sweep_summary": path_for_cli(sweep_dir / "summary.json"),
        "transfer_summaries": [path_for_cli(path / "summary.json") for path in transfer_dirs],
        "primary_transfer": json.loads((transfer_dirs[0] / "summary.json").read_text(encoding="utf-8")),
    }
    final_json = sweep_dir / "router_validation_summary.json"
    final_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    primary = payload["primary_transfer"]
    lines = [
        f"# Stage 5 Held-Out Router Validation - {sweep_dir.name}",
        "",
        f"- Primary gate status: `{primary['gate_status']}`",
        f"- Recommended next: `{primary['recommended_next']}`",
        f"- Selector: `{primary['selector_signature']}`",
        f"- Mean delta vs loop1: `{primary['transfer_summary']['mean_delta_vs_loop1']}`",
        f"- Mean delta vs base: `{primary['transfer_summary']['mean_delta_vs_base']}`",
        f"- Mean oracle gap capture: `{primary['transfer_summary']['mean_oracle_gap_capture']}`",
        "",
    ]
    (sweep_dir / "router_validation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((sweep_dir / "router_validation_summary.md").read_text(encoding="utf-8"), flush=True)
    write_pointer(final_json)
    return final_json


def run_latent_criticality(sweep_dir: Path, *, discovery_summary: str) -> Path | None:
    if not RUN_LATENT_CRITICALITY:
        print("latent_criticality_skipped=disabled", flush=True)
        return None
    output_dir = sweep_dir / "latent_criticality"
    run(
        [
            sys.executable,
            "eval/eval_latent_criticality.py",
            "--discovery_sweep_summary",
            discovery_summary,
            "--heldout_sweep_summary",
            path_for_cli(sweep_dir / "summary.json"),
            "--output_dir",
            path_for_cli(output_dir),
            "--max_examples_per_benchmark",
            os.environ.get("STAGE5_LATENT_CRITICALITY_MAX_EXAMPLES_PER_BENCHMARK", "64"),
            "--jacobian_examples_per_benchmark",
            os.environ.get("STAGE5_LATENT_CRITICALITY_JACOBIAN_EXAMPLES_PER_BENCHMARK", "8"),
            "--jacobian_random_probes",
            os.environ.get("STAGE5_LATENT_CRITICALITY_JACOBIAN_RANDOM_PROBES", "1"),
            "--jacobian_epsilon",
            os.environ.get("STAGE5_LATENT_CRITICALITY_JACOBIAN_EPSILON", "0.02"),
            "--dtype",
            os.environ.get("DTYPE", "bfloat16"),
            "--adapter_dtype",
            os.environ.get("ADAPTER_DTYPE", "float32"),
            "--device",
            os.environ.get("DEVICE", "cuda"),
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
        print("No held-out router validation outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 held-out router validation {paths[0].name} [skip ci]"], cwd=ROOT)
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
    ensure_drive_for_checkpoint_restore()
    if not DRIVE_BACKUP:
        print("Drive backup disabled; using GitHub as primary artifact store for outputs.", flush=True)
    sync_repo()
    os.chdir(ROOT)
    print(f"STAGE5_HELDOUT_ROUTER_VALIDATION_CELL_VERSION={STAGE5_HELDOUT_ROUTER_VALIDATION_CELL_VERSION}", flush=True)
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
            "tests/test_evaluate_depth_router_transfer.py",
            "tests/test_stage5_notebooks.py::test_current_bootstrap_exposes_heldout_router_validation_target",
        ],
        cwd=ROOT,
    )

    discovery_summary = current_source_summary()
    loops = parse_csv_ints(os.environ.get("STAGE5_HELDOUT_ROUTER_LOOPS", "1,2,3"))
    forward_max_loops = max(loops)
    sweep_id = os.environ.get("STAGE5_HELDOUT_ROUTER_RUN_ID") or time.strftime(
        "stage5_heldout_router_validation_%Y%m%d_%H%M%S"
    )
    run_ids = []
    result_paths: list[Path] = []
    print("heldout_router_discovery_summary:", discovery_summary, flush=True)
    print("heldout_router_sweep_id:", sweep_id, flush=True)
    print("heldout_router_loops:", loops, flush=True)
    print("heldout_router_forward_max_loops:", forward_max_loops, flush=True)

    for forced_loop in loops:
        run_id = f"{sweep_id}_loop{forced_loop}"
        run_ids.append(run_id)
        env = os.environ.copy()
        env.update(
            {
                "STAGE5_BENCHMARK_SUITE_RUN_ID": run_id,
                "STAGE5_BENCHMARK_SOURCE_SUMMARY": discovery_summary,
                "STAGE5_BENCHMARKS": os.environ.get(
                    "STAGE5_HELDOUT_ROUTER_BENCHMARKS",
                    "arc_easy,arc_challenge,open_hard_arc_challenge",
                ),
                "STAGE5_BENCHMARK_ARC_EASY_LIMIT": os.environ.get("STAGE5_HELDOUT_ROUTER_ARC_EASY_LIMIT", "128"),
                "STAGE5_BENCHMARK_ARC_EASY_OFFSET": os.environ.get("STAGE5_HELDOUT_ROUTER_ARC_EASY_OFFSET", "256"),
                "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT": os.environ.get(
                    "STAGE5_HELDOUT_ROUTER_ARC_CHALLENGE_LIMIT",
                    "128",
                ),
                "STAGE5_BENCHMARK_ARC_CHALLENGE_OFFSET": os.environ.get(
                    "STAGE5_HELDOUT_ROUTER_ARC_CHALLENGE_OFFSET",
                    "256",
                ),
                "STAGE5_BENCHMARK_OPEN_HARD_ARC_CHALLENGE_LIMIT": os.environ.get(
                    "STAGE5_HELDOUT_ROUTER_OPEN_HARD_ARC_CHALLENGE_LIMIT",
                    "128",
                ),
                "STAGE5_BENCHMARK_OPEN_HARD_ARC_CHALLENGE_SPLIT": os.environ.get(
                    "STAGE5_HELDOUT_ROUTER_OPEN_HARD_ARC_CHALLENGE_SPLIT",
                    "test",
                ),
                "STAGE5_BENCHMARK_OPEN_HARD_ARC_CHALLENGE_OFFSET": os.environ.get(
                    "STAGE5_HELDOUT_ROUTER_OPEN_HARD_ARC_CHALLENGE_OFFSET",
                    "0",
                ),
                "STAGE5_BENCHMARK_MAX_LOOPS": str(forward_max_loops),
                "STAGE5_BENCHMARK_FORCED_LOOP_COUNT": str(forced_loop),
                "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL": "0",
                "STAGE5_BENCHMARK_NUM_TRAJECTORIES": "1",
                "STAGE5_BENCHMARK_SCORE_TARGETS": os.environ.get(
                    "STAGE5_HELDOUT_ROUTER_SCORE_TARGETS",
                    "content_question_only,cyclic_label_aggregated",
                ),
                "STAGE5_BENCHMARK_AGGREGATES": "mean",
                "STAGE5_BENCHMARK_CONTINUE_ON_FAILURE": "0",
                "STAGE5_BENCHMARK_INCLUDE_LOOP_DIAGNOSTICS": "1",
                "STAGE5_BENCHMARK_PUSH": "0",
                "DTYPE": os.environ.get("DTYPE", "bfloat16"),
                "ADAPTER_DTYPE": os.environ.get("ADAPTER_DTYPE", "float32"),
                "DEVICE": os.environ.get("DEVICE", "cuda"),
            }
        )
        print(f"\n===== heldout forced_loop_count={forced_loop} =====", flush=True)
        run([sys.executable, "colab/run_stage5_benchmark_suite.py"], cwd=ROOT, env=env)
        copy_run_to_drive(run_id)
        result_paths.append(ROOT / "outputs" / "stage5" / run_id)

    sweep_dir = write_sweep_summary(sweep_id, discovery_summary, run_ids, loops)
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
    content_transfer = run_transfer(
        sweep_dir,
        discovery_summary=discovery_summary,
        score_target="content_question_only",
        aggregate="mean",
        suffix="content_question_only",
    )
    cyclic_transfer = run_transfer(
        sweep_dir,
        discovery_summary=discovery_summary,
        score_target="cyclic_label_aggregated",
        aggregate="permutation_mean",
        suffix="cyclic_label_aggregated",
    )
    final_summary = write_final_review(sweep_dir, [content_transfer, cyclic_transfer])
    criticality_dir = run_latent_criticality(sweep_dir, discovery_summary=discovery_summary)
    publish_paths = [
        sweep_dir,
        content_analysis,
        cyclic_analysis,
        content_transfer,
        cyclic_transfer,
        final_summary,
        *( [criticality_dir] if criticality_dir is not None else [] ),
        *result_paths,
    ]
    publish_results(publish_paths)
    disconnect("held-out router validation finished")
except Exception:
    disconnect("held-out router validation errored")
    raise
