"""Colab cell: final deterministic-recurrence gate.

This resolves the deterministic loop-depth line in two steps:

1. Run cyclic-label rescue detectability on the powered discovery sweep.
2. Only if that clears the permutation null, run a pooled forced-depth sweep
   and k-fold selector validation on cyclic labels.

Particles/SVGD stay paused. This is a selector-transfer resolution test, not a
training run.
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

from google.colab import drive, runtime, userdata


STAGE5_DETERMINISTIC_FINAL_GATE_CELL_VERSION = "deterministic_final_gate_v2_nested_selector"
# Bootstrap safety marker: nested_outer_fold_train_only.
# Expected terminal statuses include closed_at_detectability_gate,
# selector_transfer_passed, selector_transfer_failed, and
# selector_transfer_needs_review.

REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DISCONNECT_ON_FINISH = os.environ.get("STAGE5_FINAL_GATE_DISCONNECT", "0").strip().lower() in {
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


def secret(*names: str) -> str | None:
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


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True):
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
    completed = subprocess.CompletedProcess(cmd, process.wait())
    if check and completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    return completed


def sync_repo() -> None:
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def require_cuda_runtime() -> None:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError(
            "Final deterministic gate Step 2 requires an attached GPU runtime. "
            "Use L4/T4/A100/H100; L4 is sufficient."
        )
    run(["nvidia-smi"], check=False)


def disconnect(reason: str) -> None:
    if not DISCONNECT_ON_FINISH:
        print(f"Leaving Colab runtime connected: {reason}", flush=True)
        return
    try:
        print(f"Disconnecting Colab runtime: {reason}", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"Runtime disconnect skipped: {exc}", flush=True)


def mount_drive() -> None:
    print("Mounting Drive for checkpoint restore.", flush=True)
    drive.mount("/content/drive", force_remount=FORCE_DRIVE_REMOUNT)


def compact_loop_summary(run_id: str) -> dict[str, Any]:
    path = ROOT / "outputs" / "stage5" / run_id / "summary.json"
    payload = read_json(path)
    rows: list[dict[str, Any]] = []
    for comparison in payload.get("comparisons", []):
        rows.append(
            {
                "benchmark": comparison.get("benchmark"),
                "score_target": comparison.get("score_target"),
                "aggregate": comparison.get("aggregate"),
                "paired_examples": comparison.get("paired_examples"),
                "base_correct": comparison.get("base_correct"),
                "recurrent_correct": comparison.get("recurrent_correct"),
                "delta": comparison.get("delta"),
                "wins": comparison.get("wins"),
                "losses": comparison.get("losses"),
                "ties": comparison.get("ties"),
                "sign_test_p": comparison.get("sign_test_p"),
            }
        )
    return {
        "run_id": run_id,
        "status": payload.get("status"),
        "forced_loop_count": payload.get("recurrent_forced_loop_count"),
        "rows": rows,
        "failures": payload.get("failures", []),
    }


def write_forced_depth_sweep_summary(
    *,
    sweep_id: str,
    source_summary: str,
    loops: list[int],
    run_ids: list[str],
) -> Path:
    out_dir = ROOT / "outputs" / "stage5" / sweep_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "stage5_forced_depth_diagnostic",
        "run_id": sweep_id,
        "cell_version": STAGE5_DETERMINISTIC_FINAL_GATE_CELL_VERSION,
        "source_summary": source_summary,
        "loop_run_ids": run_ids,
        "loops": loops,
        "forward_max_loops": max(loops),
        "benchmarks": os.environ.get(
            "STAGE5_FINAL_GATE_BENCHMARKS",
            "arc_easy,arc_challenge,open_hard_arc_challenge",
        ),
        "score_targets": "cyclic_label_aggregated",
        "summaries": [compact_loop_summary(run_id) for run_id in run_ids],
    }
    write_json(out_dir / "summary.json", payload)
    lines = [
        f"# Final-Gate Forced Depth Sweep - {sweep_id}",
        "",
        f"- Source summary: `{source_summary}`",
        f"- Loops: `{loops}`",
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
                f"delta `{row['delta']}`, W/L/T `{row['wins']}/{row['losses']}/{row['ties']}`"
            )
        if summary.get("failures"):
            lines.append(f"- failures: `{summary['failures']}`")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((out_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    return out_dir


def publish(paths: list[Path]) -> None:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in paths:
        if path.exists():
            run(["git", "add", "-f", path_for_cli(path)], cwd=ROOT, check=False)
    pointer = ROOT / "config" / "stage5_current_source_summary.txt"
    if pointer.exists():
        run(["git", "add", "-f", path_for_cli(pointer)], cwd=ROOT, check=False)
    status = run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False)
    if status.returncode == 0:
        print("No final-gate outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 deterministic final gate {paths[0].name} [skip ci]"], cwd=ROOT)
    push = run(["git", "push", "origin", "main"], cwd=ROOT, check=False)
    if push.returncode == 0:
        return
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT)
    run(["git", "push", "origin", "main"], cwd=ROOT)


def write_final_summary(
    *,
    run_dir: Path,
    detectability_dir: Path,
    status: str,
    forced_depth_sweep: Path | None = None,
    kfold_dir: Path | None = None,
) -> Path:
    payload = {
        "kind": "stage5_deterministic_final_gate",
        "run_id": run_dir.name,
        "status": status,
        "detectability_summary": path_for_cli(detectability_dir / "summary.json"),
        "forced_depth_sweep_summary": path_for_cli(forced_depth_sweep / "summary.json") if forced_depth_sweep else None,
        "kfold_summary": path_for_cli(kfold_dir / "summary.json") if kfold_dir else None,
    }
    if (detectability_dir / "summary.json").exists():
        payload["detectability"] = read_json(detectability_dir / "summary.json")
    if kfold_dir and (kfold_dir / "summary.json").exists():
        payload["kfold"] = read_json(kfold_dir / "summary.json")
    write_json(run_dir / "summary.json", payload)
    lines = [
        f"# Deterministic Final Gate - {run_dir.name}",
        "",
        f"- Status: `{status}`",
        f"- Detectability: `{payload['detectability_summary']}`",
        f"- Forced-depth sweep: `{payload['forced_depth_sweep_summary']}`",
        f"- K-fold selector: `{payload['kfold_summary']}`",
        "",
    ]
    detectability = payload.get("detectability") or {}
    best = detectability.get("best_detectability") or {}
    if best:
        lines.extend(
            [
                "## Step 1 Detectability",
                "",
                f"- Detectability status: `{detectability.get('status')}`",
                f"- Best shrinkage: `{best.get('shrinkage')}`",
                f"- Observed agreement: `{best.get('observed_alignment')}`",
                f"- Null p95 agreement: `{best.get('null_p95_alignment')}`",
                f"- Margin: `{best.get('observed_minus_null_p95')}`",
                "",
            ]
        )
    kfold = payload.get("kfold") or {}
    primary = kfold.get("primary_conservative_result") or {}
    if primary:
        lines.extend(
            [
                "## Step 2 K-Fold Transfer",
                "",
                f"- K-fold status: `{kfold.get('status')}`",
                f"- Primary policy: `{primary.get('policy_label')}` shrinkage `{primary.get('shrinkage')}`",
                f"- Correct: `{primary.get('correct')}/{primary.get('total')}`",
                f"- Loop 1 correct: `{primary.get('loop1_correct')}/{primary.get('total')}`",
                f"- Delta vs loop 1: `{primary.get('delta_vs_loop1')}`",
                f"- Rescue/harm: `{primary.get('rescue_captured')}/{primary.get('harm_triggered')}`",
            ]
        )
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    return run_dir


try:
    sync_repo()
    os.chdir(ROOT)
    print(f"STAGE5_DETERMINISTIC_FINAL_GATE_CELL_VERSION={STAGE5_DETERMINISTIC_FINAL_GATE_CELL_VERSION}", flush=True)
    run(["git", "log", "--oneline", "-5"], cwd=ROOT, check=False)
    mount_drive()
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT)
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_evaluate_rescue_detectability.py",
            "tests/test_evaluate_rescue_selector_kfold.py",
            "tests/test_stage5_benchmark_suite.py",
        ],
        cwd=ROOT,
    )

    run_id = os.environ.get("STAGE5_FINAL_GATE_RUN_ID") or time.strftime(
        "stage5_deterministic_final_gate_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    detectability_dir = run_dir / "detectability_cyclic"
    discovery_sweep = os.environ.get(
        "STAGE5_FINAL_GATE_DISCOVERY_SWEEP",
        "outputs/stage5/stage5_prelude_forced_depth_heldout_arc_loop1248/summary.json",
    )
    run(
        [
            sys.executable,
            "eval/evaluate_rescue_detectability.py",
            "--sweep_summary",
            discovery_sweep,
            "--benchmark",
            "arc_challenge",
            "--score_target",
            "cyclic_label_aggregated",
            "--aggregate",
            "permutation_mean",
            "--repeats",
            os.environ.get("STAGE5_FINAL_GATE_DETECTABILITY_REPEATS", "64"),
            "--permutations",
            os.environ.get("STAGE5_FINAL_GATE_DETECTABILITY_PERMUTATIONS", "128"),
            "--sample_fraction",
            os.environ.get("STAGE5_FINAL_GATE_DETECTABILITY_SAMPLE_FRACTION", "0.7"),
            "--seed",
            os.environ.get("STAGE5_FINAL_GATE_SEED", "17"),
            "--run_id",
            f"{run_id}_detectability_cyclic",
            "--output_dir",
            path_for_cli(detectability_dir),
        ],
        cwd=ROOT,
    )
    detectability = read_json(detectability_dir / "summary.json")
    if detectability.get("status") != "passed":
        write_final_summary(
            run_dir=run_dir,
            detectability_dir=detectability_dir,
            status="closed_at_detectability_gate",
        )
        publish([run_dir, detectability_dir])
        disconnect("deterministic final gate closed at detectability")
    else:
        require_cuda_runtime()
        source_summary = os.environ.get(
            "STAGE5_FINAL_GATE_SOURCE_SUMMARY",
            "outputs/stage5/stage5_prelude_path_development/summary.json",
        )
        loops = [int(item.strip()) for item in os.environ.get("STAGE5_FINAL_GATE_LOOPS", "1,2,4,8").split(",") if item.strip()]
        loops_tag = "".join(str(loop) for loop in loops)
        sweep_id = f"{run_id}_forced_depth_pooled_loop{loops_tag}"
        loop_run_ids: list[str] = []
        for forced_loop in loops:
            loop_run_id = f"{sweep_id}_loop{forced_loop}"
            loop_run_ids.append(loop_run_id)
            env = os.environ.copy()
            env.update(
                {
                    "STAGE5_BENCHMARK_SUITE_RUN_ID": loop_run_id,
                    "STAGE5_BENCHMARK_SOURCE_SUMMARY": source_summary,
                    "STAGE5_BENCHMARKS": os.environ.get(
                        "STAGE5_FINAL_GATE_BENCHMARKS",
                        "arc_easy,arc_challenge,open_hard_arc_challenge",
                    ),
                    "STAGE5_BENCHMARK_ARC_EASY_LIMIT": os.environ.get("STAGE5_FINAL_GATE_ARC_EASY_LIMIT", "all"),
                    "STAGE5_BENCHMARK_ARC_CHALLENGE_LIMIT": os.environ.get(
                        "STAGE5_FINAL_GATE_ARC_CHALLENGE_LIMIT",
                        "all",
                    ),
                    "STAGE5_BENCHMARK_OPEN_HARD_ARC_CHALLENGE_LIMIT": os.environ.get(
                        "STAGE5_FINAL_GATE_OPEN_HARD_ARC_CHALLENGE_LIMIT",
                        "256",
                    ),
                    "STAGE5_BENCHMARK_OPEN_HARD_ARC_CHALLENGE_SPLIT": os.environ.get(
                        "STAGE5_FINAL_GATE_OPEN_HARD_ARC_CHALLENGE_SPLIT",
                        "test",
                    ),
                    "STAGE5_BENCHMARK_MAX_LOOPS": str(max(loops)),
                    "STAGE5_BENCHMARK_FORCED_LOOP_COUNT": str(forced_loop),
                    "STAGE5_BENCHMARK_USE_LEARNED_LOOP_CONTROL": "0",
                    "STAGE5_BENCHMARK_NUM_TRAJECTORIES": "1",
                    "STAGE5_BENCHMARK_SCORE_TARGETS": "cyclic_label_aggregated",
                    "STAGE5_BENCHMARK_AGGREGATES": "mean",
                    "STAGE5_BENCHMARK_INCLUDE_LOOP_DIAGNOSTICS": "1",
                    "STAGE5_BENCHMARK_LORA_RANK": "0",
                    "STAGE5_BENCHMARK_LORA_ALPHA": "16",
                    "STAGE5_BENCHMARK_PUSH": "0",
                    "DTYPE": os.environ.get("DTYPE", "bfloat16"),
                    "ADAPTER_DTYPE": os.environ.get("ADAPTER_DTYPE", "float32"),
                    "DEVICE": os.environ.get("DEVICE", "cuda"),
                }
            )
            print(f"\n===== final gate forced_loop_count={forced_loop} =====", flush=True)
            run([sys.executable, "colab/run_stage5_benchmark_suite.py"], cwd=ROOT, env=env)
        sweep_dir = write_forced_depth_sweep_summary(
            sweep_id=sweep_id,
            source_summary=source_summary,
            loops=loops,
            run_ids=loop_run_ids,
        )
        cyclic_analysis = sweep_dir / "depth_analysis_cyclic_label_aggregated"
        run(
            [
                sys.executable,
                "eval/analyze_depth_sweep.py",
                "--sweep_summary",
                path_for_cli(sweep_dir / "summary.json"),
                "--score_target",
                "cyclic_label_aggregated",
                "--aggregate",
                "permutation_mean",
                "--output_dir",
                path_for_cli(cyclic_analysis),
            ],
            cwd=ROOT,
        )
        kfold_dir = run_dir / "selector_kfold_cyclic"
        kfold_cmd = [
            sys.executable,
            "eval/evaluate_rescue_selector_kfold.py",
            "--sweep_summary",
            path_for_cli(sweep_dir / "summary.json"),
            "--benchmarks",
            os.environ.get("STAGE5_FINAL_GATE_BENCHMARKS", "arc_easy,arc_challenge,open_hard_arc_challenge"),
            "--score_target",
            "cyclic_label_aggregated",
            "--aggregate",
            "permutation_mean",
            "--folds",
            os.environ.get("STAGE5_FINAL_GATE_KFOLD_FOLDS", "5"),
            "--inner_folds",
            os.environ.get("STAGE5_FINAL_GATE_KFOLD_INNER_FOLDS", "4"),
            "--seed",
            os.environ.get("STAGE5_FINAL_GATE_SEED", "17"),
            "--shrinkages",
            os.environ.get("STAGE5_FINAL_GATE_KFOLD_SHRINKAGES", "0.1,1.0,10.0"),
            "--selection_policy_labels",
            os.environ.get("STAGE5_FINAL_GATE_KFOLD_POLICY_LABELS", "zero_harm,harm_budget_1"),
            "--run_id",
            f"{run_id}_selector_kfold_cyclic",
            "--output_dir",
            path_for_cli(kfold_dir),
        ]
        primary_shrinkage = os.environ.get("STAGE5_FINAL_GATE_KFOLD_PRIMARY_SHRINKAGE", "").strip()
        if primary_shrinkage:
            kfold_cmd.extend(["--primary_shrinkage", primary_shrinkage])
        run(kfold_cmd, cwd=ROOT)
        write_final_summary(
            run_dir=run_dir,
            detectability_dir=detectability_dir,
            status=read_json(kfold_dir / "summary.json").get("status", "selector_transfer_needs_review"),
            forced_depth_sweep=sweep_dir,
            kfold_dir=kfold_dir,
        )
        publish([run_dir, detectability_dir, sweep_dir, cyclic_analysis, kfold_dir])
        disconnect("deterministic final gate finished")
except Exception:
    disconnect("deterministic final gate errored")
    raise
