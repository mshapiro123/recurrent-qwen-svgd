"""Colab launcher: candidate-conversion diagnostic for particle breadth.

This bounded Stage 5 diagnostic asks whether perturbation-generated pathways
produce additional correct candidates, or mostly fragment into wrong answers.
It is intentionally smaller than a benchmark run and is suitable for L4/T4.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from google.colab import drive, runtime, userdata

STAGE5_CANDIDATE_CONVERSION_CELL_VERSION = "stage5_candidate_conversion_v3_chunk_merge"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")


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
    print("HF token loaded", flush=True)
else:
    print("HF token not found; downloads will be anonymous.", flush=True)


def run(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print(f"$ {printable}", flush=True)
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    proc.check_returncode()
    return proc


def run_stream(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print(f"$ {printable}", flush=True)
    proc_env = dict(os.environ)
    if env:
        proc_env.update(env)
    proc_env["PYTHONUNBUFFERED"] = "1"
    with subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        env=proc_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, list(map(str, cmd)))


def ensure_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "pull", "--ff-only", "origin", "main"])
    else:
        run(["git", "clone", clone_url, ROOT], cwd=Path("/content"))
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"])


def restore_checkpoint(rel_path: str) -> Path:
    target = ROOT / rel_path
    if target.exists():
        print(f"checkpoint already local: {target}", flush=True)
        return target

    print("Mounting Drive to restore checkpoint.", flush=True)
    drive.mount("/content/drive", force_remount=False)

    candidates = [
        DRIVE_ARTIFACT_ROOT / rel_path,
        Path("/content/drive/MyDrive/recurrent-qwen-svgd") / rel_path,
    ]
    run_id = Path(rel_path).parts[2] if rel_path.startswith("outputs/stage5/") and len(Path(rel_path).parts) > 2 else ""
    for root in [DRIVE_ARTIFACT_ROOT, Path("/content/drive/MyDrive/recurrent-qwen-svgd")]:
        if root.exists() and run_id:
            candidates.extend(p for p in root.rglob(target.name) if run_id in str(p))

    for candidate in candidates:
        if candidate.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
            print(f"restored_checkpoint={candidate} -> {target}", flush=True)
            return target

    raise FileNotFoundError(f"Could not restore checkpoint from Drive: {rel_path}")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in split_csv(value)]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in split_csv(value)]


def float_key(value: object) -> str:
    return f"{float(value):.12g}"


def float_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def row_setting(row: dict[str, object]) -> tuple[int, str, int]:
    return (
        int(row.get("seed", 0)),
        float_key(row.get("particle_init_noise", 0.0)),
        int(row.get("max_loops", 0)),
    )


def setting_key(row: dict[str, object]) -> tuple[float, int, int]:
    return (
        float(row.get("particle_init_noise", 0.0)),
        int(row.get("max_loops", 0)),
        int(row.get("seed", 0)),
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def task_names(tasks_jsonl: Path) -> list[str]:
    return [json.loads(line)["name"] for line in tasks_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]


def completed_and_incomplete_settings(
    rows: list[dict[str, object]],
    *,
    tasks: list[str],
    num_trajectories: int,
) -> tuple[set[tuple[int, str, int]], set[tuple[int, str, int]]]:
    by_setting: dict[tuple[int, str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        by_setting[row_setting(row)][str(row["task"])] += 1
    complete: set[tuple[int, str, int]] = set()
    incomplete: set[tuple[int, str, int]] = set()
    expected_total = len(tasks) * num_trajectories
    for setting, counts in by_setting.items():
        total = sum(counts.values())
        if total == expected_total and all(counts.get(task, 0) == num_trajectories for task in tasks):
            complete.add(setting)
        else:
            incomplete.add(setting)
    return complete, incomplete


def prune_setting(path: Path, setting: tuple[int, str, int]) -> int:
    rows = read_jsonl(path)
    kept = [row for row in rows if row_setting(row) != setting]
    removed = len(rows) - len(kept)
    if removed:
        write_jsonl(path, kept)
    return removed


def merge_setting_rows(master_path: Path, chunk_path: Path, setting: tuple[int, str, int]) -> int:
    chunk_rows = read_jsonl(chunk_path)
    if not chunk_rows:
        raise RuntimeError(f"Chunk produced no rows: {chunk_path}")
    mismatched = [row_setting(row) for row in chunk_rows if row_setting(row) != setting]
    if mismatched:
        raise RuntimeError(
            f"Chunk setting mismatch for {chunk_path}: expected={setting} first_bad={mismatched[0]}"
        )
    master_rows = [row for row in read_jsonl(master_path) if row_setting(row) != setting]
    write_jsonl(master_path, master_rows + chunk_rows)
    return len(chunk_rows)


def pathway_q2(split: dict[str, object], bucket: str) -> float | None:
    payload = split.get(bucket)
    if not isinstance(payload, dict):
        return None
    diversity = payload.get("effective_pathways")
    if not isinstance(diversity, dict):
        return None
    value = diversity.get("2")
    return float(value) if value is not None else None


def summarize_candidate_conversion(jsonl_path: Path) -> dict[str, object]:
    rows = read_jsonl(jsonl_path)
    by_task: dict[tuple[float, int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        noise, loops, seed = setting_key(row)
        by_task[(noise, loops, seed, str(row["task"]))].append(row)

    settings: dict[tuple[float, int], dict[str, object]] = {}
    for key, task_rows in by_task.items():
        noise, loops, seed, task = key
        first = task_rows[0]
        setting = settings.setdefault(
            (noise, loops),
            {
                "particle_init_noise": noise,
                "max_loops": loops,
                "task_groups": 0,
                "best_hits": 0,
                "candidate_hits": 0,
                "total_candidates": 0,
                "unique_counts": [],
                "generation_steps": [],
                "all_q2": [],
                "correct_q2": [],
                "wrong_q2": [],
                "tasks": defaultdict(lambda: {"task_groups": 0, "best_hits": 0, "candidate_hits": 0, "total_candidates": 0}),
            },
        )
        hits = [bool(row["hit"]) for row in task_rows]
        candidate_hits = sum(hits)
        total_candidates = len(task_rows)
        best_hit = any(hits)
        setting["task_groups"] += 1
        setting["best_hits"] += int(best_hit)
        setting["candidate_hits"] += candidate_hits
        setting["total_candidates"] += total_candidates
        setting["unique_counts"].append(int(first.get("unique_count", 0)))
        setting["generation_steps"].append(int(first.get("generation_steps", 0)))

        split = first.get("pathway_split_diagnostics") or {}
        if isinstance(split, dict):
            for bucket, values in [("all", setting["all_q2"]), ("correct", setting["correct_q2"]), ("wrong", setting["wrong_q2"])]:
                value = pathway_q2(split, bucket)
                if value is not None:
                    values.append(value)

        task_bucket = setting["tasks"][task]
        task_bucket["task_groups"] += 1
        task_bucket["best_hits"] += int(best_hit)
        task_bucket["candidate_hits"] += candidate_hits
        task_bucket["total_candidates"] += total_candidates

    summary_rows = []
    for (_, _), setting in sorted(settings.items()):
        tasks = {
            task: dict(values)
            for task, values in sorted(setting.pop("tasks").items())
        }
        unique_counts = list(setting.pop("unique_counts"))
        generation_steps = list(setting.pop("generation_steps"))
        all_q2 = list(setting.pop("all_q2"))
        correct_q2 = list(setting.pop("correct_q2"))
        wrong_q2 = list(setting.pop("wrong_q2"))
        summary_rows.append(
            {
                **setting,
                "best_rate": setting["best_hits"] / max(setting["task_groups"], 1),
                "candidate_hit_rate": setting["candidate_hits"] / max(setting["total_candidates"], 1),
                "mean_unique": mean(unique_counts) if unique_counts else None,
                "mean_generation_steps": mean(generation_steps) if generation_steps else None,
                "mean_all_q2": mean(all_q2) if all_q2 else None,
                "mean_correct_q2": mean(correct_q2) if correct_q2 else None,
                "mean_wrong_q2": mean(wrong_q2) if wrong_q2 else None,
                "task_breakdown": tasks,
            }
        )

    return {
        "kind": "stage5_candidate_conversion_summary",
        "rows": len(rows),
        "task_groups": len(by_task),
        "settings": summary_rows,
    }


def markdown_summary(summary: dict[str, object], *, run_id: str, checkpoint: str, jsonl_path: Path) -> str:
    lines = [
        f"# Stage 5 Candidate Conversion - {run_id}",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- JSONL: `{jsonl_path}`",
        "",
        "## Setting Summary",
        "",
        "| noise | loops | best | candidates | mean unique | all q2 | correct q2 | wrong q2 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["settings"]:
        lines.append(
            "| "
            f"{row['particle_init_noise']:.4g} | {row['max_loops']} | "
            f"{row['best_hits']}/{row['task_groups']} | "
            f"{row['candidate_hits']}/{row['total_candidates']} | "
            f"{row['mean_unique']:.3f} | "
            f"{row['mean_all_q2'] if row['mean_all_q2'] is not None else 'n/a'} | "
            f"{row['mean_correct_q2'] if row['mean_correct_q2'] is not None else 'n/a'} | "
            f"{row['mean_wrong_q2'] if row['mean_wrong_q2'] is not None else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Prompt",
            "",
            "Use this to decide whether particle breadth is correct-bearing. If candidate hits rise with noise and correct q2 rises, selector work is justified. If uniqueness/q2 rise while candidate hits fall, noise is fragmentation and the next move is training-time pathway shaping, not stronger inference noise.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    print(f"cell_version={STAGE5_CANDIDATE_CONVERSION_CELL_VERSION}", flush=True)
    run_id = os.environ.get("STAGE5_CANDIDATE_CONVERSION_RUN_ID") or time.strftime(
        "stage5_candidate_conversion_%Y%m%d_%H%M%S"
    )
    checkpoint_override = os.environ.get("STAGE5_CANDIDATE_CONVERSION_CHECKPOINT")
    tasks_jsonl = os.environ.get("STAGE5_CANDIDATE_CONVERSION_TASKS", "eval/smoke_exact_tasks_v2.jsonl")
    seeds = os.environ.get("STAGE5_CANDIDATE_CONVERSION_SEEDS", "0,1,2")
    noise_sweep = os.environ.get("STAGE5_CANDIDATE_CONVERSION_NOISE_SWEEP", "0,0.005,0.01,0.02,0.05")
    loops_sweep = os.environ.get("STAGE5_CANDIDATE_CONVERSION_MAX_LOOPS_SWEEP", "4,8")
    num_trajectories = os.environ.get("STAGE5_CANDIDATE_CONVERSION_K", "4")
    max_new_tokens = os.environ.get("STAGE5_CANDIDATE_CONVERSION_MAX_NEW_TOKENS", "96")
    dtype = os.environ.get("STAGE5_CANDIDATE_CONVERSION_DTYPE", "bfloat16")
    disconnect = os.environ.get("STAGE5_CANDIDATE_CONVERSION_DISCONNECT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    ensure_repo()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from colab.master_sequence_gate import require_phase1_depth_gate_for_breadth

    phase_gate = require_phase1_depth_gate_for_breadth(
        root=ROOT,
        action_name="candidate_conversion_diagnostic",
        allow_env="STAGE5_ALLOW_PRE_PHASE1_BREADTH",
    )
    print("master_sequence_phase2_gate=" + json.dumps(phase_gate, sort_keys=True), flush=True)
    checkpoint = checkpoint_override or str(phase_gate.get("checkpoint") or "")
    if not checkpoint:
        raise RuntimeError(
            "No recurrent checkpoint resolved for candidate-conversion diagnostics. "
            "Run through the Phase 1 gate or set STAGE5_CANDIDATE_CONVERSION_CHECKPOINT explicitly."
        )
    print(
        "candidate_conversion_checkpoint_source="
        + json.dumps(
            {
                "checkpoint": checkpoint,
                "override": checkpoint_override is not None,
                "phase_gate_checkpoint_source_summary": phase_gate.get("checkpoint_source_summary"),
                "phase_gate_checkpoint_source_kind": phase_gate.get("checkpoint_source_kind"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    run(["nvidia-smi"], cwd=Path("/content"))
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_eval_best_of_k_generation.py",
            "tests/test_recurrent_wrapper_tiny.py",
            "tests/test_pathway_diversity.py",
            "tests/test_eval_effective_pathways.py",
        ]
    )
    restore_checkpoint(checkpoint)

    out_dir = ROOT / "outputs" / "stage5" / run_id
    diag_dir = out_dir / "candidate_conversion"
    chunk_dir = diag_dir / "chunks"
    diag_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = diag_dir / "candidate_conversion.jsonl"

    names = task_names(ROOT / tasks_jsonl)
    seed_values = parse_ints(seeds)
    noise_values = parse_floats(noise_sweep)
    loop_values = parse_ints(loops_sweep)
    expected_setting_count = len(seed_values) * len(noise_values) * len(loop_values)
    print(
        "resume_plan="
        + json.dumps(
            {
                "run_id": run_id,
                "settings": expected_setting_count,
                "tasks": len(names),
                "num_trajectories": int(num_trajectories),
                "rows_per_setting": len(names) * int(num_trajectories),
                "output_jsonl": str(output_jsonl.relative_to(ROOT)),
            }
        ),
        flush=True,
    )

    for loops in loop_values:
        for noise in noise_values:
            for seed in seed_values:
                setting = (seed, float_key(noise), loops)
                rows = read_jsonl(output_jsonl)
                complete, incomplete = completed_and_incomplete_settings(
                    rows,
                    tasks=names,
                    num_trajectories=int(num_trajectories),
                )
                if setting in complete:
                    print(f"skip_completed seed={seed} noise={noise:g} loops={loops}", flush=True)
                    continue
                if setting in incomplete:
                    removed = prune_setting(output_jsonl, setting)
                    print(f"pruned_incomplete seed={seed} noise={noise:g} loops={loops} rows={removed}", flush=True)

                print(f"run_setting seed={seed} noise={noise:g} loops={loops}", flush=True)
                chunk_jsonl = chunk_dir / f"seed{seed}_noise{float_label(noise)}_loops{loops}.jsonl"
                if chunk_jsonl.exists():
                    chunk_jsonl.unlink()
                run_stream(
                    [
                        sys.executable,
                        "eval/eval_best_of_k_jsonl.py",
                        "--tasks_jsonl",
                        tasks_jsonl,
                        "--skip_phase1",
                        "--compact",
                        "--seed",
                        str(seed),
                        "--phase1_checkpoint",
                        checkpoint,
                        "--phase2_checkpoint",
                        checkpoint,
                        "--max_loops",
                        str(loops),
                        "--phase2_num_trajectories",
                        num_trajectories,
                        "--phase2_particle_update_mode",
                        "none",
                        "--particle_init_noise",
                        str(noise),
                        "--max_new_tokens",
                        max_new_tokens,
                        "--temperature",
                        os.environ.get("STAGE5_CANDIDATE_CONVERSION_TEMPERATURE", "0.0"),
                        "--dtype",
                        dtype,
                        "--adapter_dtype",
                        os.environ.get("STAGE5_CANDIDATE_CONVERSION_ADAPTER_DTYPE", "float32"),
                        "--device",
                        os.environ.get("STAGE5_CANDIDATE_CONVERSION_DEVICE", "cuda"),
                        "--output_jsonl",
                        str(chunk_jsonl.relative_to(ROOT)),
                    ]
                )
                merged_rows = merge_setting_rows(output_jsonl, chunk_jsonl, setting)
                print(
                    f"merged_setting seed={seed} noise={noise:g} loops={loops} rows={merged_rows}",
                    flush=True,
                )
                rows = read_jsonl(output_jsonl)
                complete, _ = completed_and_incomplete_settings(rows, tasks=names, num_trajectories=int(num_trajectories))
                print(f"progress_completed_settings={len(complete)}/{expected_setting_count}", flush=True)

    summary = summarize_candidate_conversion(output_jsonl)
    summary.update(
        {
            "run_id": run_id,
            "cell_version": STAGE5_CANDIDATE_CONVERSION_CELL_VERSION,
            "checkpoint": checkpoint,
            "phase_gate": phase_gate,
            "checkpoint_override": checkpoint_override is not None,
            "tasks_jsonl": tasks_jsonl,
            "seeds": split_csv(seeds),
            "noise_sweep": split_csv(noise_sweep),
            "max_loops_sweep": split_csv(loops_sweep),
            "num_trajectories": int(num_trajectories),
            "output_jsonl": str(output_jsonl.relative_to(ROOT)),
        }
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        markdown_summary(summary, run_id=run_id, checkpoint=checkpoint, jsonl_path=output_jsonl.relative_to(ROOT)),
        encoding="utf-8",
    )
    print((out_dir / "summary.md").read_text(encoding="utf-8"), flush=True)

    if Path("/content/drive/MyDrive").exists():
        backup_dir = DRIVE_ARTIFACT_ROOT / "outputs" / "stage5" / run_id
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(out_dir, backup_dir)
        print(f"drive_backup={backup_dir}", flush=True)

    run(["git", "status", "-sb"])
    run(["git", "add", "-f", str(out_dir.relative_to(ROOT))])
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(ROOT))
    if diff.returncode != 0:
        run(["git", "commit", "-m", f"Record Stage 5 candidate conversion {run_id} [skip ci]"])
        if os.environ.get("STAGE5_CANDIDATE_CONVERSION_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}:
            run(["git", "fetch", "origin", "main"])
            run(["git", "rebase", "origin/main"])
            run(["git", "push", "origin", "main"])
    else:
        print("No staged changes to commit.", flush=True)

    if disconnect:
        print("Disconnecting Colab runtime to conserve credits.", flush=True)
        runtime.unassign()


main()
