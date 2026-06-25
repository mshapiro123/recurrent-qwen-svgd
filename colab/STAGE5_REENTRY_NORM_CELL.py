"""Colab launcher: Stage 5 eval-only re-entry RMS normalization comparison.

This is Stage 2 after the re-entry drift diagnostic. It compares the current
loop closure against an eval-only RMS rescale of loop inputs back to the
Prelude->recurrent entry RMS. It does not train or save a model checkpoint.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from google.colab import drive, runtime, userdata

STAGE5_REENTRY_NORM_CELL_VERSION = "stage5_reentry_norm_v1_eval_only"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")

DEFAULT_CHECKPOINT = (
    "outputs/stage5/stage5_traced_sft_direct_preservation_20260623_scale64_lr1e6/"
    "phase1_direct_preserve/phase1_step_75.pt"
)
FALLBACK_CHECKPOINTS = [
    DEFAULT_CHECKPOINT,
    (
        "outputs/stage5/stage5_content_arcmix_qonly_optiontext_20260623_121707/"
        "arc_mix_response_w02_lr2e6/phase1/phase1_step_150.pt"
    ),
    (
        "outputs/stage5/stage5_arc_mix_recovery_once_20260622_003331/"
        "arc_mix_response_w005_lr2e6/phase1/phase1_step_50.pt"
    ),
]


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
    check: bool = True,
) -> subprocess.CompletedProcess:
    printable = " ".join(map(str, cmd)).replace(GH_TOKEN, "****")
    print(f"$ {printable}", flush=True)
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout, flush=True)
    if check:
        proc.check_returncode()
    return proc


def ensure_repo() -> None:
    clone_url = f"https://x-access-token:{GH_TOKEN}@github.com/{REPO}.git"
    if ROOT.exists():
        run(["git", "remote", "set-url", "origin", clone_url])
        run(["git", "fetch", "origin", "main"])
        run(["git", "checkout", "main"])
        run(["git", "reset", "--hard", "origin/main"])
    else:
        run(["git", "clone", clone_url, ROOT], cwd=Path("/content"))
    run(["git", "config", "user.email", "colab-runner@local"])
    run(["git", "config", "user.name", "Colab Runner"])
    run(["git", "log", "--oneline", "-5"])


def normalize_rel_path(path: str) -> str:
    return str(path).replace("\\", "/").lstrip("/")


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def checkpoint_candidates(requested: str, *, allow_fallback: bool) -> list[str]:
    requested = normalize_rel_path(requested)
    candidates = [requested]
    if allow_fallback:
        candidates.extend(path for path in FALLBACK_CHECKPOINTS if normalize_rel_path(path) not in candidates)
    return candidates


def drive_checkpoint_candidates(rel_path: str, target: Path) -> list[Path]:
    rel_path = normalize_rel_path(rel_path)
    rel = Path(rel_path)
    roots = [DRIVE_ARTIFACT_ROOT, Path("/content/drive/MyDrive/recurrent-qwen-svgd")]
    candidates: list[Path] = []
    for root in roots:
        candidates.append(root / rel_path)
        if rel_path.startswith("outputs/stage5/") and len(rel.parts) > 3:
            run_id = rel.parts[2]
            after_run = Path(*rel.parts[3:])
            candidates.append(root / run_id / after_run)
            candidates.append(root / "outputs" / "stage5" / run_id / after_run)
            if root.exists():
                candidates.extend(
                    p
                    for p in root.rglob(target.name)
                    if run_id in p.as_posix() and p.name == target.name
                )
    return unique_paths(candidates)


def restore_one_checkpoint(rel_path: str) -> Path:
    rel_path = normalize_rel_path(rel_path)
    target = ROOT / rel_path
    if target.exists():
        print(f"checkpoint already local: {target}", flush=True)
        return target

    print("Mounting Drive to restore checkpoint.", flush=True)
    drive.mount("/content/drive", force_remount=False)

    candidates = drive_checkpoint_candidates(rel_path, target)

    for candidate in candidates:
        if candidate.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
            print(f"restored_checkpoint={candidate} -> {target}", flush=True)
            return target

    preview = "\n".join(f"  - {path}" for path in candidates[:12])
    raise FileNotFoundError(f"Could not restore checkpoint from Drive: {rel_path}\nTried:\n{preview}")


def restore_checkpoint(rel_path: str, *, allow_fallback: bool) -> Path:
    errors: list[str] = []
    for candidate in checkpoint_candidates(rel_path, allow_fallback=allow_fallback):
        try:
            return restore_one_checkpoint(candidate)
        except FileNotFoundError as exc:
            errors.append(str(exc))
            if not allow_fallback:
                break
            print(f"checkpoint candidate unavailable, trying fallback: {candidate}", flush=True)
    raise FileNotFoundError("\n\n".join(errors))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def has_valid_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return True


def has_valid_jsonl(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(read_jsonl(path))
    except (OSError, json.JSONDecodeError):
        return False


def summarize_candidate_jsonl(path: Path) -> dict[str, object]:
    rows = read_jsonl(path)
    grouped: dict[tuple[object, object, object, object], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row.get("reentry_rescale_mode", "none"),
            row.get("max_loops"),
            row.get("particle_init_noise"),
            row.get("task"),
        )
        grouped.setdefault(key, []).append(row)
    by_mode: dict[str, dict[str, float | int]] = {}
    for (mode, _loops, _noise, _task), group in grouped.items():
        mode_key = str(mode)
        current = by_mode.setdefault(
            mode_key,
            {
                "task_groups": 0,
                "best_hits": 0,
                "candidate_hits": 0,
                "total_candidates": 0,
                "unique_sum": 0,
            },
        )
        current["task_groups"] += 1
        current["best_hits"] += int(any(bool(row.get("hit")) for row in group))
        current["candidate_hits"] += sum(int(bool(row.get("hit"))) for row in group)
        current["total_candidates"] += len(group)
        current["unique_sum"] += len({str(row.get("candidate", "")).strip() for row in group})
    for mode, current in by_mode.items():
        groups = max(int(current["task_groups"]), 1)
        current["mean_unique"] = float(current["unique_sum"]) / groups
    return {"rows": len(rows), "by_mode": by_mode}


def limited_tasks_jsonl(source: str, out_dir: Path) -> str:
    limit = int(
        os.environ.get(
            "STAGE5_REENTRY_NORM_CANDIDATE_TASK_LIMIT",
            os.environ.get("STAGE5_REENTRY_NORM_LIMIT", "8"),
        )
    )
    if limit <= 0:
        return source
    source_path = ROOT / normalize_rel_path(source)
    rows = [line for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = rows[:limit]
    if len(selected) == len(rows):
        return source
    out_path = out_dir / f"candidate_tasks_first_{limit}.jsonl"
    out_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(f"candidate_task_limit={limit} task_file={out_path.relative_to(ROOT).as_posix()}", flush=True)
    return out_path.relative_to(ROOT).as_posix()


def candidate_conversion_is_complete(path: Path, tasks_jsonl: str) -> bool:
    if not has_valid_jsonl(path):
        return False
    rows = read_jsonl(path)
    tasks = [line for line in (ROOT / normalize_rel_path(tasks_jsonl)).read_text(encoding="utf-8").splitlines() if line.strip()]
    seeds = parse_csv_ints(os.environ.get("STAGE5_REENTRY_NORM_SEEDS", "0"))
    noises = parse_csv_floats(os.environ.get("STAGE5_REENTRY_NORM_NOISE_SWEEP", "0,0.05"))
    loops = parse_csv_ints(os.environ.get("STAGE5_REENTRY_NORM_LOOP_SWEEP", "4,8"))
    k = int(os.environ.get("STAGE5_REENTRY_NORM_K", "4"))
    expected_rows = len(tasks) * len(seeds) * len(noises) * len(loops) * k
    if len(rows) != expected_rows:
        print(
            f"incomplete_candidate_conversion={path.relative_to(ROOT).as_posix()} "
            f"rows={len(rows)} expected={expected_rows}",
            flush=True,
        )
        return False

    grouped: dict[tuple[int, int, float, str], int] = {}
    for row in rows:
        key = (
            int(row.get("seed", -1)),
            int(row.get("max_loops", -1)),
            float(row.get("particle_init_noise", -1.0)),
            str(row.get("task", "")),
        )
        grouped[key] = grouped.get(key, 0) + 1
    if len(grouped) != len(tasks) * len(seeds) * len(noises) * len(loops):
        print(
            f"incomplete_candidate_groups={path.relative_to(ROOT).as_posix()} "
            f"groups={len(grouped)} expected={len(tasks) * len(seeds) * len(noises) * len(loops)}",
            flush=True,
        )
        return False
    bad_groups = {key: count for key, count in grouped.items() if count != k}
    if bad_groups:
        preview = list(bad_groups.items())[:5]
        print(f"incomplete_candidate_group_sizes={preview}", flush=True)
        return False
    return True


def run_drift(mode: str, checkpoint: str, prompts: str, out_dir: Path) -> Path:
    out_json = out_dir / f"reentry_drift_{mode}.json"
    out_jsonl = out_dir / f"reentry_drift_{mode}.jsonl"
    if has_valid_json(out_json) and has_valid_jsonl(out_jsonl):
        print(f"resume_skip=reentry_drift_{mode}", flush=True)
        return out_json
    run(
        [
            sys.executable,
            "eval/eval_reentry_drift.py",
            "--checkpoint",
            checkpoint,
            "--prompts_jsonl",
            prompts,
            "--limit",
            os.environ.get("STAGE5_REENTRY_NORM_LIMIT", "8"),
            "--max_loops",
            os.environ.get("STAGE5_REENTRY_NORM_MAX_LOOPS", "8"),
            "--max_length",
            os.environ.get("STAGE5_REENTRY_NORM_MAX_LENGTH", "256"),
            "--reentry_rescale_mode",
            mode,
            "--dtype",
            os.environ.get("STAGE5_REENTRY_NORM_DTYPE", "bfloat16"),
            "--adapter_dtype",
            os.environ.get("STAGE5_REENTRY_NORM_ADAPTER_DTYPE", "float32"),
            "--device",
            os.environ.get("STAGE5_REENTRY_NORM_DEVICE", "cuda"),
            "--output_json",
            str(out_json.relative_to(ROOT)),
            "--output_jsonl",
            str(out_jsonl.relative_to(ROOT)),
        ]
    )
    return out_json


def run_effective_pathways(mode: str, checkpoint: str, prompts: str, out_dir: Path) -> Path:
    out_json = out_dir / f"effective_pathways_{mode}.json"
    out_jsonl = out_dir / f"effective_pathways_{mode}.jsonl"
    if has_valid_json(out_json) and has_valid_jsonl(out_jsonl):
        print(f"resume_skip=effective_pathways_{mode}", flush=True)
        return out_json
    run(
        [
            sys.executable,
            "eval/eval_effective_pathways.py",
            "--checkpoint",
            checkpoint,
            "--prompts_jsonl",
            prompts,
            "--limit",
            os.environ.get("STAGE5_REENTRY_NORM_LIMIT", "8"),
            "--max_loops",
            os.environ.get("STAGE5_REENTRY_NORM_EFFECTIVE_MAX_LOOPS", "8"),
            "--num_particles",
            os.environ.get("STAGE5_REENTRY_NORM_NUM_PARTICLES", "16"),
            "--particle_init_noise",
            os.environ.get("STAGE5_REENTRY_NORM_PARTICLE_NOISE", "0.05"),
            "--max_length",
            os.environ.get("STAGE5_REENTRY_NORM_MAX_LENGTH", "256"),
            "--reentry_rescale_mode",
            mode,
            "--dtype",
            os.environ.get("STAGE5_REENTRY_NORM_DTYPE", "bfloat16"),
            "--adapter_dtype",
            os.environ.get("STAGE5_REENTRY_NORM_ADAPTER_DTYPE", "float32"),
            "--device",
            os.environ.get("STAGE5_REENTRY_NORM_DEVICE", "cuda"),
            "--output_json",
            str(out_json.relative_to(ROOT)),
            "--output_jsonl",
            str(out_jsonl.relative_to(ROOT)),
        ]
    )
    return out_json


def run_candidate_conversion(mode: str, checkpoint: str, tasks: str, out_dir: Path) -> Path:
    out_jsonl = out_dir / f"candidate_conversion_{mode}.jsonl"
    candidate_tasks = limited_tasks_jsonl(tasks, out_dir)
    if candidate_conversion_is_complete(out_jsonl, candidate_tasks):
        print(f"resume_skip=candidate_conversion_{mode}", flush=True)
        return out_jsonl
    if out_jsonl.exists():
        out_jsonl.unlink()
    run(
        [
            sys.executable,
            "eval/eval_best_of_k_jsonl.py",
            "--tasks_jsonl",
            candidate_tasks,
            "--skip_phase1",
            "--compact",
            "--seeds",
            os.environ.get("STAGE5_REENTRY_NORM_SEEDS", "0"),
            "--phase1_checkpoint",
            checkpoint,
            "--phase2_checkpoint",
            checkpoint,
            "--phase2_num_trajectories",
            os.environ.get("STAGE5_REENTRY_NORM_K", "4"),
            "--phase2_particle_update_mode",
            "none",
            "--particle_init_noise_sweep",
            os.environ.get("STAGE5_REENTRY_NORM_NOISE_SWEEP", "0,0.05"),
            "--max_loops_sweep",
            os.environ.get("STAGE5_REENTRY_NORM_LOOP_SWEEP", "4,8"),
            "--reentry_rescale_mode",
            mode,
            "--temperature",
            "0.0",
            "--max_new_tokens",
            os.environ.get("STAGE5_REENTRY_NORM_MAX_NEW_TOKENS", "80"),
            "--dtype",
            os.environ.get("STAGE5_REENTRY_NORM_DTYPE", "bfloat16"),
            "--adapter_dtype",
            os.environ.get("STAGE5_REENTRY_NORM_ADAPTER_DTYPE", "float32"),
            "--device",
            os.environ.get("STAGE5_REENTRY_NORM_DEVICE", "cuda"),
            "--output_jsonl",
            str(out_jsonl.relative_to(ROOT)),
        ]
    )
    return out_jsonl


def incremental_backup(out_dir: Path) -> None:
    if os.environ.get("STAGE5_REENTRY_NORM_INCREMENTAL_BACKUP", "1").strip().lower() not in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return
    if not Path("/content/drive/MyDrive").exists():
        return
    backup_dir = DRIVE_ARTIFACT_ROOT / "outputs" / "stage5" / out_dir.name
    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(out_dir, backup_dir)
    print(f"incremental_backup={backup_dir}", flush=True)


def write_summary(
    run_id: str,
    out_dir: Path,
    checkpoint: str,
    prompts: str,
    paths: dict[str, dict[str, str]],
) -> dict[str, object]:
    drift = {mode: read_json(ROOT / path["drift"]) for mode, path in paths.items()}
    effective = {mode: read_json(ROOT / path["effective_pathways"]) for mode, path in paths.items()}
    candidate = {mode: summarize_candidate_jsonl(ROOT / path["candidate_conversion"]) for mode, path in paths.items()}
    summary: dict[str, object] = {
        "kind": "stage5_reentry_norm_eval_only",
        "run_id": run_id,
        "cell_version": STAGE5_REENTRY_NORM_CELL_VERSION,
        "checkpoint": checkpoint,
        "prompts": prompts,
        "paths": paths,
        "drift": {
            mode: payload.get("aggregate", {})
            for mode, payload in drift.items()
        },
        "effective_pathways": {
            mode: payload.get("aggregate", {})
            for mode, payload in effective.items()
        },
        "candidate_conversion": candidate,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Stage 5 Re-entry Norm - {run_id}",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Prompts/tasks: `{prompts}`",
        f"- Cell version: `{STAGE5_REENTRY_NORM_CELL_VERSION}`",
        "",
        "## Drift",
        "| mode | exit/entry RMS | loop8 input/entry RMS | loop8 output/entry RMS | subspace overlap |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, payload in summary["drift"].items():
        loop8 = payload.get("loop_summary", {}).get("8", {}) if isinstance(payload, dict) else {}
        subspace = payload.get("entry_exit_subspace", {}) if isinstance(payload, dict) else {}
        lines.append(
            f"| {mode} | {payload.get('mean_exit_over_entry_rms')} | "
            f"{loop8.get('input_over_entry_rms')} | {loop8.get('output_over_entry_rms')} | "
            f"{subspace.get('overlap')} |"
        )
    lines.extend(
        [
            "",
            "## Effective Pathways",
            "| mode | initial distance | final distance | spread ratio | q2 pathways | unique next-token argmax |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, payload in summary["effective_pathways"].items():
        q2 = (payload.get("mean_effective_pathways") or {}).get("2") if isinstance(payload, dict) else None
        lines.append(
            f"| {mode} | {payload.get('mean_initial_pairwise_distance')} | "
            f"{payload.get('mean_final_pairwise_distance')} | "
            f"{payload.get('mean_spread_ratio_final_over_initial')} | "
            f"{q2} | {payload.get('mean_unique_next_token_argmax')} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Conversion",
            "| mode | task groups | best | candidates | mean unique |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode, payload in summary["candidate_conversion"].items():
        mode_stats = (payload.get("by_mode") or {}).get(mode, {})
        lines.append(
            f"| {mode} | {mode_stats.get('task_groups')} | "
            f"{mode_stats.get('best_hits')} | "
            f"{mode_stats.get('candidate_hits')}/{mode_stats.get('total_candidates')} | "
            f"{mode_stats.get('mean_unique')} |"
        )
    lines.extend(
        [
            "",
            "## Readout Pause",
            "This run intentionally stops after Stage 2. Review before implementing trainable bridge/re-entry repair.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((out_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    return summary


def publish_outputs(out_dir: Path, run_id: str) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from colab.stage5_publish_utils import publishable_artifact_paths

    run(["git", "status", "-sb"])
    publishable = publishable_artifact_paths(out_dir)
    if not publishable:
        print(f"No lightweight publishable artifacts found under {out_dir}.", flush=True)
        return
    for path in publishable:
        run(["git", "add", "-f", str(path.relative_to(ROOT))])
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(ROOT))
    if diff.returncode == 0:
        print("No staged changes to commit.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 re-entry norm {run_id} [skip ci]"])
    if os.environ.get("STAGE5_REENTRY_NORM_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}:
        pushed = run(["git", "push", "origin", "main"], check=False)
        if pushed.returncode != 0:
            print("Initial push failed; attempting one fast rebase and retry.", flush=True)
            run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
            run(["git", "push", "origin", "main"])


def run_reentry_assessment(out_dir: Path) -> None:
    run(
        [
            sys.executable,
            "colab/assess_stage5_reentry.py",
            "--summary_json",
            str((out_dir / "summary.json").relative_to(ROOT)),
            "--output_json",
            str((out_dir / "reentry_assessment.json").relative_to(ROOT)),
            "--output_md",
            str((out_dir / "reentry_assessment.md").relative_to(ROOT)),
        ]
    )
    print((out_dir / "reentry_assessment.md").read_text(encoding="utf-8"), flush=True)


def main() -> None:
    print(f"cell_version={STAGE5_REENTRY_NORM_CELL_VERSION}", flush=True)
    run_id = os.environ.get("STAGE5_REENTRY_NORM_RUN_ID") or time.strftime("stage5_reentry_norm_%Y%m%d_%H%M%S")
    checkpoint_override = os.environ.get("STAGE5_REENTRY_NORM_CHECKPOINT")
    checkpoint = checkpoint_override or DEFAULT_CHECKPOINT
    prompts = os.environ.get("STAGE5_REENTRY_NORM_PROMPTS", "eval/smoke_exact_tasks_v2.jsonl")
    disconnect = os.environ.get("STAGE5_REENTRY_NORM_DISCONNECT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    ensure_repo()
    run(["nvidia-smi"], cwd=Path("/content"))
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_eval_reentry_drift.py",
            "tests/test_eval_effective_pathways.py",
            "tests/test_eval_best_of_k_generation.py",
            "tests/test_recurrent_wrapper_tiny.py",
        ]
    )
    checkpoint_path = restore_checkpoint(checkpoint, allow_fallback=checkpoint_override is None)
    checkpoint = checkpoint_path.relative_to(ROOT).as_posix()

    out_dir = ROOT / "outputs" / "stage5" / run_id
    diag_dir = out_dir / "reentry_norm"
    diag_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, dict[str, str]] = {}
    for mode in ("none", "entry_rms"):
        drift = run_drift(mode, checkpoint, prompts, diag_dir)
        effective = run_effective_pathways(mode, checkpoint, prompts, diag_dir)
        candidates = run_candidate_conversion(mode, checkpoint, prompts, diag_dir)
        paths[mode] = {
            "drift": drift.relative_to(ROOT).as_posix(),
            "effective_pathways": effective.relative_to(ROOT).as_posix(),
            "candidate_conversion": candidates.relative_to(ROOT).as_posix(),
        }
        incremental_backup(out_dir)

    write_summary(run_id, out_dir, checkpoint, prompts, paths)
    run_reentry_assessment(out_dir)

    incremental_backup(out_dir)

    publish_outputs(out_dir, run_id)

    if disconnect:
        print("Disconnecting Colab runtime to conserve credits after Stage 2 readout.", flush=True)
        runtime.unassign()


main()
