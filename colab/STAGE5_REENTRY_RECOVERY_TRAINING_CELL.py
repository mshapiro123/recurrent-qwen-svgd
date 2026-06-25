"""Colab launcher: bounded recovery SFT after re-entry repair passes.

This is Stage 4 of the re-entry reset. It deliberately refuses to run until a
Stage 3 repair-smoke assessment recommends
``run_bounded_recovery_training_with_reentry_repair``. When cleared, it resumes
from the repaired checkpoint and reuses the existing capability-ladder
curriculum SFT runner with depth-label supervision enabled.
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


STAGE5_REENTRY_RECOVERY_CELL_VERSION = "reentry_recovery_training_v2_depth_count_gate"
STAGE5_REENTRY_RECOVERY_TARGET = "reentry_recovery_training"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")
LEGACY_DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd")
DEFAULT_TRACE_COLLECTION = (
    "outputs/stage5/stage5_capability_ladder_trace_collection_20260623_194537/summary.json"
)


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
    print("HF token not found; downloads will use anonymous Hub access.", flush=True)


def redact(text: str) -> str:
    out = str(text)
    for token in (GH_TOKEN, HF_TOKEN):
        if token:
            out = out.replace(token, "****")
    return out


def run(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    printable = redact(" ".join(map(str, cmd)))
    print(f"$ {printable}", flush=True)
    process = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        chunks.append(safe)
    stdout = "".join(chunks)
    proc = subprocess.CompletedProcess(list(map(str, cmd)), process.wait(), stdout, None)
    if check and proc.returncode:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(stdout.splitlines()[-180:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=stdout)
    return proc


def normalize_rel_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").lstrip("/")


def resolve_repo_path(path: str | Path) -> Path:
    raw = Path(str(path).replace("\\", "/"))
    return raw if raw.is_absolute() else ROOT / normalize_rel_path(path)


def path_for_cli(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return candidate.relative_to(ROOT).as_posix()
    except ValueError:
        return str(candidate).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


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
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def mount_drive_if_needed() -> None:
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive", force_remount=False)


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def latest_matching(paths: list[Path]) -> Path | None:
    existing = [path for path in unique_paths(paths) if path.exists()]
    if not existing:
        return None
    return sorted(existing, key=lambda path: path.stat().st_mtime)[-1]


def repair_assessment_candidates() -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("STAGE5_REENTRY_RECOVERY_REPAIR_ASSESSMENT", "").strip()
    if override:
        rel = normalize_rel_path(override)
        candidates.extend([ROOT / rel, DRIVE_ARTIFACT_ROOT / rel, LEGACY_DRIVE_ROOT / rel])
    for root in (ROOT / "outputs" / "stage5", DRIVE_ARTIFACT_ROOT / "outputs" / "stage5"):
        if root.exists():
            candidates.extend(sorted(root.glob("stage5_reentry_repair_smoke_*/reentry_assessment.json")))
    return unique_paths(candidates)


def checkpoint_drive_candidates(rel_path: str, run_id: str | None = None) -> list[Path]:
    rel_path = normalize_rel_path(rel_path)
    rel = Path(rel_path)
    roots = [DRIVE_ARTIFACT_ROOT, LEGACY_DRIVE_ROOT]
    candidates: list[Path] = []
    for root in roots:
        candidates.append(root / rel_path)
        if rel_path.startswith("outputs/stage5/") and len(rel.parts) > 3:
            detected_run_id = rel.parts[2]
            after_run = Path(*rel.parts[3:])
            candidates.append(root / "outputs" / "stage5" / detected_run_id / after_run)
            candidates.append(root / detected_run_id / after_run)
        if run_id and root.exists():
            candidates.extend(path for path in root.rglob(rel.name) if run_id in path.as_posix())
    return unique_paths(candidates)


def restore_checkpoint(rel_path: str, *, run_id: str | None = None) -> Path:
    target = resolve_repo_path(rel_path)
    if target.exists():
        print(f"checkpoint already local: {target}", flush=True)
        return target
    mount_drive_if_needed()
    for candidate in checkpoint_drive_candidates(rel_path, run_id):
        if candidate.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
            print(f"restored_checkpoint={candidate} -> {target}", flush=True)
            return target
    tried = "\n".join(f"  - {path}" for path in checkpoint_drive_candidates(rel_path, run_id)[:16])
    raise FileNotFoundError(f"Could not restore Stage 3 repaired checkpoint: {rel_path}\nTried:\n{tried}")


def load_required_repair_assessment() -> dict[str, Any]:
    if not Path("/content/drive/MyDrive").exists():
        mount_drive_if_needed()
    assessment_path = latest_matching(repair_assessment_candidates())
    if assessment_path is None:
        raise FileNotFoundError(
            "Stage 4 recovery training requires a passed Stage 3 repair smoke. "
            "Run STAGE5_CURRENT_A100_TARGET=reentry_repair_smoke first."
        )
    assessment = read_json(assessment_path)
    recommendation = str(assessment.get("recommendation") or "")
    status = str(assessment.get("status") or "")
    print(f"stage3_repair_assessment={assessment_path}", flush=True)
    print(f"stage3_repair_status={status} recommendation={recommendation}", flush=True)
    if recommendation != "run_bounded_recovery_training_with_reentry_repair":
        raise RuntimeError(
            "Stage 3 repair smoke did not clear recovery training. "
            f"status={status!r} recommendation={recommendation!r}."
        )
    summary_path = assessment_path.parent / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Stage 3 assessment has no sibling summary.json: {summary_path}")
    summary = read_json(summary_path)
    checkpoint = str(summary.get("trained_checkpoint") or "")
    if not checkpoint:
        raise KeyError(f"Stage 3 summary is missing trained_checkpoint: {summary_path}")
    restored = restore_checkpoint(checkpoint, run_id=str(summary.get("run_id") or assessment_path.parent.name))
    return {
        "assessment_path": path_for_cli(assessment_path) if assessment_path.is_relative_to(ROOT) else assessment_path.as_posix(),
        "summary_path": path_for_cli(summary_path) if summary_path.is_relative_to(ROOT) else summary_path.as_posix(),
        "status": status,
        "recommendation": recommendation,
        "checkpoint": path_for_cli(restored),
    }


def is_gate_ready_trace_collection(path: Path) -> bool:
    try:
        payload = read_json(path)
    except Exception:
        return False
    return (
        payload.get("kind") == "stage5_capability_ladder_trace_collection"
        and payload.get("status") == "trace_curriculum_gate_ready"
        and isinstance(payload.get("gate"), dict)
        and payload["gate"].get("go") is True
    )


def trace_collection_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = (
        os.environ.get("STAGE5_REENTRY_RECOVERY_TRACE_SOURCE_SUMMARY")
        or os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_SOURCE_SUMMARY")
        or ""
    ).strip()
    if explicit:
        candidates.append(resolve_repo_path(explicit))
        return unique_paths(candidates)
    for root in (
        ROOT / "outputs" / "stage5",
        DRIVE_ARTIFACT_ROOT / "outputs" / "stage5",
        LEGACY_DRIVE_ROOT / "outputs" / "stage5",
    ):
        if root.exists():
            candidates.extend(sorted(root.glob("**/summary.json"), key=lambda path: path.stat().st_mtime, reverse=True))
    candidates.append(resolve_repo_path(DEFAULT_TRACE_COLLECTION))
    return unique_paths(candidates)


def resolve_trace_collection_summary() -> Path:
    for candidate in trace_collection_candidates():
        if candidate.exists() and is_gate_ready_trace_collection(candidate):
            return candidate
    raise RuntimeError(
        "No gate-ready capability-ladder trace collection summary found. "
        "Set STAGE5_REENTRY_RECOVERY_TRACE_SOURCE_SUMMARY to a gate-ready trace collection."
    )


def int_dict_max_key(payload: Any, default: int) -> int:
    from colab.reentry_recovery_config import int_dict_max_key as _int_dict_max_key

    return _int_dict_max_key(payload, default)


def mode_rows_from_counts(mode_counts: Any) -> str:
    from colab.reentry_recovery_config import mode_rows_from_counts as _mode_rows_from_counts

    return _mode_rows_from_counts(mode_counts)


def target_loop_rows_from_counts(target_loop_counts: Any) -> str:
    from colab.reentry_recovery_config import target_loop_rows_from_counts as _target_loop_rows_from_counts

    return _target_loop_rows_from_counts(target_loop_counts)


def derive_sft_env(trace_summary: Path, repair: dict[str, Any]) -> dict[str, str]:
    payload = read_json(trace_summary)
    curriculum = payload.get("curriculum") if isinstance(payload.get("curriculum"), dict) else {}
    counts = curriculum.get("counts") if isinstance(curriculum.get("counts"), dict) else {}
    collection = payload.get("collection") if isinstance(payload.get("collection"), dict) else {}
    drive_backup = payload.get("drive_backup") if isinstance(payload.get("drive_backup"), dict) else {}
    work_dir = str(curriculum.get("work_dir") or "").replace("\\", "/")
    summary_json = str(curriculum.get("summary_json") or "").replace("\\", "/")
    if not work_dir or not summary_json:
        raise RuntimeError(f"Trace collection summary is missing curriculum paths: {trace_summary}")
    positive_rows = int(counts.get("positive_sft_rows") or counts.get("typed_records") or 0)
    min_rows = int(os.environ.get("STAGE5_REENTRY_RECOVERY_MIN_TRACE_ROWS", "16"))
    if positive_rows < min_rows:
        raise RuntimeError(f"Trace collection has only {positive_rows} positive rows; need at least {min_rows}.")
    target_loop_counts = collection.get("target_loop_counts")
    if not isinstance(target_loop_counts, dict):
        target_loop_counts = counts.get("target_loop_counts")
    max_loops = int(os.environ.get("STAGE5_REENTRY_RECOVERY_MAX_LOOPS", str(int_dict_max_key(target_loop_counts, 4))))
    run_id = os.environ.get("STAGE5_REENTRY_RECOVERY_RUN_ID") or time.strftime(
        "stage5_reentry_recovery_%Y%m%d_%H%M%S"
    )
    env = os.environ.copy()
    env.update(
        {
            "MODEL_NAME": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
            "STAGE5_CURRICULUM_SFT_RUN_ID": run_id,
            "STAGE5_CURRICULUM_WORK_DIR": work_dir,
            "STAGE5_CURRICULUM_SUMMARY_JSON": summary_json,
            "STAGE5_CURRICULUM_RESUME_FROM": str(repair["checkpoint"]),
            "STAGE5_CURRICULUM_MIN_POSITIVE_ROWS": os.environ.get(
                "STAGE5_REENTRY_RECOVERY_MIN_POSITIVE_ROWS",
                str(positive_rows),
            ),
            "STAGE5_CURRICULUM_MIN_MODE_ROWS": os.environ.get(
                "STAGE5_REENTRY_RECOVERY_MIN_MODE_ROWS",
                mode_rows_from_counts(counts.get("mode_counts")),
            ),
            "STAGE5_CURRICULUM_MIN_TARGET_LOOP_ROWS": os.environ.get(
                "STAGE5_REENTRY_RECOVERY_MIN_TARGET_LOOP_ROWS",
                target_loop_rows_from_counts(target_loop_counts),
            ),
            "STAGE5_CURRICULUM_MAX_LOOPS": str(max_loops),
            "STAGE5_CURRICULUM_PHASE1_STEPS": os.environ.get("STAGE5_REENTRY_RECOVERY_STEPS", "75"),
            "STAGE5_CURRICULUM_PHASE1_LR": os.environ.get("STAGE5_REENTRY_RECOVERY_LR", "5e-6"),
            "STAGE5_CURRICULUM_PHASE1_BETA": os.environ.get("STAGE5_REENTRY_RECOVERY_BETA", "0.10"),
            "STAGE5_CURRICULUM_PHASE1_MAX_GRAD_NORM": os.environ.get(
                "STAGE5_REENTRY_RECOVERY_MAX_GRAD_NORM",
                "0.3",
            ),
            "STAGE5_CURRICULUM_HALT_TARGET_NLL_WEIGHT": os.environ.get(
                "STAGE5_REENTRY_RECOVERY_HALT_TARGET_NLL_WEIGHT",
                "5.0",
            ),
            "STAGE5_CURRICULUM_USE_LEARNED_LOOP_CONTROL": "1",
            "STAGE5_CURRICULUM_LOOP_CONTROL_CE_WEIGHT": os.environ.get(
                "STAGE5_REENTRY_RECOVERY_LOOP_CONTROL_CE_WEIGHT",
                "4.0",
            ),
            "STAGE5_CURRICULUM_SFT_REQUIRE_TARGET_LOOP_GRADIENT": os.environ.get(
                "STAGE5_REENTRY_RECOVERY_REQUIRE_TARGET_LOOP_GRADIENT",
                "1",
            ),
            "STAGE5_CURRICULUM_OPTIMIZER_MODULES": os.environ.get(
                "STAGE5_REENTRY_RECOVERY_OPTIMIZER_MODULES",
                "bridge,reentry,halt,lora",
            ),
            "STAGE5_CURRICULUM_REENTRY_RESCALE_MODE": os.environ.get(
                "STAGE5_REENTRY_RECOVERY_REENTRY_RESCALE_MODE",
                "entry_rms",
            ),
            "STAGE5_CURRICULUM_USE_REENTRY_ADAPTER": "1",
            "STAGE5_CURRICULUM_DEPTH_HINT_STYLE": os.environ.get(
                "STAGE5_REENTRY_RECOVERY_DEPTH_HINT_STYLE",
                "natural",
            ),
            "STAGE5_CURRICULUM_ALLOW_ANSWER_LINE_VERIFICATION": "1",
            "STAGE5_CURRICULUM_ALLOW_NO_DRIVE_BACKUP": os.environ.get(
                "STAGE5_REENTRY_RECOVERY_ALLOW_NO_DRIVE_BACKUP",
                "1",
            ),
            "STAGE5_CURRICULUM_SFT_COMMIT_CHECKPOINTS": os.environ.get(
                "STAGE5_REENTRY_RECOVERY_COMMIT_CHECKPOINTS",
                "0",
            ),
            "STAGE5_CURRICULUM_SFT_PUSH": "1",
            "DTYPE": os.environ.get("DTYPE", "bfloat16"),
            "ADAPTER_DTYPE": os.environ.get("ADAPTER_DTYPE", "float32"),
            "DEVICE": os.environ.get("DEVICE", "cuda"),
        }
    )
    drive_root = str(drive_backup.get("dest_root") or "").strip()
    if drive_root:
        env["STAGE5_CURRICULUM_INPUT_BACKUP_DIR"] = drive_root
    print(
        json.dumps(
            {
                "trace_summary": path_for_cli(trace_summary),
                "stage3_repair": repair,
                "positive_rows": positive_rows,
                "target_loop_counts": target_loop_counts,
                "max_loops": max_loops,
                "run_id": run_id,
                "optimizer_modules": env["STAGE5_CURRICULUM_OPTIMIZER_MODULES"],
                "reentry_rescale_mode": env["STAGE5_CURRICULUM_REENTRY_RESCALE_MODE"],
                "loop_control_ce_weight": env["STAGE5_CURRICULUM_LOOP_CONTROL_CE_WEIGHT"],
                "halt_target_nll_weight": env["STAGE5_CURRICULUM_HALT_TARGET_NLL_WEIGHT"],
            },
            indent=2,
        ),
        flush=True,
    )
    return env


def main() -> None:
    print(f"cell_version={STAGE5_REENTRY_RECOVERY_CELL_VERSION}", flush=True)
    ensure_repo()
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("Attach an L4/T4/A100/H100 GPU runtime before running Stage 4 recovery training.")
    run(["nvidia-smi"], cwd=Path("/content"))
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_stage5_curriculum_sft.py",
            "tests/test_curriculum_sft_gate.py",
            "tests/test_review_stage5_reentry.py",
        ]
    )
    repair = load_required_repair_assessment()
    trace_summary = resolve_trace_collection_summary()
    env = derive_sft_env(trace_summary, repair)
    run([sys.executable, "colab/run_stage5_curriculum_sft.py"], env=env)
    if env_flag("STAGE5_REENTRY_RECOVERY_DISCONNECT", "1"):
        print("Disconnecting Colab runtime to conserve credits after Stage 4 recovery SFT.", flush=True)
        runtime.unassign()


main()
