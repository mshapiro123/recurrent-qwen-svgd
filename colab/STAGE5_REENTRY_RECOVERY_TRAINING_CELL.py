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


STAGE5_REENTRY_RECOVERY_CELL_VERSION = "reentry_recovery_training_v4_post_reentry_health"
STAGE5_REENTRY_RECOVERY_TARGET = "reentry_recovery_training"
REPO = "mshapiro123/recurrent-qwen-svgd"
ROOT = Path("/content/recurrent-qwen-svgd")
DRIVE_ARTIFACT_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts")
LEGACY_DRIVE_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd")


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


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in {float("inf"), float("-inf")}:
        return default
    return out


def nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def attached_gpu_names() -> list[str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def require_gpu_runtime() -> None:
    names = attached_gpu_names()
    if not names:
        raise RuntimeError("Attach an L4/T4/A100/H100 GPU runtime before running Stage 4 recovery training.")
    print("stage4_gpu_runtime=" + "; ".join(names), flush=True)


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


def current_pointer_repair_assessment_candidates() -> list[Path]:
    """Return the repair assessment implied by the current source pointer.

    The normal Stage 3 publish path advances ``config/stage5_current_source_summary.txt``
    to the repair-smoke ``summary.json``. Prefer that explicit front-of-queue
    artifact over broad Drive globs so a newer failed/partial attempt does not
    accidentally shadow the repair result the planner selected. Non-repair
    pointers, such as the Stage 2 norm diagnostic, are ignored.
    """

    pointer = current_source_summary_file()
    if not pointer.exists():
        return []
    raw = pointer.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    raw_path = Path(raw.replace("\\", "/"))
    summary_candidates = [raw_path] if raw_path.is_absolute() else [
        ROOT / normalize_rel_path(raw),
        DRIVE_ARTIFACT_ROOT / normalize_rel_path(raw),
        LEGACY_DRIVE_ROOT / normalize_rel_path(raw),
    ]
    assessments: list[Path] = []
    for summary_path in summary_candidates:
        if summary_path.name != "summary.json" or not summary_path.exists():
            continue
        try:
            payload = read_json(summary_path)
        except Exception:
            continue
        if payload.get("kind") == "stage5_reentry_repair_smoke":
            assessments.append(summary_path.with_name("reentry_assessment.json"))
    return unique_paths(assessments)


def repair_assessment_candidates() -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("STAGE5_REENTRY_RECOVERY_REPAIR_ASSESSMENT", "").strip()
    if override:
        rel = normalize_rel_path(override)
        for candidate in (ROOT / rel, DRIVE_ARTIFACT_ROOT / rel, LEGACY_DRIVE_ROOT / rel):
            candidates.append(candidate)
            if candidate.name == "summary.json":
                candidates.append(candidate.with_name("reentry_assessment.json"))
    candidates.extend(current_pointer_repair_assessment_candidates())
    for root in (
        ROOT / "outputs" / "stage5",
        DRIVE_ARTIFACT_ROOT / "outputs" / "stage5",
        LEGACY_DRIVE_ROOT / "outputs" / "stage5",
    ):
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
    from colab.reentry_recovery_config import repair_assessment_recovery_block_reason

    if not Path("/content/drive/MyDrive").exists():
        mount_drive_if_needed()
    assessment_path = latest_matching(current_pointer_repair_assessment_candidates())
    if assessment_path is not None:
        print(f"stage3_repair_assessment_source=current_pointer assessment={assessment_path}", flush=True)
    else:
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
    block_reason = repair_assessment_recovery_block_reason(assessment)
    if block_reason:
        raise RuntimeError(block_reason)
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
        "metrics": assessment.get("metrics", {}),
        "reason": assessment.get("reason"),
        "checkpoint": path_for_cli(restored),
    }


def resolve_trace_collection_summary() -> Path:
    from colab.review_stage5_recovery_curriculum import (
        resolve_trace_collection_summary as _resolve_trace_collection_summary,
    )

    explicit = (
        os.environ.get("STAGE5_REENTRY_RECOVERY_TRACE_SOURCE_SUMMARY")
        or os.environ.get("STAGE5_TRACED_CAPABILITY_SFT_SOURCE_SUMMARY")
        or ""
    ).strip()
    path = _resolve_trace_collection_summary(
        explicit=explicit,
        root=ROOT,
        extra_roots=(DRIVE_ARTIFACT_ROOT, LEGACY_DRIVE_ROOT),
    )
    print(f"stage4_trace_collection_summary={path_for_cli(path)}", flush=True)
    return path


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
    child_run_id = os.environ.get("STAGE5_REENTRY_RECOVERY_CHILD_RUN_ID") or f"{run_id}_curriculum_sft"
    env = os.environ.copy()
    env.update(
        {
            "MODEL_NAME": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
            "STAGE5_CURRICULUM_LAYER_SPLIT": os.environ.get(
                "STAGE5_REENTRY_RECOVERY_LAYER_SPLIT",
                os.environ.get("STAGE5_RECURRENT_LAYER_SPLIT", os.environ.get("LAYER_SPLIT", "auto")),
            ),
            "STAGE5_REENTRY_RECOVERY_WRAPPER_RUN_ID": run_id,
            "STAGE5_CURRICULUM_SFT_RUN_ID": child_run_id,
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
                "child_run_id": child_run_id,
                "optimizer_modules": env["STAGE5_CURRICULUM_OPTIMIZER_MODULES"],
                "layer_split": env["STAGE5_CURRICULUM_LAYER_SPLIT"],
                "reentry_rescale_mode": env["STAGE5_CURRICULUM_REENTRY_RESCALE_MODE"],
                "loop_control_ce_weight": env["STAGE5_CURRICULUM_LOOP_CONTROL_CE_WEIGHT"],
                "halt_target_nll_weight": env["STAGE5_CURRICULUM_HALT_TARGET_NLL_WEIGHT"],
            },
            indent=2,
        ),
        flush=True,
    )
    return env


def post_reentry_health_checks(payload: dict[str, Any]) -> dict[str, Any]:
    bridge = payload.get("bridge") if isinstance(payload.get("bridge"), dict) else {}
    bridge_live = payload.get("bridge_gradient_liveness") if isinstance(payload.get("bridge_gradient_liveness"), dict) else {}
    adapter = payload.get("reentry_adapter") if isinstance(payload.get("reentry_adapter"), dict) else {}
    adapter_live = (
        payload.get("reentry_adapter_gradient_liveness")
        if isinstance(payload.get("reentry_adapter_gradient_liveness"), dict)
        else {}
    )
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), dict) else {}
    loop_summary = aggregate.get("loop_summary") if isinstance(aggregate.get("loop_summary"), dict) else {}

    min_gate = finite_float(os.environ.get("STAGE5_REENTRY_RECOVERY_HEALTH_MIN_BRIDGE_GATE", "0.05"), 0.05)
    max_loop8_output = finite_float(os.environ.get("STAGE5_REENTRY_RECOVERY_HEALTH_MAX_LOOP8_OUTPUT_OVER_ENTRY_RMS", "3.0"), 3.0)
    gate = finite_float(bridge.get("bridge_gate"))
    bridge_delta = finite_float(bridge.get("sample_bridge_delta_rms"))
    bridge_weight_grad = finite_float(bridge_live.get("weight_grad_rms"))
    bridge_bias_grad = finite_float(bridge_live.get("bias_grad_rms"))
    adapter_delta = finite_float(adapter.get("sample_adapter_delta_rms"))
    adapter_scale_grad = finite_float(adapter_live.get("scale_grad_rms"))
    adapter_bias_grad = finite_float(adapter_live.get("bias_grad_rms"))
    loop8 = loop_summary.get("8") if isinstance(loop_summary.get("8"), dict) else {}
    loop8_output_over_entry = finite_float(loop8.get("output_over_entry_rms"), 0.0)
    loop8_output_over_input = finite_float(loop8.get("output_over_input_rms"), 0.0)

    issues: list[str] = []
    if abs(gate) < min_gate:
        issues.append("bridge_gate_inactive_after_recovery")
    if bridge_weight_grad <= 0.0 or bridge_bias_grad <= 0.0:
        issues.append("bridge_gradient_not_live_after_recovery")
    if bridge_delta <= 0.0:
        issues.append("bridge_delta_zero_after_recovery")
    if adapter_scale_grad <= 0.0 or adapter_bias_grad <= 0.0:
        issues.append("reentry_adapter_gradient_not_live_after_recovery")
    if adapter_delta <= 0.0:
        issues.append("reentry_adapter_delta_zero_after_recovery")
    if loop8_output_over_entry <= 0.0 or loop8_output_over_entry > max_loop8_output:
        issues.append("loop8_output_over_entry_rms_unbounded_after_recovery")

    return {
        "status": "reentry_health_sane" if not issues else "reentry_health_needs_review",
        "issues": issues,
        "thresholds": {
            "min_bridge_gate_abs": min_gate,
            "max_loop8_output_over_entry_rms": max_loop8_output,
        },
        "metrics": {
            "bridge_gate": gate,
            "bridge_delta_rms": bridge_delta,
            "bridge_weight_grad_rms": bridge_weight_grad,
            "bridge_bias_grad_rms": bridge_bias_grad,
            "reentry_adapter_delta_rms": adapter_delta,
            "reentry_adapter_scale_grad_rms": adapter_scale_grad,
            "reentry_adapter_bias_grad_rms": adapter_bias_grad,
            "loop8_output_over_entry_rms": loop8_output_over_entry,
            "loop8_output_over_input_rms": loop8_output_over_input,
            "mean_exit_over_entry_rms": finite_float(aggregate.get("mean_exit_over_entry_rms")),
            "subspace_overlap": finite_float(nested(aggregate, "entry_exit_subspace", "overlap")),
        },
    }


def run_post_reentry_health_probe(
    *,
    checkpoint: str,
    env: dict[str, str],
    out_dir: Path,
) -> dict[str, Any]:
    output_json = out_dir / "post_reentry_drift.json"
    output_jsonl = out_dir / "post_reentry_drift.jsonl"
    max_loops = os.environ.get("STAGE5_REENTRY_RECOVERY_HEALTH_MAX_LOOPS", "8")
    limit = os.environ.get("STAGE5_REENTRY_RECOVERY_HEALTH_PROMPT_LIMIT", "6")
    cmd = [
        sys.executable,
        "eval/eval_reentry_drift.py",
        "--model_name",
        env.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "--checkpoint",
        checkpoint,
        "--split",
        env.get("STAGE5_CURRICULUM_LAYER_SPLIT", "auto"),
        "--max_loops",
        max_loops,
        "--max_length",
        os.environ.get("STAGE5_REENTRY_RECOVERY_HEALTH_MAX_LENGTH", "256"),
        "--limit",
        limit,
        "--reentry_rescale_mode",
        env.get("STAGE5_CURRICULUM_REENTRY_RESCALE_MODE", "entry_rms"),
        "--dtype",
        env.get("DTYPE", "bfloat16"),
        "--adapter_dtype",
        env.get("ADAPTER_DTYPE", "float32"),
        "--device",
        env.get("DEVICE", "cuda"),
        "--output_json",
        path_for_cli(output_json),
        "--output_jsonl",
        path_for_cli(output_jsonl),
    ]
    if env.get("STAGE5_CURRICULUM_USE_REENTRY_ADAPTER", "0").strip().lower() in {"1", "true", "yes", "y", "on"}:
        cmd.append("--use_reentry_adapter")
    run(cmd, env=env)
    payload = read_json(output_json)
    health = post_reentry_health_checks(payload)
    health_path = out_dir / "post_reentry_health_checks.json"
    health_path.write_text(json.dumps(health, indent=2), encoding="utf-8")
    print("post_reentry_health_checks=" + json.dumps(health, indent=2), flush=True)
    return {
        "summary_path": path_for_cli(output_json),
        "records_path": path_for_cli(output_jsonl),
        "health_path": path_for_cli(health_path),
        "health_checks": health,
    }


def current_source_summary_file() -> Path:
    return ROOT / "config" / "stage5_current_source_summary.txt"


def current_source_summary_path() -> Path:
    pointer = current_source_summary_file()
    if not pointer.exists():
        raise FileNotFoundError(pointer)
    value = pointer.read_text(encoding="utf-8").strip()
    if not value:
        raise FileNotFoundError("config/stage5_current_source_summary.txt is empty")
    return resolve_repo_path(value)


def write_reentry_recovery_wrapper_summary(
    *,
    repair: dict[str, Any],
    trace_summary: Path,
    env: dict[str, str],
    post_reentry_probe: dict[str, Any],
) -> Path:
    from colab.stage5_publish_utils import update_current_source_summary

    child_summary = current_source_summary_path()
    child_payload = read_json(child_summary)
    wrapper_run_id = env["STAGE5_REENTRY_RECOVERY_WRAPPER_RUN_ID"]
    run_dir = ROOT / "outputs" / "stage5" / wrapper_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = str(child_payload.get("phase1_checkpoint") or child_payload.get("checkpoint") or "")
    validation_checks = child_payload.get("validation_checks") if isinstance(child_payload.get("validation_checks"), dict) else {}
    validation_status = str(validation_checks.get("status") or child_payload.get("status") or "")
    summary = {
        "kind": "stage5_reentry_recovery_training",
        "run_id": wrapper_run_id,
        "cell_version": STAGE5_REENTRY_RECOVERY_CELL_VERSION,
        "status": validation_status or "unknown",
        "passed": validation_status == "validation_sane",
        "child_summary": path_for_cli(child_summary),
        "trace_summary": path_for_cli(trace_summary),
        "stage3_repair": repair,
        "checkpoint": checkpoint,
        "phase1_checkpoint": checkpoint,
        "dataset": child_payload.get("dataset", {}),
        "config": child_payload.get("config", {}),
        "phase1_val": child_payload.get("phase1_val", {}),
        "phase1_val_by_mode": child_payload.get("phase1_val_by_mode", {}),
        "phase1_val_by_target_loop": child_payload.get("phase1_val_by_target_loop", {}),
        "validation_checks": validation_checks,
        "post_reentry_drift": {
            "summary_path": post_reentry_probe.get("summary_path"),
            "records_path": post_reentry_probe.get("records_path"),
        },
        "post_reentry_health_checks": post_reentry_probe.get("health_checks", {}),
        "next_step": (
            "Run debiased_benchmark_suite against this repaired deterministic recurrent checkpoint before "
            "dense control, breadth diagnostics, particles, or SVGD."
        ),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 Re-entry Recovery Training - {wrapper_run_id}",
        "",
        f"- Cell version: `{STAGE5_REENTRY_RECOVERY_CELL_VERSION}`",
        f"- Child curriculum summary: `{summary['child_summary']}`",
        f"- Trace summary: `{summary['trace_summary']}`",
        f"- Stage 3 repair assessment: `{repair.get('assessment_path')}`",
        f"- Status: `{summary['status']}`",
        f"- Passed: `{summary['passed']}`",
        f"- Checkpoint: `{checkpoint}`",
        "",
        "## Validation Checks",
        "```json",
        json.dumps(validation_checks, indent=2),
        "```",
        "",
        "## Post-Recovery Re-entry Health",
        f"- Drift summary: `{post_reentry_probe.get('summary_path')}`",
        f"- Health status: `{post_reentry_probe.get('health_checks', {}).get('status')}`",
        "```json",
        json.dumps(post_reentry_probe.get("health_checks", {}), indent=2),
        "```",
        "",
        "## Next Step",
        summary["next_step"],
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    update_current_source_summary(ROOT, summary_path)
    print((run_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    return summary_path


def publish_reentry_recovery_wrapper(summary_path: Path) -> None:
    from colab.stage5_publish_utils import publishable_artifact_paths

    run_dir = summary_path.parent
    paths = publishable_artifact_paths(run_dir)
    pointer = current_source_summary_file()
    if pointer.exists():
        paths.append(pointer)
    for path in paths:
        run(["git", "add", "-f", path_for_cli(path)], check=False)
    if run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 0:
        print("No Stage 5 re-entry recovery wrapper outputs changed.", flush=True)
        return
    run(["git", "commit", "-m", f"Record Stage 5 re-entry recovery wrapper {summary_path.parent.name} [skip ci]"])
    pushed = run(["git", "push", "origin", "main"], check=False)
    if pushed.returncode != 0:
        print("Initial wrapper push failed; attempting one fast rebase and retry.", flush=True)
        run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
        run(["git", "push", "origin", "main"])


def main() -> None:
    print(f"cell_version={STAGE5_REENTRY_RECOVERY_CELL_VERSION}", flush=True)
    require_gpu_runtime()
    ensure_repo()
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
    child_summary = current_source_summary_path()
    child_payload = read_json(child_summary)
    checkpoint = str(child_payload.get("phase1_checkpoint") or child_payload.get("checkpoint") or "")
    if not checkpoint:
        raise RuntimeError(f"Child Stage 4 curriculum summary is missing checkpoint: {child_summary}")
    wrapper_run_id = env["STAGE5_REENTRY_RECOVERY_WRAPPER_RUN_ID"]
    wrapper_run_dir = ROOT / "outputs" / "stage5" / wrapper_run_id
    wrapper_run_dir.mkdir(parents=True, exist_ok=True)
    post_reentry_probe = run_post_reentry_health_probe(
        checkpoint=checkpoint,
        env=env,
        out_dir=wrapper_run_dir,
    )
    wrapper_summary = write_reentry_recovery_wrapper_summary(
        repair=repair,
        trace_summary=trace_summary,
        env=env,
        post_reentry_probe=post_reentry_probe,
    )
    publish_reentry_recovery_wrapper(wrapper_summary)
    if env_flag("STAGE5_REENTRY_RECOVERY_DISCONNECT", "1"):
        print("Disconnecting Colab runtime to conserve credits after Stage 4 recovery SFT.", flush=True)
        runtime.unassign()


main()
