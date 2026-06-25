"""Colab launcher: Stage 5 trainable re-entry bridge repair smoke.

This is Stage 3 after the read-only drift diagnostic and eval-only RMS
normalization check. It runs a tiny bounded continuation that explicitly revives
the bridge as an identity-preserving but gradient-live module, trains for a few
steps, and compares pre/post re-entry drift. It is not a recovery-scale run.
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

STAGE5_REENTRY_REPAIR_SMOKE_CELL_VERSION = "stage5_reentry_repair_smoke_v1_trainable"
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

SMOKE_ROWS = [
    {
        "prompt": "Solve exactly. If a train travels 120 miles in 3 hours, what is its average speed?\n",
        "completion": "The average speed is 120 / 3 = 40 miles per hour.",
        "target_loop_count": 2,
    },
    {
        "prompt": "Solve exactly. The pharmacy has 20 tubs and needs 100 total. It buys one quarter of the remaining need from a new vendor. How many tubs come from the usual vendor?\n",
        "completion": "It needs 80 more tubs. One quarter of 80 is 20, so the usual vendor supplies 60 tubs.",
        "target_loop_count": 3,
    },
    {
        "prompt": "Solve exactly. What is 17 + 28?\n",
        "completion": "17 + 28 = 45.",
        "target_loop_count": 1,
    },
    {
        "prompt": "Solve exactly. A rectangle is 7 units by 6 units. What is its area?\n",
        "completion": "Area is length times width: 7 * 6 = 42 square units.",
        "target_loop_count": 2,
    },
    {
        "prompt": "Name one valid strategy for solving a Sudoku puzzle.\n",
        "completion": "One valid strategy is elimination: remove impossible digits using row, column, and box constraints.",
        "target_loop_count": 1,
    },
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


def has_valid_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


def has_valid_jsonl(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return False
        for line in lines:
            json.loads(line)
    except (OSError, json.JSONDecodeError):
        return False
    return True


def drive_backup_enabled() -> bool:
    return os.environ.get("STAGE5_REENTRY_REPAIR_DRIVE_BACKUP", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def incremental_backup(out_dir: Path) -> None:
    if not drive_backup_enabled() or not Path("/content/drive/MyDrive").exists() or not out_dir.exists():
        return
    backup_dir = DRIVE_ARTIFACT_ROOT / "outputs" / "stage5" / out_dir.name
    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(out_dir, backup_dir)
    print(f"drive_backup={backup_dir}", flush=True)


def restore_incremental_backup(out_dir: Path) -> None:
    if os.environ.get("STAGE5_REENTRY_REPAIR_RESTORE_INCREMENTAL_BACKUP", "1").strip().lower() not in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return
    backup_dir = DRIVE_ARTIFACT_ROOT / "outputs" / "stage5" / out_dir.name
    if not backup_dir.exists():
        return
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(backup_dir, out_dir, dirs_exist_ok=True)
    print(f"restored_repair_incremental_backup={backup_dir} -> {out_dir}", flush=True)


def latest_matching(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return sorted(existing, key=lambda path: path.as_posix())[-1]


def norm_assessment_candidates() -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("STAGE5_REENTRY_REPAIR_NORM_ASSESSMENT", "").strip()
    if override:
        candidates.append(ROOT / normalize_rel_path(override))
        candidates.append(DRIVE_ARTIFACT_ROOT / normalize_rel_path(override))
        candidates.append(Path("/content/drive/MyDrive/recurrent-qwen-svgd") / normalize_rel_path(override))

    for root in (ROOT / "outputs" / "stage5", DRIVE_ARTIFACT_ROOT / "outputs" / "stage5"):
        if root.exists():
            candidates.extend(sorted(root.glob("stage5_reentry_norm_*/reentry_assessment.json")))
    return unique_paths(candidates)


def load_required_norm_assessment() -> dict[str, object] | None:
    required = os.environ.get("STAGE5_REENTRY_REPAIR_REQUIRE_NORM_PASS", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if not required:
        print("Stage 2 norm assessment gate disabled by STAGE5_REENTRY_REPAIR_REQUIRE_NORM_PASS=0.", flush=True)
        return None

    local_candidate = latest_matching(norm_assessment_candidates())
    if local_candidate is None and not Path("/content/drive/MyDrive").exists():
        print("Mounting Drive to find Stage 2 re-entry norm assessment.", flush=True)
        drive.mount("/content/drive", force_remount=False)
        local_candidate = latest_matching(norm_assessment_candidates())

    if local_candidate is None:
        raise FileNotFoundError(
            "Stage 3 repair smoke requires a passing Stage 2 re-entry norm assessment. "
            "Run STAGE5_CURRENT_A100_TARGET=reentry_norm_diagnostic first, or set "
            "STAGE5_REENTRY_REPAIR_REQUIRE_NORM_PASS=0 for an intentional override."
        )

    assessment = read_json(local_candidate)
    recommendation = str(assessment.get("recommendation", ""))
    status = str(assessment.get("status", ""))
    print(f"stage2_norm_assessment={local_candidate}", flush=True)
    print(f"stage2_norm_status={status} recommendation={recommendation}", flush=True)
    if recommendation != "run_reentry_repair_smoke":
        raise RuntimeError(
            "Stage 2 re-entry norm assessment did not recommend repair smoke. "
            f"status={status!r} recommendation={recommendation!r}. Review before spending GPU."
        )
    return {
        "path": local_candidate.relative_to(ROOT).as_posix() if local_candidate.is_relative_to(ROOT) else local_candidate.as_posix(),
        "status": status,
        "recommendation": recommendation,
        "reason": assessment.get("reason"),
        "metrics": assessment.get("metrics", {}),
    }


def write_smoke_data(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in SMOKE_ROWS) + "\n", encoding="utf-8")


def write_config(path: Path, *, checkpoint: str, out_dir: Path) -> dict[str, object]:
    cfg = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "dtype": os.environ.get("STAGE5_REENTRY_REPAIR_DTYPE", "bfloat16"),
        "adapter_dtype": "float32",
        "layer_split": "6,18",
        "max_length": int(os.environ.get("STAGE5_REENTRY_REPAIR_MAX_LENGTH", "256")),
        "max_loops": int(os.environ.get("STAGE5_REENTRY_REPAIR_MAX_LOOPS", "4")),
        "initial_halt_prob": 0.15,
        "beta": 0.08,
        "halt_target_nll_weight": float(os.environ.get("STAGE5_REENTRY_REPAIR_HALT_NLL_WEIGHT", "0.05")),
        "batch_size": 1,
        "learning_rate": float(os.environ.get("STAGE5_REENTRY_REPAIR_LR", "1e-5")),
        "weight_decay": 0.0,
        "max_grad_norm": 0.3,
        "max_steps": int(os.environ.get("STAGE5_REENTRY_REPAIR_MAX_STEPS", "25")),
        "log_every": 5,
        "train_on_prompt": False,
        "optimizer_modules": os.environ.get("STAGE5_REENTRY_REPAIR_OPTIMIZER_MODULES", "bridge,reentry,halt"),
        "resume_from": checkpoint,
        "bridge_reset_identity": True,
        "bridge_gate_override": 1.0,
        "reentry_rescale_mode": "entry_rms",
        "use_reentry_adapter": True,
        "output_dir": str(out_dir.relative_to(ROOT)),
        "lora": {"enabled": True, "rank": 8, "alpha": 16, "dropout": 0.0},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def run_drift(label: str, checkpoint: str, out_dir: Path) -> Path:
    out_json = out_dir / f"reentry_drift_{label}.json"
    out_jsonl = out_dir / f"reentry_drift_{label}.jsonl"
    if has_valid_json(out_json) and has_valid_jsonl(out_jsonl):
        print(f"resume_skip=reentry_drift_{label}", flush=True)
        return out_json
    run(
        [
            sys.executable,
            "eval/eval_reentry_drift.py",
            "--checkpoint",
            checkpoint,
            "--prompts_jsonl",
            os.environ.get("STAGE5_REENTRY_REPAIR_PROMPTS", "eval/smoke_exact_tasks_v2.jsonl"),
            "--limit",
            os.environ.get("STAGE5_REENTRY_REPAIR_LIMIT", "8"),
            "--max_loops",
            os.environ.get("STAGE5_REENTRY_REPAIR_DRIFT_MAX_LOOPS", "8"),
            "--max_length",
            os.environ.get("STAGE5_REENTRY_REPAIR_MAX_LENGTH", "256"),
            "--reentry_rescale_mode",
            "entry_rms",
            "--use_reentry_adapter",
            "--dtype",
            os.environ.get("STAGE5_REENTRY_REPAIR_DTYPE", "bfloat16"),
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("STAGE5_REENTRY_REPAIR_DEVICE", "cuda"),
            "--output_json",
            str(out_json.relative_to(ROOT)),
            "--output_jsonl",
            str(out_jsonl.relative_to(ROOT)),
        ]
    )
    return out_json


def write_preservation_tasks(out_path: Path) -> Path:
    source = ROOT / os.environ.get("STAGE5_REENTRY_REPAIR_PRESERVE_TASKS", "eval/smoke_exact_tasks_v2.jsonl")
    limit = int(os.environ.get("STAGE5_REENTRY_REPAIR_PRESERVE_LIMIT", "6"))
    rows = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = rows[:limit]
    if not selected:
        raise ValueError(f"No loop-1 preservation tasks found in {source}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return out_path


def summarize_loop1_preservation(jsonl_path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_label: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_label.setdefault(str(row.get("label")), []).append(row)

    def stats(label: str) -> dict[str, object]:
        label_rows = by_label.get(label, [])
        by_task_seed: dict[tuple[object, object], list[dict[str, object]]] = {}
        for row in label_rows:
            by_task_seed.setdefault((row.get("seed"), row.get("task")), []).append(row)
        return {
            "label": label,
            "task_groups": len(by_task_seed),
            "best_hits": sum(any(bool(item.get("hit")) for item in group) for group in by_task_seed.values()),
            "candidate_hits": sum(bool(row.get("hit")) for row in label_rows),
            "total_candidates": len(label_rows),
        }

    source = stats("Phase 1 K=1")
    trained = stats("Phase 2 K=1")
    return {
        "jsonl": jsonl_path.relative_to(ROOT).as_posix(),
        "source": source,
        "trained": trained,
        "best_hits_delta_trained_minus_source": trained["best_hits"] - source["best_hits"],
        "candidate_hits_delta_trained_minus_source": trained["candidate_hits"] - source["candidate_hits"],
    }


def run_loop1_preservation(source_checkpoint: str, trained_checkpoint: str, out_dir: Path) -> dict[str, object]:
    tasks_jsonl = write_preservation_tasks(out_dir / "loop1_preservation_tasks.jsonl")
    out_jsonl = out_dir / "loop1_preservation.jsonl"
    if has_valid_jsonl(out_jsonl):
        print("resume_skip=loop1_preservation", flush=True)
        return summarize_loop1_preservation(out_jsonl)
    run(
        [
            sys.executable,
            "eval/eval_best_of_k_jsonl.py",
            "--tasks_jsonl",
            str(tasks_jsonl.relative_to(ROOT)),
            "--phase1_checkpoint",
            source_checkpoint,
            "--phase2_checkpoint",
            trained_checkpoint,
            "--phase2_num_trajectories",
            "1",
            "--max_loops",
            "1",
            "--max_new_tokens",
            os.environ.get("STAGE5_REENTRY_REPAIR_PRESERVE_MAX_NEW_TOKENS", "80"),
            "--temperature",
            "0.0",
            "--dtype",
            os.environ.get("STAGE5_REENTRY_REPAIR_DTYPE", "bfloat16"),
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("STAGE5_REENTRY_REPAIR_DEVICE", "cuda"),
            "--compact",
            "--output_jsonl",
            str(out_jsonl.relative_to(ROOT)),
        ]
    )
    return summarize_loop1_preservation(out_jsonl)


def bridge_summary(payload: dict[str, object]) -> dict[str, object]:
    bridge = payload.get("bridge") if isinstance(payload.get("bridge"), dict) else {}
    live = payload.get("bridge_gradient_liveness") if isinstance(payload.get("bridge_gradient_liveness"), dict) else {}
    loops = (payload.get("aggregate") or {}).get("loop_summary", {}) if isinstance(payload.get("aggregate"), dict) else {}
    return {
        "bridge_gate": bridge.get("bridge_gate"),
        "proj_identity_max_abs_diff": bridge.get("proj_identity_max_abs_diff"),
        "proj_bias_max_abs": bridge.get("proj_bias_max_abs"),
        "bridge_delta_rms": bridge.get("sample_bridge_delta_rms"),
        "weight_grad_rms": live.get("weight_grad_rms"),
        "bias_grad_rms": live.get("bias_grad_rms"),
        "loop4_output_over_input_rms": (loops.get("4") or {}).get("output_over_input_rms") if isinstance(loops, dict) else None,
        "loop8_output_over_input_rms": (loops.get("8") or {}).get("output_over_input_rms") if isinstance(loops, dict) else None,
    }


def write_markdown(summary: dict[str, object], path: Path) -> None:
    lines = [
        f"# Stage 5 Re-entry Repair Smoke - {summary['run_id']}",
        "",
        f"- Source checkpoint: `{summary['source_checkpoint']}`",
        f"- Trained checkpoint: `{summary['trained_checkpoint']}`",
        f"- Cell version: `{STAGE5_REENTRY_REPAIR_SMOKE_CELL_VERSION}`",
        f"- Max steps: `{summary['config']['max_steps']}`",
        f"- Optimizer modules: `{summary['config']['optimizer_modules']}`",
        f"- Re-entry mode: `{summary['config']['reentry_rescale_mode']}`",
        f"- Use re-entry adapter: `{summary['config'].get('use_reentry_adapter')}`",
        "",
        "## Bridge Liveness",
        "| stage | gate | proj identity max diff | proj bias max | bridge delta RMS | weight grad RMS | bias grad RMS | loop4 out/in RMS | loop8 out/in RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ("pre", "post"):
        row = summary[f"{stage}_bridge"]
        lines.append(
            f"| {stage} | {row.get('bridge_gate')} | {row.get('proj_identity_max_abs_diff')} | "
            f"{row.get('proj_bias_max_abs')} | {row.get('bridge_delta_rms')} | "
            f"{row.get('weight_grad_rms')} | {row.get('bias_grad_rms')} | "
            f"{row.get('loop4_output_over_input_rms')} | {row.get('loop8_output_over_input_rms')} |"
        )
    lines.extend(
        [
            "",
            "## Re-entry Adapter",
            "| stage | scale identity max diff | bias max abs | adapter delta RMS | scale grad RMS | bias grad RMS |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in ("pre", "post"):
        row = summary[f"{stage}_reentry_adapter"]
        live = summary[f"{stage}_reentry_adapter_liveness"]
        lines.append(
            f"| {stage} | {row.get('scale_identity_max_abs_diff')} | {row.get('bias_max_abs')} | "
            f"{row.get('sample_adapter_delta_rms')} | {live.get('scale_grad_rms')} | {live.get('bias_grad_rms')} |"
        )
    preservation = summary.get("loop1_preservation") if isinstance(summary.get("loop1_preservation"), dict) else {}
    source = preservation.get("source") if isinstance(preservation.get("source"), dict) else {}
    trained = preservation.get("trained") if isinstance(preservation.get("trained"), dict) else {}
    lines.extend(
        [
            "",
            "## Loop-1 Preservation",
            f"- Source loop-1 best hits: `{source.get('best_hits')}/{source.get('task_groups')}`",
            f"- Trained loop-1 best hits: `{trained.get('best_hits')}/{trained.get('task_groups')}`",
            f"- Best-hit delta: `{preservation.get('best_hits_delta_trained_minus_source')}`",
            f"- Candidate-hit delta: `{preservation.get('candidate_hits_delta_trained_minus_source')}`",
            "",
            "## Readout Pause",
            "This run intentionally stops after Stage 3. Review bridge movement and loop behavior before recovery training.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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
    run(["git", "commit", "-m", f"Record Stage 5 re-entry repair smoke {run_id} [skip ci]"])
    if os.environ.get("STAGE5_REENTRY_REPAIR_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}:
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
    print(f"cell_version={STAGE5_REENTRY_REPAIR_SMOKE_CELL_VERSION}", flush=True)
    run_id = os.environ.get("STAGE5_REENTRY_REPAIR_RUN_ID") or time.strftime("stage5_reentry_repair_smoke_%Y%m%d_%H%M%S")
    checkpoint_override = os.environ.get("STAGE5_REENTRY_REPAIR_CHECKPOINT")
    checkpoint = checkpoint_override or DEFAULT_CHECKPOINT
    disconnect = os.environ.get("STAGE5_REENTRY_REPAIR_DISCONNECT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    ensure_repo()
    norm_assessment = load_required_norm_assessment()
    run(["nvidia-smi"], cwd=Path("/content"))
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_bridge.py",
            "tests/test_eval_reentry_drift.py",
            "tests/test_recurrent_wrapper_tiny.py",
        ]
    )

    checkpoint_path = restore_checkpoint(checkpoint, allow_fallback=checkpoint_override is None)
    checkpoint = checkpoint_path.relative_to(ROOT).as_posix()
    out_dir = ROOT / "outputs" / "stage5" / run_id
    restore_incremental_backup(out_dir)
    diag_dir = out_dir / "reentry_repair_smoke"
    diag_dir.mkdir(parents=True, exist_ok=True)

    pre_drift_path = run_drift("pre", checkpoint, diag_dir)
    incremental_backup(out_dir)
    train_jsonl = diag_dir / "reentry_repair_smoke_train.jsonl"
    config_path = diag_dir / "reentry_repair_smoke_config.yaml"
    train_out_dir = out_dir / "phase1_reentry_repair"
    write_smoke_data(train_jsonl)
    config = write_config(config_path, checkpoint=checkpoint, out_dir=train_out_dir)

    trained_checkpoint = train_out_dir / f"phase1_step_{config['max_steps']}.pt"
    if trained_checkpoint.exists():
        print(f"resume_skip=train_phase1_ponder checkpoint={trained_checkpoint}", flush=True)
    else:
        train_proc = run(
            [
                sys.executable,
                "training/train_phase1_ponder.py",
                "--config",
                str(config_path.relative_to(ROOT)),
                "--train_jsonl",
                str(train_jsonl.relative_to(ROOT)),
                "--device",
                os.environ.get("STAGE5_REENTRY_REPAIR_DEVICE", "cuda"),
            ]
        )
        (diag_dir / "train_phase1_ponder.log").write_text(train_proc.stdout, encoding="utf-8")
        incremental_backup(out_dir)
    if not trained_checkpoint.exists():
        raise FileNotFoundError(f"Expected trained checkpoint missing: {trained_checkpoint}")
    trained_checkpoint_rel = trained_checkpoint.relative_to(ROOT).as_posix()
    post_drift_path = run_drift("post", trained_checkpoint_rel, diag_dir)
    loop1_preservation = run_loop1_preservation(checkpoint, trained_checkpoint_rel, diag_dir)
    incremental_backup(out_dir)

    pre_payload = json.loads(pre_drift_path.read_text(encoding="utf-8"))
    post_payload = json.loads(post_drift_path.read_text(encoding="utf-8"))
    summary = {
        "kind": "stage5_reentry_repair_smoke",
        "run_id": run_id,
        "cell_version": STAGE5_REENTRY_REPAIR_SMOKE_CELL_VERSION,
        "source_checkpoint": checkpoint,
        "trained_checkpoint": trained_checkpoint_rel,
        "config": config,
        "paths": {
            "pre_drift": pre_drift_path.relative_to(ROOT).as_posix(),
            "post_drift": post_drift_path.relative_to(ROOT).as_posix(),
            "train_jsonl": train_jsonl.relative_to(ROOT).as_posix(),
            "train_config": config_path.relative_to(ROOT).as_posix(),
            "train_log": (diag_dir / "train_phase1_ponder.log").relative_to(ROOT).as_posix(),
            "loop1_preservation": loop1_preservation["jsonl"],
        },
        "pre_bridge": bridge_summary(pre_payload),
        "post_bridge": bridge_summary(post_payload),
        "pre_reentry_adapter": pre_payload.get("reentry_adapter", {}),
        "post_reentry_adapter": post_payload.get("reentry_adapter", {}),
        "pre_reentry_adapter_liveness": pre_payload.get("reentry_adapter_gradient_liveness", {}),
        "post_reentry_adapter_liveness": post_payload.get("reentry_adapter_gradient_liveness", {}),
        "loop1_preservation": loop1_preservation,
        "stage2_norm_assessment": norm_assessment,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, out_dir / "summary.md")
    print((out_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    run_reentry_assessment(out_dir)

    incremental_backup(out_dir)

    publish_outputs(out_dir, run_id)

    if disconnect:
        print("Disconnecting Colab runtime to conserve credits after Stage 3 smoke.", flush=True)
        runtime.unassign()


main()
