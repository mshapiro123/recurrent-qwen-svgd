"""Sweep ARC geometry TTA variants for base and recurrent checkpoints.

This runner is evaluation-only. It answers a focused benchmark question:

Does ARC-style geometry test-time augmentation help the recovered recurrent
model, and does it help more or less than it helps the unmodified base Qwen?

Each arm calls ``eval/eval_arc_agi.py`` with a different ``--geometry_tta``
setting and records exact-grid metrics.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = os.environ.get("STAGE5_ARC_AGI_TTA_SWEEP_RUN_ID") or time.strftime("stage5_arc_agi_tta_sweep_%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "outputs" / "stage5" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

BASE_RUN_ID = os.environ.get("STAGE5_BASE_RUN_ID", "stage4_opus_a100_20260620")
BASE_RUN_DIR = ROOT / "outputs" / "stage4" / BASE_RUN_ID
DEFAULT_PHASE1_CKPT = BASE_RUN_DIR / "phase1" / "phase1_step_500.pt"

CURRICULUM_SUMMARY = os.environ.get("STAGE5_ARC_AGI_CURRICULUM_SUMMARY", "")
RECOVERED_CKPT = os.environ.get("STAGE5_ARC_AGI_RECOVERED_CKPT", "")
PHASE1_START_CKPT = os.environ.get("STAGE5_PHASE1_CKPT", "")
DATA_ROOT = ROOT / "data" / "arc_agi"
ARC_AGI_1_REPO = os.environ.get("ARC_AGI_1_REPO", "https://github.com/fchollet/ARC-AGI.git")
ARC_AGI_2_REPO = os.environ.get("ARC_AGI_2_REPO", "https://github.com/arcprize/ARC-AGI-2.git")
ARC_VERSION = os.environ.get("STAGE5_ARC_AGI_VERSION", "1")
ARC_SPLIT = os.environ.get("STAGE5_ARC_AGI_SPLIT", "evaluation")
LIMIT = int(os.environ.get("STAGE5_ARC_AGI_LIMIT", "10"))
MAX_NEW_TOKENS = int(os.environ.get("STAGE5_ARC_AGI_MAX_NEW_TOKENS", "512"))
GRID_FORMAT = os.environ.get("STAGE5_ARC_AGI_GRID_FORMAT", "compact")
PROGRAM_PARSE_MODE = os.environ.get("STAGE5_ARC_AGI_PROGRAM_PARSE_MODE", "fallback")
DTYPE = os.environ.get("DTYPE", "bfloat16")
ADAPTER_DTYPE = os.environ.get("ADAPTER_DTYPE", "float32")
DEVICE = os.environ.get("DEVICE", "cuda")
PUSH_RESULTS = os.environ.get("STAGE5_ARC_AGI_TTA_SWEEP_PUSH", "1").strip().lower() in {"1", "true", "yes", "y"}
RESUME = os.environ.get("STAGE5_ARC_AGI_TTA_SWEEP_RESUME", "1").strip().lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class TtaVariant:
    name: str
    geometry_tta: str


@dataclass(frozen=True)
class ModelArm:
    name: str
    mode: str
    checkpoint: Path | None = None


TTA_VARIANTS = {
    "none": TtaVariant("none", "none"),
    "identity": TtaVariant("identity", "identity"),
    "rotations": TtaVariant("rotations", "rot90,rot180,rot270"),
    "flips": TtaVariant("flips", "flip_h,flip_v"),
    "diagonals": TtaVariant("diagonals", "transpose,anti_transpose"),
    "all": TtaVariant("all", "all"),
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


def requested_tta_variants() -> list[TtaVariant]:
    value = os.environ.get("STAGE5_ARC_AGI_TTA_VARIANTS", "none,all")
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(names) - set(TTA_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown TTA variants: {sorted(unknown)}")
    return [TTA_VARIANTS[name] for name in names]


def requested_model_arms(start_ckpt: Path, recovered_ckpt: Path) -> list[ModelArm]:
    all_arms = {
        "base": ModelArm("base", "base"),
        "phase1_start": ModelArm("phase1_start", "phase1", start_ckpt),
        "recovered": ModelArm("recovered", "phase1", recovered_ckpt),
    }
    value = os.environ.get("STAGE5_ARC_AGI_TTA_MODELS", "base,phase1_start,recovered")
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(names) - set(all_arms)
    if unknown:
        raise ValueError(f"Unknown TTA model arms: {sorted(unknown)}")
    return [all_arms[name] for name in names]


def eval_arm(arm: ModelArm, variant: TtaVariant, tasks_path: Path) -> dict[str, Any]:
    label = f"{arm.name}__tta_{variant.name}"
    summary_json = RUN_DIR / f"{label}_summary.json"
    if RESUME and summary_json.exists():
        print(f"resume_existing={summary_json}")
        payload = read_json(summary_json)
        return summarize_arm_payload(arm, variant, payload)

    cmd = [
        sys.executable,
        "eval/eval_arc_agi.py",
        "--tasks_path",
        str(tasks_path),
        "--limit",
        str(LIMIT),
        "--mode",
        arm.mode,
        "--max_new_tokens",
        str(MAX_NEW_TOKENS),
        "--grid_format",
        GRID_FORMAT,
        "--geometry_tta",
        variant.geometry_tta,
        "--program_parse_mode",
        PROGRAM_PARSE_MODE,
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
    if arm.mode != "base":
        if arm.checkpoint is None:
            raise ValueError(f"checkpoint required for arm={arm.name}")
        cmd += ["--checkpoint", path_for_cli(arm.checkpoint), "--max_loops", "4", "--num_candidates", "1"]
    run(cmd, log_name=f"{label}.log")
    return summarize_arm_payload(arm, variant, read_json(summary_json))


def summarize_arm_payload(arm: ModelArm, variant: TtaVariant, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    source_summary = payload.get("candidate_source_summary", {})
    model_candidate_count = sum(stats["count"] for source, stats in source_summary.items() if source.startswith("model"))
    model_exact_count = sum(stats["exact"] for source, stats in source_summary.items() if source.startswith("model"))
    return {
        "arm": arm.name,
        "mode": arm.mode,
        "geometry_tta": variant.geometry_tta,
        "tta_variant": variant.name,
        "checkpoint": path_for_cli(arm.checkpoint) if arm.checkpoint is not None else None,
        "first_exact": summary["first_exact"],
        "selected_exact": summary["selected_exact"],
        "best_of_k_exact": summary["best_of_k_exact"],
        "examples_with_targets": summary["examples_with_targets"],
        "tasks_solved_best_of_k": summary["tasks_solved_best_of_k"],
        "tasks_with_targets": summary["tasks_with_targets"],
        "valid_candidate_rate": summary["valid_candidate_rate"],
        "model_candidate_count": model_candidate_count,
        "model_exact_count": model_exact_count,
    }


def delta_row(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    keys = ("first_exact", "selected_exact", "best_of_k_exact", "tasks_solved_best_of_k", "model_exact_count")
    return {f"{key}_delta": int(candidate.get(key, 0)) - int(reference.get(key, 0)) for key in keys}


def compute_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["arm"], row["tta_variant"]): row for row in rows}
    deltas: dict[str, Any] = {}
    for row in rows:
        baseline = by_key.get((row["arm"], "none"))
        if baseline is not None and row["tta_variant"] != "none":
            deltas[f"{row['arm']}:{row['tta_variant']}_vs_none"] = delta_row(row, baseline)
        base_same_variant = by_key.get(("base", row["tta_variant"]))
        if base_same_variant is not None and row["arm"] != "base":
            deltas[f"{row['arm']}:vs_base_at_{row['tta_variant']}"] = delta_row(row, base_same_variant)
    return deltas


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
        print("No ARC geometry TTA sweep outputs changed.")
        return
    run(["git", "commit", "-m", "Record ARC geometry TTA sweep"])
    run(["git", "push", "origin", "main"], check=False)


def write_report(payload: dict[str, Any]) -> None:
    (RUN_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Stage 5 ARC Geometry TTA Sweep - {RUN_ID}",
        "",
        f"- ARC version/split: `{ARC_VERSION}` / `{ARC_SPLIT}`",
        f"- Limit: `{LIMIT}`",
        f"- Tasks path: `{payload['metadata']['tasks_path']}`",
        f"- Program parse mode: `{PROGRAM_PARSE_MODE}`",
        f"- TTA variants: `{', '.join(payload['metadata']['tta_variants'])}`",
        f"- Model arms: `{', '.join(payload['metadata']['model_arms'])}`",
        "",
        "## Results",
        "",
        "| Arm | TTA | Selected | Best-of-K | Tasks best | Valid rate | Model exact candidates |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        examples = row["examples_with_targets"]
        tasks = row["tasks_with_targets"]
        lines.append(
            f"| `{row['arm']}` | `{row['tta_variant']}` | "
            f"{row['selected_exact']}/{examples} | {row['best_of_k_exact']}/{examples} | "
            f"{row['tasks_solved_best_of_k']}/{tasks} | {row['valid_candidate_rate']:.4f} | "
            f"{row['model_exact_count']}/{row['model_candidate_count']} |"
        )
    lines += ["", "## Deltas"]
    for name, delta in sorted(payload["deltas"].items()):
        lines.append(f"- `{name}`: `{delta}`")
    lines += [
        "",
        "Interpretation guide: useful TTA should improve best-of-K or selected exact without only "
        "inflating invalid candidates. Recurrent-specific lift shows up when recovered-vs-base deltas "
        "narrow under the same TTA variant.",
    ]
    (RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"))


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Run from Colab. Configure STAGE5_ARC_AGI_TTA_VARIANTS, "
            "STAGE5_ARC_AGI_TTA_MODELS, STAGE5_ARC_AGI_LIMIT, and checkpoint env vars."
        )
        return 0

    curriculum = maybe_curriculum_summary()
    tasks_path = resolve_tasks_path()
    start_ckpt = phase1_start_checkpoint_from_curriculum(curriculum)
    recovered_ckpt = recovered_checkpoint(curriculum)
    if not start_ckpt.exists():
        raise FileNotFoundError(start_ckpt)
    if not recovered_ckpt.exists():
        raise FileNotFoundError(recovered_ckpt)

    variants = requested_tta_variants()
    arms = requested_model_arms(start_ckpt, recovered_ckpt)
    metadata = {
        "run_id": RUN_ID,
        "arc_version": ARC_VERSION,
        "arc_split": ARC_SPLIT,
        "limit": LIMIT,
        "tasks_path": str(tasks_path),
        "curriculum_summary": CURRICULUM_SUMMARY or None,
        "phase1_start_checkpoint": path_for_cli(start_ckpt),
        "recovered_checkpoint": path_for_cli(recovered_ckpt),
        "grid_format": GRID_FORMAT,
        "program_parse_mode": PROGRAM_PARSE_MODE,
        "tta_variants": [variant.name for variant in variants],
        "model_arms": [arm.name for arm in arms],
    }
    (RUN_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

    rows = [eval_arm(arm, variant, tasks_path) for variant in variants for arm in arms]
    payload = {
        "run_id": RUN_ID,
        "metadata": metadata,
        "rows": rows,
        "deltas": compute_deltas(rows),
    }
    write_report(payload)
    backup_to_drive()
    if PUSH_RESULTS:
        git_commit_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
