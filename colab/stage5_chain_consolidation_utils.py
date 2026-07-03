"""Shared helpers for post-positive synthetic-depth consolidation runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("STAGE5_ROOT", "/content/recurrent-qwen-svgd"))
DRIVE_CHECKPOINT_ROOT = Path(
    os.environ.get(
        "STAGE5_CHAIN_CONSOLIDATION_DRIVE_ROOT",
        "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints",
    )
)


def root_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def path_for_cli(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return candidate.relative_to(ROOT).as_posix()
    except ValueError:
        return str(candidate).replace("\\", "/")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(root_path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = root_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in root_path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = root_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def maybe_existing_path(raw: str | Path | None) -> Path | None:
    if not raw:
        return None
    raw_path = Path(raw)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.insert(0, ROOT / raw_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def restore_checkpoint(candidates: list[str | Path | None], dest: Path, *, label: str) -> Path:
    for raw in candidates:
        source = maybe_existing_path(raw)
        if source is None:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        print(f"restored_{label}={dest}", flush=True)
        return dest
    raise FileNotFoundError(
        f"Could not restore {label}. Tried: " + ", ".join(str(item) for item in candidates if item)
    )


def final_stage_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    stages = list(summary.get("stages") or [])
    if stages:
        return dict(stages[-1])
    return dict(summary)


def resolve_checkpoint_reference(raw: str | Path, dest: Path, *, label: str = "checkpoint") -> tuple[Path, dict[str, Any]]:
    candidate = root_path(raw)
    metadata: dict[str, Any] = {"source_reference": str(raw)}
    if candidate.is_dir():
        summary_path = candidate / "summary.json"
    elif candidate.suffix == ".json":
        summary_path = candidate
    else:
        summary_path = None
    if summary_path is None:
        return restore_checkpoint([raw], dest, label=label), metadata

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stage = final_stage_from_summary(summary)
    checkpoint = restore_checkpoint(
        [
            stage.get("checkpoint_drive_backup"),
            stage.get("checkpoint"),
            summary.get("checkpoint_drive_backup"),
            summary.get("checkpoint"),
        ],
        dest,
        label=label,
    )
    metadata.update(
        {
            "source_summary": path_for_cli(summary_path),
            "source_run_id": summary.get("run_id"),
            "source_stage_name": stage.get("stage_name"),
            "source_checkpoint": stage.get("checkpoint") or summary.get("checkpoint"),
            "source_checkpoint_drive_backup": stage.get("checkpoint_drive_backup")
            or summary.get("checkpoint_drive_backup"),
        }
    )
    return checkpoint, metadata


def latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = sorted(output_dir.glob("unfrozen_recurrent_step_*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint under {output_dir}")
    return checkpoints[-1]


def checkpoint_at_step(output_dir: Path, step: int) -> Path:
    checkpoint = output_dir / f"unfrozen_recurrent_step_{int(step)}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint at step {step}: {checkpoint}")
    return checkpoint


def backup_checkpoint_to_drive(checkpoint: Path, *, run_id: str, stage_name: str, enabled: bool = True) -> str | None:
    if not enabled:
        return None
    dest_dir = DRIVE_CHECKPOINT_ROOT / run_id / stage_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / checkpoint.name
    shutil.copy2(checkpoint, dest)
    print(f"checkpoint_drive_backup={dest}", flush=True)
    return str(dest)


def publish_run(run_dir: Path, *, message: str, update_pointer: bool = False) -> None:
    from colab.stage5_publish_utils import publishable_artifact_paths, update_current_source_summary

    paths = publishable_artifact_paths(run_dir)
    if update_pointer:
        paths.append(update_current_source_summary(ROOT, run_dir / "summary.json"))
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in paths:
        if path.exists():
            subprocess.run(["git", "add", "-f", path_for_cli(path)], cwd=ROOT, check=False)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if diff.returncode == 0:
        print(f"No artifacts to publish for {run_dir}", flush=True)
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    push = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
    if push.returncode == 0:
        return
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
