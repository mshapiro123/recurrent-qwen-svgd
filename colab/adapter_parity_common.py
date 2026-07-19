"""Shared runtime helpers for the Arm E adapter-parity battery."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from training.adapter_parity_battery import (
    ARM_E_ALPHA,
    ARM_E_FINAL_SHA256,
    ARM_E_FORWARD_ACTIVE_PARAMETERS,
    ARM_E_PRETRAINED_BASE_SHA256,
    ARM_E_RANK,
)


ROOT = Path(os.environ.get("STAGE5_ROOT", Path(__file__).resolve().parents[1]))
ARM_E_SUMMARY = ROOT / "outputs/stage5/stage5_adapter_budget_arm_e_20260718/summary.json"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def path_for_cli(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(value)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    command: list[str | os.PathLike[str]],
    *,
    accepted_returncodes: set[int] = {0},
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(
        list(map(str, command)),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    result = subprocess.CompletedProcess(command, process.wait(), "".join(lines), None)
    if result.returncode not in accepted_returncodes:
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
    return result


def restore_arm_e_checkpoint(destination: Path) -> tuple[Path, dict[str, Any]]:
    summary = read_json(ARM_E_SUMMARY)
    if summary.get("final_checkpoint_sha256") != ARM_E_FINAL_SHA256:
        raise RuntimeError("Landed Arm E summary does not contain the locked final SHA")
    stage = list(summary.get("stages") or [])[-1]
    backup = stage.get("drive_backup") or {}
    candidates = [backup.get("path"), summary.get("final_checkpoint")]
    for raw in candidates:
        if not raw:
            continue
        source = Path(raw)
        if not source.is_absolute():
            source = ROOT / source
        if not source.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        observed = sha256_file(destination)
        if observed != ARM_E_FINAL_SHA256:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Arm E checkpoint SHA mismatch: {observed}")
        receipt = {
            "source_summary": path_for_cli(ARM_E_SUMMARY),
            "source_checkpoint": summary["final_checkpoint"],
            "source_drive_backup": backup.get("path"),
            "selected_checkpoint_sha256": observed,
        }
        print(f"[assert-ok] arm_e_final_sha256={observed}", flush=True)
        return destination, receipt
    raise FileNotFoundError("Could not restore the locked Arm E checkpoint from Drive or the repo")


def lora_eval_args() -> list[str]:
    return [
        "--split",
        "6,18",
        "--bridge_projection_mode",
        "split",
        "--lora_rank",
        str(ARM_E_RANK),
        "--lora_alpha",
        str(ARM_E_ALPHA),
        "--dtype",
        os.environ.get("STAGE5_ADAPTER_PARITY_DTYPE", "bfloat16"),
        "--adapter_dtype",
        "float32",
        "--device",
        os.environ.get("DEVICE", "cuda"),
    ]


def assert_adapter_training_summary(summary: dict[str, Any]) -> None:
    if summary.get("pretrained_base_hash_unchanged") is not True:
        raise RuntimeError("Adapter continuation changed pretrained base parameters")
    if summary.get("pretrained_base_sha256_start") != ARM_E_PRETRAINED_BASE_SHA256:
        raise RuntimeError("Adapter continuation started from the wrong pretrained base")
    if summary.get("pretrained_base_sha256_end") != ARM_E_PRETRAINED_BASE_SHA256:
        raise RuntimeError("Adapter continuation ended with a changed pretrained base")
    trainable = summary.get("trainable_parameters") or {}
    if int(trainable.get("total", -1)) != ARM_E_FORWARD_ACTIVE_PARAMETERS:
        raise RuntimeError(f"Unexpected adapter trainable budget: {trainable}")


def adapter_resume_config() -> dict[str, Any]:
    return {
        "training_mode": "frozen_lora",
        "resume_lora": {
            "enabled": True,
            "rank": ARM_E_RANK,
            "alpha": ARM_E_ALPHA,
            "dropout": 0.0,
        },
        "require_all_lora_loaded": True,
        "require_loaded_prefixes": ["bridge."],
        "merge_lora_before_unfreeze": False,
        "require_lora_loaded_before_merge": False,
        "require_frozen_base_hash": True,
        "train_auxiliary": {
            "bridge": True,
            "halting": False,
            "reentry_adapter": False,
            "latent": False,
        },
    }
