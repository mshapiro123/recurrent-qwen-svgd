"""Run matched full-model AdamW dense Phase-A controls B, C, and/or D."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(os.environ.get("STAGE5_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.stage5_chain_consolidation_utils import path_for_cli  # noqa: E402


TRAIN_SOURCE = ROOT / "outputs/stage5/stage5_depth_support_ladder8_20260705_204923/data/train_chain_symbol_sft.jsonl"
EVAL_SOURCE = ROOT / "outputs/stage5/stage5_synthetic_depth_frozen_eval_v2_depth14/data/test_chain_mcq.jsonl"
TRAIN_SHA256 = "cf61c14c2629f2caa7e1b6bd100adb122a468d5285b74970aaa4aebfbb56fd12"
EVAL_SHA256 = "3de844669aba303063e6932f5852914ee0993e531c8e65c2a4c4b18e219b3fc8"
MODEL_REVISIONS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
}
ALLOWED_ARMS = ("B", "C", "D")


def sha256_jsonl_content(path: str | Path) -> str:
    """Hash JSONL content independently of checkout newline convention."""
    content = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def parse_arms(value: str) -> list[str]:
    arms = [item.strip().upper() for item in str(value).split(",") if item.strip()]
    if not arms or any(arm not in ALLOWED_ARMS for arm in arms) or len(set(arms)) != len(arms):
        raise ValueError(f"STAGE5_PHASE_A_DENSE_ARMS must be a unique subset of {ALLOWED_ARMS}, got {value!r}")
    return arms


def arm_spec(arm: str) -> dict[str, Any]:
    arm = arm.upper()
    if arm == "B":
        model = "Qwen/Qwen2.5-0.5B-Instruct"
        surface = "direct"
    elif arm == "C":
        model = "Qwen/Qwen2.5-0.5B-Instruct"
        surface = "serialized_orbit_scratchpad"
    elif arm == "D":
        model = "Qwen/Qwen2.5-1.5B-Instruct"
        surface = "direct"
    else:
        raise ValueError(f"Unknown Phase A arm {arm!r}")
    return {
        "arm": arm,
        "model_name": model,
        "revision": MODEL_REVISIONS[model],
        "training_surface": surface,
        "optimizer": "adamw_full_fp32_state",
        "steps": 4000,
        "effective_batch_size": 8,
        "learning_rate": 2e-6,
    }


def build_arm_rows(rows: list[dict[str, Any]], *, surface: str) -> list[dict[str, Any]]:
    if surface not in {"direct", "serialized_orbit_scratchpad"}:
        raise ValueError(f"Unknown dense training surface {surface!r}")
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        orbit = [str(value).strip() for value in row["orbit"]]
        target = str(row["target"]).strip()
        if not orbit or orbit[-1] != target:
            raise RuntimeError(f"Orbit/target mismatch for {row.get('instance_id')}")
        row.pop("loop_completions", None)
        row.pop("chain_symbol_by_loop", None)
        row.pop("intermediate_chain_supervision", None)
        row["target_loop_count"] = 1
        if surface == "direct":
            row["completion"] = f" {target}"
        else:
            steps = " -> ".join(orbit[1:])
            row["completion"] = f" steps: {steps} answer: {target}"
        row["phase_a_training_surface"] = surface
        output.append(row)
    return output


def _run_stream(cmd: list[str], *, log_path: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
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
        if log_path is not None and len(lines) % 50 == 0:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("".join(lines), encoding="utf-8")
    returncode = process.wait()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("".join(lines), encoding="utf-8")
    if returncode:
        raise subprocess.CalledProcessError(returncode, cmd)


def _publish(run_dir: Path, message: str) -> None:
    paths = [run_dir / "summary.json", run_dir / "summary.md"]
    paths.extend(run_dir.glob("configs/*.yaml"))
    paths.extend(run_dir.glob("train/*/train.log"))
    paths.extend(run_dir.glob("eval/*/summary.json"))
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in paths:
        if path.exists():
            subprocess.run(["git", "add", "-f", path_for_cli(path)], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    push = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
    if push.returncode:
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)


def _gpu_preflight(arms: list[str]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase A dense-full training requires CUDA")
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024**3
    minimum = 35.0 if "D" in arms else 20.0
    if total_gib < minimum:
        raise RuntimeError(f"Arms {arms} require at least {minimum:.0f} GiB GPU memory; found {total_gib:.1f} GiB")
    receipt = {"name": props.name, "total_gib": total_gib, "required_gib": minimum}
    print("phase_a_gpu_preflight:", json.dumps(receipt, indent=2), flush=True)
    return receipt


def _write_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase A Dense Full Controls - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Arms: `{payload['arms_requested']}`",
        f"- Train SHA256: `{payload['data']['train_sha256']}`",
        f"- Eval SHA256: `{payload['data']['eval_sha256']}`",
        "",
        "| Arm | Model | Surface | Accuracy | Checkpoint |",
        "|---|---|---|---:|---|",
    ]
    for arm, result in payload.get("results", {}).items():
        spec = result["spec"]
        evaluation = result.get("evaluation") or {}
        lines.append(
            f"| {arm} | {spec['model_name']} | {spec['training_surface']} | "
            f"{evaluation.get('accuracy')} | {result.get('checkpoint_drive_backup')} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    arms = parse_arms(os.environ.get("STAGE5_PHASE_A_DENSE_ARMS", "B,C"))
    run_id = os.environ.get("STAGE5_PHASE_A_DENSE_RUN_ID") or time.strftime(
        "stage5_phase_a_dense_full_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs/stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if sha256_jsonl_content(TRAIN_SOURCE) != TRAIN_SHA256:
        raise RuntimeError("Phase A train rows do not match the locked SHA256")
    if sha256_jsonl_content(EVAL_SOURCE) != EVAL_SHA256:
        raise RuntimeError("Phase A frozen eval rows do not match the locked SHA256")
    gpu = _gpu_preflight(arms)
    source_rows = read_jsonl(TRAIN_SOURCE)
    summary_path = run_dir / "summary.json"
    payload = read_json(summary_path) if summary_path.exists() else {
        "kind": "stage5_phase_a_dense_full",
        "run_id": run_id,
        "status": "started",
        "arms_requested": arms,
        "optimizer_contract": "full_model_adamw_fp32_parameters_and_moments_bf16_compute",
        "steps": 4000,
        "data": {
            "hash_semantics": "sha256_newline_normalized_jsonl_bytes",
            "train_source": path_for_cli(TRAIN_SOURCE),
            "train_sha256": TRAIN_SHA256,
            "train_rows": len(source_rows),
            "eval_source": path_for_cli(EVAL_SOURCE),
            "eval_sha256": EVAL_SHA256,
        },
        "gpu": gpu,
        "results": {},
    }
    if payload.get("arms_requested") != arms:
        raise RuntimeError("Existing Phase A run ID was created for different arms")

    for arm in arms:
        if (payload.get("results") or {}).get(arm, {}).get("status") == "finished":
            print(f"phase_a_arm_{arm}=already_finished", flush=True)
            continue
        spec = arm_spec(arm)
        rows = build_arm_rows(source_rows, surface=spec["training_surface"])
        train_jsonl = run_dir / "data" / f"arm_{arm}.jsonl"
        write_jsonl(train_jsonl, rows)
        config = {
            "model_name": spec["model_name"],
            "revision": spec["revision"],
            "optimizer": "adamw",
            "parameter_dtype": "float32",
            "compute_dtype": "bfloat16",
            "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "default"),
            "max_length": 512,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "seed": int(os.environ.get("STAGE5_PHASE_A_DENSE_SEED", "931337")),
            "learning_rate": float(os.environ.get("STAGE5_PHASE_A_DENSE_LR", "2e-6")),
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "max_steps": 4000,
            "save_every": 2000,
            "log_every": 25,
            "gradient_checkpointing": True,
            "output_dir": path_for_cli(run_dir / "train" / arm),
            "backup_root": f"/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/{run_id}/{arm}",
            "train_sha256": TRAIN_SHA256,
            "eval_sha256": EVAL_SHA256,
            "training_surface": spec["training_surface"],
        }
        config_path = run_dir / "configs" / f"arm_{arm}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        backup_checkpoint = Path(config["backup_root"]) / "dense_full_step_4000"
        train_log = run_dir / "train" / arm / "train.log"
        if not (backup_checkpoint / "config.json").exists():
            payload["status"] = f"training_arm_{arm}"
            payload.setdefault("results", {})[arm] = {"status": "training", "spec": spec}
            write_json(summary_path, payload)
            _write_markdown(run_dir, payload)
            _run_stream(
                [
                    sys.executable,
                    "training/train_dense_full.py",
                    "--config",
                    path_for_cli(config_path),
                    "--train_jsonl",
                    path_for_cli(train_jsonl),
                    "--device",
                    "cuda",
                ],
                log_path=train_log,
            )
            train_summary = read_json(run_dir / "train" / arm / "train_dense_full_summary.json")
            checkpoint = Path(str(train_summary["final_checkpoint_drive_backup"]))
        else:
            checkpoint = backup_checkpoint
            train_summary = {
                "status": "restored_from_existing_drive_backup",
                "final_checkpoint_drive_backup": str(checkpoint),
            }
            print(f"phase_a_arm_{arm}_checkpoint_already_on_drive={checkpoint}", flush=True)

        eval_dir = run_dir / "eval" / arm
        eval_summary_path = eval_dir / "summary.json"
        if not eval_summary_path.exists():
            _run_stream(
                [
                    sys.executable,
                    "eval/eval_synthetic_depth_dense.py",
                    "--data_jsonl",
                    path_for_cli(EVAL_SOURCE),
                    "--checkpoint",
                    str(checkpoint),
                    "--output_jsonl",
                    path_for_cli(eval_dir / "rows.jsonl"),
                    "--output_summary",
                    path_for_cli(eval_summary_path),
                    "--batch_size",
                    "8" if arm == "D" else "16",
                    "--max_new_tokens",
                    "96" if arm == "C" else "32",
                    "--dtype",
                    "bfloat16",
                    "--device",
                    "cuda",
                ]
            )
        evaluation = read_json(eval_summary_path)
        payload.setdefault("results", {})[arm] = {
            "status": "finished",
            "spec": spec,
            "checkpoint_drive_backup": str(checkpoint),
            "training": train_summary,
            "evaluation_summary": path_for_cli(eval_summary_path),
            "evaluation": {
                "correct": evaluation["correct"],
                "total": evaluation["total"],
                "accuracy": evaluation["accuracy"],
                "by_depth": evaluation["by_depth"],
            },
        }
        payload["status"] = f"arm_{arm}_finished"
        write_json(summary_path, payload)
        _write_markdown(run_dir, payload)
        _publish(run_dir, f"Record Phase A dense-full arm {arm} {run_id} [skip ci]")
        local_checkpoint_root = run_dir / "train" / arm
        for local_checkpoint in local_checkpoint_root.glob("dense_full_step_*"):
            shutil.rmtree(local_checkpoint, ignore_errors=True)

    payload["status"] = "finished"
    write_json(summary_path, payload)
    _write_markdown(run_dir, payload)
    _publish(run_dir, f"Finish Phase A dense-full controls {run_id} [skip ci]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
