"""Run and publish the descriptive Paper One wall-clock latency receipt."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_wall_clock_latency import build_markdown_table, read_jsonl, summarize_records, update_claim_ledger

RUN_ID = os.environ.get("STAGE5_WALL_CLOCK_RUN_ID", "stage5_wall_clock_latency_20260719")
RUN_DIR = ROOT / "outputs/stage5" / RUN_ID
DATA = ROOT / "outputs/stage5/stage5_synthetic_depth_frozen_eval_v2_depth14/data/test_chain_mcq.jsonl"
DRIVE_BACKUP = Path(
    os.environ.get(
        "STAGE5_WALL_CLOCK_DRIVE_BACKUP",
        f"/content/drive/MyDrive/recurrent-qwen-svgd-paper-one/{RUN_ID}",
    )
)

CHECKPOINTS = {
    "A": {
        "source": os.environ.get(
            "STAGE5_WALL_CLOCK_A_CHECKPOINT",
            "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_support8_dose_arm_20260706_153028/anneal_to_outcome_final/unfrozen_recurrent_step_2000.pt",
        ),
        "sha256": "dc00f7b694ce32427eb13b0b85d365bc15e0c0317130bd22d4bbc3568544f71b",
        "kind": "file",
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "lora_rank": 0,
        "lora_alpha": 16,
        "max_new_tokens": 1,
    },
    "E": {
        "source": os.environ.get(
            "STAGE5_WALL_CLOCK_E_CHECKPOINT",
            "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_adapter_budget_arm_e_20260718/chain_depth_le8_dose/unfrozen_recurrent_step_2000.pt",
        ),
        "sha256": "bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839",
        "kind": "file",
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "lora_rank": 16,
        "lora_alpha": 32,
        "max_new_tokens": 1,
    },
    "B": {
        "source": os.environ.get(
            "STAGE5_WALL_CLOCK_B_CHECKPOINT",
            "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_phase_a_dense_full_bc_20260713/B/dense_full_step_4000",
        ),
        "sha256": "bb4fbaa628c11bc40f9d21f8e8f08c42b064463cd6cf357f196dadef27d0fa74",
        "kind": "directory",
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_new_tokens": 32,
    },
    "C": {
        "source": os.environ.get(
            "STAGE5_WALL_CLOCK_C_CHECKPOINT",
            "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_phase_a_dense_full_bc_20260713/C/dense_full_step_4000",
        ),
        "sha256": "f2e7d600e057cb742b28d2f053615520e5257a16c39a3057ad34d89d4301c801",
        "kind": "directory",
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_new_tokens": 96,
    },
    "D": {
        "source": os.environ.get(
            "STAGE5_WALL_CLOCK_D_CHECKPOINT",
            "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_phase_a_dense_full_d_20260713/D/dense_full_step_4000",
        ),
        "sha256": "1e2999731352ac8f36d6cbd03359f4e68e6f93e8c5f7c9c35e04cdfc72b118d2",
        "kind": "directory",
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_new_tokens": 32,
    },
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        if path.is_dir():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, accepted: set[int] = {0}) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
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
    if result.returncode not in accepted:
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
    return result


def restore_run_receipts() -> None:
    if not RUN_DIR.exists() and DRIVE_BACKUP.exists():
        shutil.copytree(DRIVE_BACKUP, RUN_DIR)
        print(f"restored_latency_receipts={DRIVE_BACKUP}", flush=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_BACKUP.mkdir(parents=True, exist_ok=True)


def restore_checkpoint(arm: str, spec: dict[str, Any]) -> Path:
    source = Path(str(spec["source"]))
    if not source.exists():
        raise FileNotFoundError(f"Missing Arm {arm} checkpoint: {source}")
    suffix = source.suffix if source.is_file() else ""
    destination = RUN_DIR / "restored" / f"arm_{arm}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    observed = sha256_path(destination)
    if observed != spec["sha256"]:
        raise RuntimeError(f"Arm {arm} checkpoint hash mismatch: {observed} != {spec['sha256']}")
    print(f"[assert-ok] arm_{arm}_checkpoint_sha256={observed}", flush=True)
    return destination


def backup_compact() -> None:
    for source in RUN_DIR.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(RUN_DIR)
        destination = DRIVE_BACKUP / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative.parts and relative.parts[0] == "restored":
            continue
        shutil.copy2(source, destination)


def publish() -> None:
    run(["git", "pull", "--rebase", "--autostash", "origin", "main"], accepted={0})
    publish_paths = [
        RUN_DIR / "summary.json",
        RUN_DIR / "summary.md",
        RUN_DIR / "conditions.json",
        ROOT / "docs/part1_claim_evidence_ledger.json",
    ]
    for arm in CHECKPOINTS:
        publish_paths.extend([RUN_DIR / "conditions" / arm / "summary.json", RUN_DIR / "conditions" / arm / "status.json"])
    for path in publish_paths:
        if path.exists():
            run(["git", "add", "-f", str(path.relative_to(ROOT))])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    run(["git", "commit", "-m", f"Record Paper One wall-clock latency {RUN_ID} [skip ci]"])
    push = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
    if push.returncode:
        run(["git", "pull", "--rebase", "--autostash", "origin", "main"])
        run(["git", "push", "origin", "main"])


def main() -> int:
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    restore_run_receipts()
    selected_arms = [item.strip().upper() for item in os.environ.get("STAGE5_WALL_CLOCK_ARMS", "A,E,C,B,D").split(",") if item.strip()]
    if len(selected_arms) != len(CHECKPOINTS) or set(selected_arms) != set(CHECKPOINTS):
        raise RuntimeError("The publication receipt requires exactly arms A,E,C,B,D in one hardware session")
    conditions = {
        "kind": "stage5_wall_clock_latency_conditions",
        "run_id": RUN_ID,
        "started_at_unix": time.time(),
        "batch_size": 1,
        "decoding": "greedy",
        "precision": "bfloat16",
        "data_jsonl": str(DATA.relative_to(ROOT)),
        "rows": len(read_jsonl(DATA)),
        "arms": selected_arms,
        "scope": "single hardware configuration, batch size 1, registered evaluation paths",
    }
    (RUN_DIR / "conditions.json").write_text(json.dumps(conditions, indent=2) + "\n", encoding="utf-8")
    for arm in selected_arms:
        spec = CHECKPOINTS[arm]
        output_dir = RUN_DIR / "conditions" / arm
        status_path = output_dir / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "finished" and int(status.get("completed_observations", 0)) == 2080:
                summary_path = output_dir / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("checkpoint_sha256") != spec["sha256"]:
                    raise RuntimeError(f"Completed Arm {arm} receipt has the wrong checkpoint SHA")
                print(f"latency_arm_already_complete={arm} observations=2080", flush=True)
                continue
        checkpoint = restore_checkpoint(arm, spec)
        mirror_dir = DRIVE_BACKUP / "conditions" / arm
        command = [
            sys.executable,
            "eval/eval_wall_clock_latency.py",
            "--arm", arm,
            "--data_jsonl", str(DATA),
            "--checkpoint", str(checkpoint),
            "--checkpoint_sha256", spec["sha256"],
            "--output_dir", str(output_dir),
            "--mirror_dir", str(mirror_dir),
            "--model_name", spec["model_name"],
            "--max_new_tokens", str(spec["max_new_tokens"]),
            "--dtype", "bfloat16",
            "--device", "cuda",
        ]
        if arm in {"A", "E"}:
            command.extend(
                [
                    "--split", "6,18",
                    "--bridge_projection_mode", "split",
                    "--lora_rank", str(spec["lora_rank"]),
                    "--lora_alpha", str(spec["lora_alpha"]),
                    "--adapter_dtype", "float32",
                ]
            )
        run(command)
        backup_compact()
    all_records: list[dict[str, Any]] = []
    arm_summaries: dict[str, Any] = {}
    for arm in selected_arms:
        all_records.extend(read_jsonl(RUN_DIR / "conditions" / arm / "raw_timings.jsonl"))
        arm_summaries[arm] = json.loads((RUN_DIR / "conditions" / arm / "summary.json").read_text(encoding="utf-8"))
    gpu_names = {str(summary["hardware"]["gpu_name"]) for summary in arm_summaries.values()}
    smi_lines = {str(summary["hardware"]["nvidia_smi"]) for summary in arm_summaries.values()}
    if len(gpu_names) != 1 or len(smi_lines) != 1:
        raise RuntimeError(f"Arms were not measured on one hardware configuration: gpu={gpu_names}, smi={smi_lines}")
    if not all(bool(item["hardware"].get("gpu_exclusivity_observed")) for item in arm_summaries.values()):
        raise RuntimeError("A foreign concurrent GPU compute process was visible during at least one arm")
    summary = summarize_records(all_records)
    gpu_name = next(iter(gpu_names))
    summary.update(
        {
            "run_id": RUN_ID,
            "status": "finished",
            "data_jsonl": str(DATA.relative_to(ROOT)),
            "hardware": arm_summaries["A"]["hardware"],
            "checkpoint_receipts": {
                arm: {"sha256": CHECKPOINTS[arm]["sha256"], "registered_path": CHECKPOINTS[arm]["source"]}
                for arm in selected_arms
            },
            "conditions_sentence": (
                f"Measured on one {gpu_name} with bfloat16 model weights, batch size 1, greedy decoding, "
                "and each arm's registered evaluation path; no batched-throughput claim is made."
            ),
            "recurrent_timing_note": (
                "For Arms A/E, total is the actual forced-depth call; prefill is a synchronized one-loop reference "
                "and decode-side is the nonnegative difference."
            ),
            "dense_timing_note": (
                "For Arms B/C/D, total time is the registered Transformers greedy generate path; "
                "prefill is a synchronized one-token generate reference and decode-side time is the nonnegative difference."
            ),
        }
    )
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RUN_DIR / "summary.md").write_text(build_markdown_table(summary), encoding="utf-8")
    update_claim_ledger(
        ROOT / "docs/part1_claim_evidence_ledger.json",
        evidence_path=str((RUN_DIR / "summary.json").relative_to(ROOT)).replace("\\", "/"),
    )
    backup_compact()
    publish()
    print((RUN_DIR / "summary.md").read_text(encoding="utf-8"), flush=True)
    print(f"drive_receipt={DRIVE_BACKUP}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
