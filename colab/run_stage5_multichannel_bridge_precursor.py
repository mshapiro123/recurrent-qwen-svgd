"""Run the eval-only multi-channel bridge precursor battery on frozen checkpoints."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("STAGE5_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_natural_surface_transfer import restore_checkpoint  # noqa: E402
from colab.stage5_publish_utils import publishable_artifact_paths  # noqa: E402
from training.abductive_injective_task import (  # noqa: E402
    AbductiveInjectiveConfig,
    build_rows,
    row_manifest,
    write_jsonl,
)


EXPECTED_BACKWARD_TEST_SHA = "4dd29d9fb7b4170390234646c7c1773377eea56145f6ae659e38f3ae443f2068"
N24_CHECKPOINT_SHA = "898a259db2ab344ece4545e2910b051840e8408dbe4927f799e7cdb3cdd8c7dc"
NATURAL_KEEPER_SHA = "0f657b653078ba403cbc666410e7598ca20c836d5bd6e19a0e85a186a82c5d2f"
BACKWARD_FIXED_SHA = "0d6cf119bd66290a2c85686bf58fdc6f9363109c8fdae0ea625f32d13409a1a6"
BACKWARD_RECOVERY_SHA = "fc98feb5d5bd450f7ecc4f6d43ce36fd436418d7ad2cd69df38a089d5ec453d1"

N24_DATA = ROOT / "outputs/stage5/stage5_synthetic_depth_frozen_eval_v3_depth22_n24/data/test_chain_mcq.jsonl"
STAIRCASE_SUMMARY = ROOT / "outputs/stage5/stage5_inverse_composition_staircase_20260713/summary.json"
DRIVE_RESUME_ROOT = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
STAIRCASE_READING_ONE_LABELS = frozenset({"reading_one", "per_position_install_cost_confirmed"})


CONDITION_SPECS: dict[str, dict[str, Any]] = {
    "n24_step6000": {
        "sha256": N24_CHECKPOINT_SHA,
        "candidates": [
            "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_n24_support12_rung_20260707_140139/anneal_to_outcome_final/unfrozen_recurrent_step_6000.pt",
        ],
        "data_family": "n24",
        "max_depth": 14,
        "max_loops": 14,
        "measurements": "m1,m2,m3",
        "value_prefix": "letter:",
    },
    "natural_surface_keeper": {
        "sha256": NATURAL_KEEPER_SHA,
        "candidates": [
            "/content/drive/MyDrive/recurrent-qwen-svgd-backups/natural_surface_backup_20260709_180835/checkpoints/stage5_natural_surface_transfer_rung0_fixed_prompt_20260709_133812/unfrozen_recurrent_step_2000.pt",
        ],
        "data_family": "n24",
        "max_depth": 14,
        "max_loops": 14,
        "measurements": "m1,m2",
        "value_prefix": "letter:",
    },
    "backward_fixed_boundary": {
        "sha256": BACKWARD_FIXED_SHA,
        "candidates": [
            "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_phase_g_experiment1_fixed_boundary_20260712/injective_control/unfrozen_recurrent_step_1000.pt",
        ],
        "data_family": "backward",
        "max_depth": 8,
        "max_loops": 8,
        "measurements": "m1,m2",
        "value_prefix": "name:",
    },
    "backward_recovery": {
        "sha256": BACKWARD_RECOVERY_SHA,
        "candidates": [
            "/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_phase_g_injective_curriculum_recovery_20260712/injective_control/unfrozen_recurrent_step_2000.pt",
        ],
        "data_family": "backward",
        "max_depth": 8,
        "max_loops": 8,
        "measurements": "m1,m2",
        "value_prefix": "name:",
    },
}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def path_for_cli(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def publish(run_dir: Path, message: str) -> None:
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    for path in publishable_artifact_paths(run_dir):
        relative = path.relative_to(run_dir)
        if "data" in relative.parts or path.name == "m3_progress.json":
            continue
        subprocess.run(["git", "add", "-f", path_for_cli(path)], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    pushed = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
    if pushed.returncode:
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)


def prepare_backward_data(run_dir: Path) -> Path:
    path = run_dir / "data" / "test_injective_depth8.jsonl"
    rows = build_rows(
        AbductiveInjectiveConfig(n_symbols=20, max_depth=8, rows_per_depth=128, seed=1_104_729),
        split="test",
        mode="injective",
    )
    manifest = row_manifest(rows)
    if manifest["row_sha256"] != EXPECTED_BACKWARD_TEST_SHA:
        raise RuntimeError(
            "Regenerated backward frozen rows differ from the locked manifest: "
            f"{manifest['row_sha256']} != {EXPECTED_BACKWARD_TEST_SHA}"
        )
    write_jsonl(path, rows)
    write_json(run_dir / "data" / "backward_manifest.json", manifest)
    return path


def resolve_staircase_reading(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize historical scalar, direct-entry, and per-cap staircase receipts."""

    raw = payload.get("matched_arm_reading")
    cap: str | None = None
    entry: dict[str, Any] = {}
    source_schema = "missing"

    if isinstance(raw, str):
        reported_reading: str | None = raw
        source_schema = "scalar"
    elif isinstance(raw, dict) and isinstance(raw.get("reading"), str):
        entry = raw
        reported_reading = raw["reading"]
        source_schema = "direct_entry"
    elif isinstance(raw, dict):
        numeric_entries: list[tuple[int, str, dict[str, Any]]] = []
        fallback_entries: list[tuple[str, dict[str, Any]]] = []
        for key, value in raw.items():
            if not isinstance(value, dict) or not isinstance(value.get("reading"), str):
                continue
            key_text = str(key)
            fallback_entries.append((key_text, value))
            try:
                numeric_entries.append((int(key_text), key_text, value))
            except ValueError:
                continue
        if numeric_entries:
            _, cap, entry = max(numeric_entries, key=lambda item: item[0])
        elif fallback_entries:
            cap, entry = sorted(fallback_entries, key=lambda item: item[0])[-1]
        reported_reading = entry.get("reading")
        source_schema = "per_cap" if entry else "missing"
    else:
        reported_reading = None

    reading = reported_reading
    correction: str | None = None
    if (
        reading == "non_native_position_cost"
        and entry.get("experiment_weighted_labels_to_bar") is None
        and entry.get("control_weighted_labels_to_bar") is not None
    ):
        reading = "experiment_stalled_at_matched_dose"
        correction = "post_run_clarification_no_experiment_dose_to_bar"

    return {
        "cap": cap,
        "source_schema": source_schema,
        "reported_reading": reported_reading,
        "reading": reading,
        "correction": correction,
        "reading_one": reading in STAIRCASE_READING_ONE_LABELS,
    }


def staircase_reading() -> dict[str, Any]:
    summary = path_for_cli(STAIRCASE_SUMMARY)
    if not STAIRCASE_SUMMARY.exists():
        return {"summary": summary, **resolve_staircase_reading({})}
    return {"summary": summary, **resolve_staircase_reading(read_json(STAIRCASE_SUMMARY))}


def aggregate_battery(condition_summaries: dict[str, dict[str, Any]], *, reading_one: bool) -> dict[str, Any]:
    def observed(condition: str, measurement: str) -> bool | None:
        payload = (
            condition_summaries.get(condition, {})
            .get("measurements", {})
            .get(measurement, {})
            .get("classification")
        )
        if payload is None:
            return None
        return bool(payload.get("confirmed", False))

    evidence = {
        "m1": {
            "n24_step6000": observed("n24_step6000", "m1"),
            "backward_recovery": observed("backward_recovery", "m1"),
        },
        "m2": {
            "n24_step6000": observed("n24_step6000", "m2"),
            "backward_recovery": observed("backward_recovery", "m2"),
        },
        "m3": {"n24_step6000": observed("n24_step6000", "m3")},
    }

    def status(required: tuple[str, ...], measurement: str) -> str:
        readings = [evidence[measurement][condition] for condition in required]
        if all(reading is True for reading in readings):
            return "confirmed"
        if any(reading is False for reading in readings):
            return "not_confirmed"
        return "pending_replication" if any(reading is True for reading in readings) else "not_run"

    measurement_status = {
        "m1": status(("n24_step6000", "backward_recovery"), "m1"),
        "m2": status(("n24_step6000", "backward_recovery"), "m2"),
        "m3": status(("n24_step6000",), "m3"),
    }

    measurement_votes = {
        measurement: state == "confirmed" for measurement, state in measurement_status.items()
    }
    count = sum(measurement_votes.values())
    specialization = count >= 2
    return {
        "measurement_evidence": evidence,
        "measurement_status": measurement_status,
        "measurement_votes": measurement_votes,
        "confirmed_measurements": count,
        "battery_specialization_confirmed": specialization,
        "staircase_reading_one": bool(reading_one),
        "architecture_activation_eligible": bool(specialization and reading_one),
        "decision": (
            "activate_candidate_arm"
            if specialization and reading_one
            else "remain_banked"
        ),
        "replication_rule": "M1/M2 require n24_step6000 and backward_recovery; M3 requires n24_step6000",
    }


def write_master_markdown(run_dir: Path, payload: dict[str, Any]) -> None:
    decision = payload.get("decision", {})
    lines = [
        f"# Multi-Channel Bridge Precursor Battery - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Completed conditions: `{payload.get('completed_conditions', [])}`",
        f"- Staircase reading: `{payload.get('staircase', {}).get('reading')}`",
        f"- Measurement evidence: `{decision.get('measurement_evidence')}`",
        f"- Measurement status: `{decision.get('measurement_status')}`",
        f"- Measurement votes: `{decision.get('measurement_votes')}`",
        f"- Battery specialization confirmed: `{decision.get('battery_specialization_confirmed')}`",
        f"- Architecture activation eligible: `{decision.get('architecture_activation_eligible')}`",
        f"- Decision: `{decision.get('decision')}`",
        "",
        "The architecture remains banked unless both the battery and the independent staircase condition pass. "
        "This run is eval-only and changes no queue position.",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_with_liveness(
    command: list[str],
    *,
    status_paths: list[Path],
    max_idle_seconds: float,
) -> None:
    """Stream a child process and terminate it if it stops emitting receipts.

    This is deliberately parent-side: an attention capture can stall inside a
    single CUDA call, where a child-side elapsed-time check cannot execute.
    """

    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def drain_stdout() -> None:
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=drain_stdout, daemon=True).start()
    latest_mtime = {path: path.stat().st_mtime if path.exists() else 0.0 for path in status_paths}
    last_activity = time.monotonic()
    output: list[str] = []
    while True:
        try:
            line = lines.get(timeout=10.0)
        except queue.Empty:
            line = ""
        if line is None and process.poll() is not None:
            break
        if line:
            print(line, end="", flush=True)
            output.append(line)
            last_activity = time.monotonic()
        for path in status_paths:
            mtime = path.stat().st_mtime if path.exists() else 0.0
            if mtime > latest_mtime[path]:
                latest_mtime[path] = mtime
                last_activity = time.monotonic()
        if time.monotonic() - last_activity > max_idle_seconds:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
            raise TimeoutError(
                f"Multichannel evaluator produced no stdout or status update for {max_idle_seconds:.0f}s; "
                f"terminated command. status_paths={[str(path) for path in status_paths]}"
            )
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command, output="".join(output))


def main() -> int:
    run_id = os.environ.get("STAGE5_MULTICHANNEL_RUN_ID") or time.strftime(
        "stage5_multichannel_bridge_precursor_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs" / "stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    backward_data = prepare_backward_data(run_dir)
    if not N24_DATA.exists():
        raise FileNotFoundError(f"Missing locked N24 data: {N24_DATA}")

    mode = os.environ.get("STAGE5_MULTICHANNEL_MODE", "full").strip().lower()
    if mode not in {"pilot", "full"}:
        raise ValueError("STAGE5_MULTICHANNEL_MODE must be 'pilot' or 'full'")
    requested = [
        item.strip()
        for item in os.environ.get(
            "STAGE5_MULTICHANNEL_CONDITIONS",
            "n24_step6000" if mode == "pilot" else "n24_step6000,natural_surface_keeper,backward_fixed_boundary,backward_recovery",
        ).split(",")
        if item.strip()
    ]
    unknown = set(requested) - set(CONDITION_SPECS)
    if unknown:
        raise ValueError(f"Unknown multichannel conditions: {sorted(unknown)}")
    rows_per_depth = int(os.environ.get("STAGE5_MULTICHANNEL_ROWS_PER_DEPTH", "1" if mode == "pilot" else "64"))
    random_draws = int(os.environ.get("STAGE5_MULTICHANNEL_RANDOM_DRAWS", "20"))
    if random_draws < 20:
        raise ValueError("The preregistered battery requires at least 20 random draws")
    m3_batch_size = int(os.environ.get("STAGE5_MULTICHANNEL_M3_BATCH_SIZE", "8"))
    dynamics_row_timeout_seconds = float(os.environ.get("STAGE5_MULTICHANNEL_DYNAMICS_ROW_TIMEOUT_SECONDS", "600"))
    m3_pass_timeout_seconds = float(os.environ.get("STAGE5_MULTICHANNEL_M3_PASS_TIMEOUT_SECONDS", "1800"))
    liveness_timeout_seconds = float(os.environ.get("STAGE5_MULTICHANNEL_LIVENESS_TIMEOUT_SECONDS", "900"))
    dtype = os.environ.get("STAGE5_MULTICHANNEL_DTYPE", "bfloat16")
    resume_root = Path(os.environ.get("STAGE5_MULTICHANNEL_RESUME_ROOT", str(DRIVE_RESUME_ROOT))) / run_id
    summary_path = run_dir / "summary.json"
    existing = read_json(summary_path) if summary_path.exists() else {}
    condition_summaries: dict[str, dict[str, Any]] = dict(existing.get("condition_summaries") or {})
    staircase = staircase_reading()
    payload: dict[str, Any] = {
        "kind": "stage5_multichannel_bridge_precursor_battery",
        "run_id": run_id,
        "status": "running",
        "mode": mode,
        "requested_conditions": requested,
        "completed_conditions": sorted(condition_summaries),
        "rows_per_depth": rows_per_depth,
        "random_draws": random_draws,
        "dynamics_row_timeout_seconds": dynamics_row_timeout_seconds,
        "m3_pass_timeout_seconds": m3_pass_timeout_seconds,
        "liveness_timeout_seconds": liveness_timeout_seconds,
        "staircase": staircase,
        "condition_summaries": condition_summaries,
        "queue_effect": "none",
    }
    write_json(summary_path, payload)
    write_master_markdown(run_dir, payload)

    for condition in requested:
        if condition in condition_summaries and condition_summaries[condition].get("status") == "finished":
            print(f"multichannel_condition_already_finished={condition}", flush=True)
            continue
        spec = CONDITION_SPECS[condition]
        measurements = "m1,m2" if mode == "pilot" else str(spec["measurements"])
        checkpoint, receipt = restore_checkpoint(
            list(spec["candidates"]),
            run_dir / "restored" / f"{condition}.pt",
            label=condition,
        )
        if receipt["selected_checkpoint_sha256"] != spec["sha256"]:
            raise RuntimeError(
                f"{condition} checkpoint SHA mismatch: {receipt['selected_checkpoint_sha256']} != {spec['sha256']}"
            )
        condition_dir = run_dir / "conditions" / condition
        data_jsonl = N24_DATA if spec["data_family"] == "n24" else backward_data
        command = [
            sys.executable,
            "eval/eval_multichannel_bridge_precursor.py",
            "--checkpoint",
            str(checkpoint),
            "--checkpoint_sha256",
            str(spec["sha256"]),
            "--data_jsonl",
            path_for_cli(data_jsonl),
            "--output_summary",
            path_for_cli(condition_dir / "summary.json"),
            "--resume_cache_dir",
            str(resume_root / condition),
            "--condition",
            condition,
            "--measurements",
            measurements,
            "--max_depth",
            str(spec["max_depth"]),
            "--max_loops",
            str(spec["max_loops"]),
            "--rows_per_depth",
            str(rows_per_depth),
            "--random_draws",
            str(random_draws),
            "--m3_batch_size",
            str(m3_batch_size),
            "--dynamics_row_timeout_seconds",
            str(dynamics_row_timeout_seconds),
            "--m3_pass_timeout_seconds",
            str(m3_pass_timeout_seconds),
            "--value_prefix",
            str(spec["value_prefix"]),
            "--attn_implementation",
            "eager",
            "--bridge_projection_mode",
            "split",
            "--dtype",
            dtype,
            "--adapter_dtype",
            "float32",
            "--device",
            os.environ.get("DEVICE", "cuda"),
        ]
        if mode == "pilot":
            command.append("--dynamics_deepest_first")
        condition_resume_dir = resume_root / condition
        run_with_liveness(
            command,
            status_paths=[condition_resume_dir / "dynamics_status.json", condition_resume_dir / "m3_progress.json"],
            max_idle_seconds=liveness_timeout_seconds,
        )
        condition_payload = read_json(condition_dir / "summary.json")
        condition_payload["checkpoint_restore_receipt"] = receipt
        write_json(condition_dir / "summary.json", condition_payload)
        condition_summaries[condition] = condition_payload
        payload.update(
            condition_summaries=condition_summaries,
            completed_conditions=sorted(condition_summaries),
            decision=aggregate_battery(condition_summaries, reading_one=bool(staircase["reading_one"])),
        )
        write_json(summary_path, payload)
        write_master_markdown(run_dir, payload)
        publish(run_dir, f"Record multi-channel bridge precursor {condition} {run_id} [skip ci]")

    payload["status"] = "finished" if set(requested).issubset(condition_summaries) else "partial"
    payload["decision"] = aggregate_battery(condition_summaries, reading_one=bool(staircase["reading_one"]))
    write_json(summary_path, payload)
    write_master_markdown(run_dir, payload)
    publish(run_dir, f"Finish multi-channel bridge precursor {run_id} [skip ci]")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
