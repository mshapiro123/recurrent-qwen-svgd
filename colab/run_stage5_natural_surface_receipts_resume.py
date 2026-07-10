"""Resume an interrupted natural-surface receipts run without repeating landed work."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(os.environ.get("STAGE5_ROOT", "/content/recurrent-qwen-svgd"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_natural_surface_receipts import (  # noqa: E402
    checkpoint_map,
    env_flag,
    paired_mcnemar,
    path_for_cli,
    publish,
    read_json,
    run_active_eval,
    run_same_reader_eval,
)


DEFAULT_RESUME_RUN_ID = "stage5_natural_surface_receipts_20260709_210151"
DEFAULT_VARIANTS = (
    "robust_baton_default_d1_12,robust_relay_unseen_names_d1_12,"
    "robust_pointer_unseen_names_d1_12,robust_relay_passive_d1_12,"
    "robust_pointer_passive_d1_12,paired_relay_d1_12,paired_pointer_d1_12"
)
DEFAULT_CHECKPOINTS = "frozen_n24,step_2000,step_4000,step_6000"


def _artifact_path(path: str | os.PathLike[str], *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _valid_json_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return True


def active_eval_record_complete(record: dict[str, Any], *, root: Path = ROOT) -> bool:
    summary = record.get("active_summary")
    log = record.get("eval_log")
    if not summary or not log:
        return False
    return _valid_json_file(_artifact_path(summary, root=root)) and _artifact_path(log, root=root).is_file()


def same_reader_record_complete(record: dict[str, Any], *, root: Path = ROOT) -> bool:
    summary = record.get("same_reader_summary")
    return bool(summary) and _valid_json_file(_artifact_path(summary, root=root))


def resume_plan(
    payload: dict[str, Any],
    *,
    checkpoint_labels: list[str],
    variant_names: list[str],
    active_complete: Callable[[dict[str, Any]], bool],
    same_reader_families: list[str],
    same_reader_complete: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    active_pending: list[dict[str, str]] = []
    active_skipped = 0
    for checkpoint in checkpoint_labels:
        records = payload.get("evals", {}).get(checkpoint, {})
        for variant in variant_names:
            record = records.get(variant)
            if isinstance(record, dict) and active_complete(record):
                active_skipped += 1
            else:
                active_pending.append({"checkpoint": checkpoint, "variant": variant})

    same_reader_pending: list[dict[str, str]] = []
    same_reader_skipped = 0
    checker = same_reader_complete or (lambda record: bool(record))
    for checkpoint in [label for label in checkpoint_labels if label != "frozen_n24"]:
        records = payload.get("same_reader", {}).get(checkpoint, {})
        for family in same_reader_families:
            record = records.get(family)
            if isinstance(record, dict) and checker(record):
                same_reader_skipped += 1
            else:
                same_reader_pending.append({"checkpoint": checkpoint, "family": family})

    return {
        "active_skipped": active_skipped,
        "active_pending": active_pending,
        "same_reader_skipped": same_reader_skipped,
        "same_reader_pending": same_reader_pending,
    }


def _csv(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


def main() -> int:
    run_id = os.environ.get("STAGE5_NATURAL_RECEIPTS_RESUME_RUN_ID", DEFAULT_RESUME_RUN_ID)
    out_dir = ROOT / "outputs" / "stage5" / run_id
    summary_path = out_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Cannot resume: committed summary is missing: {summary_path}")

    payload = read_json(summary_path)
    if payload.get("kind") != "stage5_natural_surface_receipts":
        raise RuntimeError(f"Cannot resume unexpected artifact kind: {payload.get('kind')!r}")
    if payload.get("run_id") != run_id:
        raise RuntimeError(f"Resume run-id mismatch: summary={payload.get('run_id')!r}, requested={run_id!r}")

    source_summary = read_json(payload["source_summary"])
    backup_manifest = read_json(payload["backup_manifest"])
    checkpoints = checkpoint_map(source_summary, backup_manifest)
    checkpoint_labels = _csv("STAGE5_NATURAL_RECEIPTS_CHECKPOINTS", DEFAULT_CHECKPOINTS)
    for label in checkpoint_labels:
        if label not in checkpoints:
            raise KeyError(f"Requested checkpoint {label!r} not found in {sorted(checkpoints)}")
        checkpoint_path = _artifact_path(checkpoints[label], root=ROOT)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint for {label} is unavailable after Drive mount: {checkpoint_path}")

    variant_names = _csv("STAGE5_NATURAL_RECEIPTS_VARIANTS", DEFAULT_VARIANTS)
    eval_specs: dict[str, tuple[str, str]] = {}
    files = payload.get("generated_data", {}).get("files", {})
    for variant in variant_names:
        if variant not in files:
            raise KeyError(f"Frozen diagnostic data {variant!r} is absent from committed summary")
        data_path = _artifact_path(files[variant], root=ROOT)
        if not data_path.is_file():
            raise FileNotFoundError(f"Frozen diagnostic data is missing: {data_path}")
        eval_specs[variant] = (files[variant], "name:")
    if env_flag("STAGE5_NATURAL_RECEIPTS_RUN_FULL_SYNTHETIC", "1"):
        synthetic = "outputs/stage5/stage5_synthetic_depth_frozen_eval_v3_depth22_n24/data/test_chain_mcq.jsonl"
        if not _artifact_path(synthetic, root=ROOT).is_file():
            raise FileNotFoundError(f"Synthetic non-regression data is missing: {synthetic}")
        variant_names.append("synthetic_frozen_v3_d1_12")
        eval_specs["synthetic_frozen_v3_d1_12"] = (synthetic, "letter:")

    same_reader_specs = {
        "relay_original": source_summary["data_paths"]["relay_test_chain_mcq"],
        "pointer_original": source_summary["data_paths"]["pointer_test_chain_mcq"],
    }
    for data_path in same_reader_specs.values():
        if not _artifact_path(data_path, root=ROOT).is_file():
            raise FileNotFoundError(f"Same-reader source data is missing: {data_path}")

    plan = resume_plan(
        payload,
        checkpoint_labels=checkpoint_labels,
        variant_names=variant_names,
        active_complete=lambda record: active_eval_record_complete(record, root=ROOT),
        same_reader_families=list(same_reader_specs),
        same_reader_complete=lambda record: same_reader_record_complete(record, root=ROOT),
    )
    print("NATURAL_RECEIPTS_RESUME_PLAN", json.dumps(plan, indent=2), flush=True)
    previous_status = payload.get("status")
    payload.setdefault("resume_history", []).append(
        {
            "resumed_at_unix": time.time(),
            "previous_status": previous_status,
            "plan": plan,
        }
    )
    payload["status"] = "resume_started"
    publish(out_dir, payload, message=f"Resume natural-surface receipts {run_id} [skip ci]")

    dtype = os.environ.get("STAGE5_NATURAL_RECEIPTS_DTYPE", "bfloat16")
    payload.setdefault("evals", {})
    for checkpoint_label in checkpoint_labels:
        payload["evals"].setdefault(checkpoint_label, {})
        for variant in variant_names:
            existing = payload["evals"][checkpoint_label].get(variant)
            if isinstance(existing, dict) and active_eval_record_complete(existing, root=ROOT):
                print(f"resume_skip_active checkpoint={checkpoint_label} variant={variant}", flush=True)
                continue
            data_jsonl, value_prefix = eval_specs[variant]
            print(f"resume_run_active checkpoint={checkpoint_label} variant={variant}", flush=True)
            result = run_active_eval(
                out_dir / "eval" / checkpoint_label,
                label=f"{checkpoint_label}_{variant}",
                checkpoint=checkpoints[checkpoint_label],
                data_jsonl=data_jsonl,
                loop_counts="1,2,3,4,5,6,7,8,9,10,11,12",
                value_prefix=value_prefix,
                dtype=dtype,
                keep_rows=variant.startswith("paired_"),
            )
            payload["evals"][checkpoint_label][variant] = result
            if {
                "paired_relay_d1_12",
                "paired_pointer_d1_12",
            }.issubset(payload["evals"][checkpoint_label]):
                payload["evals"][checkpoint_label]["paired_relay_pointer_mcnemar"] = paired_mcnemar(
                    payload["evals"][checkpoint_label]["paired_relay_d1_12"],
                    payload["evals"][checkpoint_label]["paired_pointer_d1_12"],
                )
            payload["status"] = f"evaluated_{checkpoint_label}_{variant}"
            publish(
                out_dir,
                payload,
                message=f"Record resumed natural-surface receipt eval {run_id} {checkpoint_label} {variant} [skip ci]",
            )

    if env_flag("STAGE5_NATURAL_RECEIPTS_RUN_SAME_READER", "1"):
        payload.setdefault("same_reader", {})
        for checkpoint_label in [label for label in checkpoint_labels if label != "frozen_n24"]:
            payload["same_reader"].setdefault(checkpoint_label, {})
            for family_label, data_jsonl in same_reader_specs.items():
                existing = payload["same_reader"][checkpoint_label].get(family_label)
                if isinstance(existing, dict) and same_reader_record_complete(existing, root=ROOT):
                    print(f"resume_skip_same_reader checkpoint={checkpoint_label} family={family_label}", flush=True)
                    continue
                print(f"resume_run_same_reader checkpoint={checkpoint_label} family={family_label}", flush=True)
                result = run_same_reader_eval(
                    out_dir / "same_reader" / checkpoint_label,
                    label=f"{checkpoint_label}_{family_label}",
                    checkpoint=checkpoints[checkpoint_label],
                    data_jsonl=data_jsonl,
                    max_loops=12,
                    value_prefix="name:",
                    dtype=dtype,
                )
                payload["same_reader"][checkpoint_label][family_label] = result
                payload["status"] = f"same_reader_{checkpoint_label}_{family_label}"
                publish(
                    out_dir,
                    payload,
                    message=f"Record resumed natural-surface same-reader receipt {run_id} {checkpoint_label} {family_label} [skip ci]",
                )

    payload["status"] = "finished"
    payload["resume_completed_at_unix"] = time.time()
    publish(out_dir, payload, message=f"Record resumed natural-surface receipts final {run_id} [skip ci]")
    print(json.dumps({"run_id": run_id, "summary": path_for_cli(summary_path), "status": "finished"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
