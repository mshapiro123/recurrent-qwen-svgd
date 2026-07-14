"""Evaluate Phase-A dense checkpoints at steps 2,000 and 4,000."""

from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(os.environ.get("STAGE5_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_phase_a_dense_full import (  # noqa: E402
    EVAL_SHA256,
    EVAL_SOURCE,
    MODEL_REVISIONS,
    read_jsonl,
    sha256_jsonl_content,
)
from colab.stage5_chain_consolidation_utils import path_for_cli  # noqa: E402
from eval.eval_synthetic_depth_dense import summarize_rows  # noqa: E402


BC_RUN = "stage5_phase_a_dense_full_bc_20260713"
D_RUN = "stage5_phase_a_dense_full_d_20260713"
DRIVE_ROOT = PurePosixPath("/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints")
CHECKPOINT_SPECS = [
    {
        "label": f"{arm}_step{step}",
        "arm": arm,
        "step": step,
        "checkpoint": str(
            DRIVE_ROOT
            / (BC_RUN if arm in {"B", "C"} else D_RUN)
            / arm
            / f"dense_full_step_{step}"
        ),
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct" if arm in {"B", "C"} else "Qwen/Qwen2.5-1.5B-Instruct",
        "surface": "serialized_orbit_scratchpad" if arm == "C" else "direct",
        "max_new_tokens": 96 if arm == "C" else 32,
        "batch_size": 16 if arm in {"B", "C"} else 8,
        "reference_summary": str(
            ROOT
            / "outputs/stage5"
            / (BC_RUN if arm in {"B", "C"} else D_RUN)
            / "eval"
            / arm
            / "summary.json"
        )
        if step == 4000
        else None,
    }
    for arm in "BCD"
    for step in (2000, 4000)
]

# This is a reproducibility receipt, not a performance gate. Greedy BF16 GPU
# generation can vary by a few rows across independent process/model loads.
GPU_REPEATABILITY_MAX_TOTAL_CORRECT_DELTA = 4
GPU_REPEATABILITY_MAX_DEPTH_CORRECT_DELTA = 3
GPU_REPEATABILITY_MAX_DEPTH_PARSE_FAILURE_DELTA = 1


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sign_test_two_sided(helped: int, hurt: int) -> float:
    non_ties = int(helped) + int(hurt)
    if non_ties == 0:
        return 1.0
    tail = sum(math.comb(non_ties, k) for k in range(0, min(helped, hurt) + 1)) / (2**non_ties)
    return min(1.0, 2.0 * tail)


def paired_delta(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_by_id = {str(row["id"]): row for row in left}
    right_by_id = {str(row["id"]): row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise RuntimeError("Paired evaluation row IDs differ")
    helped = hurt = tied = 0
    by_depth: dict[str, dict[str, int]] = {}
    for row_id, left_row in left_by_id.items():
        right_row = right_by_id[row_id]
        if (left_row["depth"], left_row["target"]) != (right_row["depth"], right_row["target"]):
            raise RuntimeError(f"Paired row metadata differs for {row_id}")
        delta = int(bool(left_row["correct"])) - int(bool(right_row["correct"]))
        bucket = by_depth.setdefault(str(int(left_row["depth"])), {"helped": 0, "hurt": 0, "tied": 0})
        key = "helped" if delta > 0 else "hurt" if delta < 0 else "tied"
        bucket[key] += 1
        if delta > 0:
            helped += 1
        elif delta < 0:
            hurt += 1
        else:
            tied += 1
    return {
        "helped": helped,
        "hurt": hurt,
        "tied": tied,
        "net_correct": helped - hurt,
        "two_sided_sign_p": sign_test_two_sided(helped, hurt),
        "by_depth": by_depth,
    }


def _depth2_classes(rows: list[dict[str, Any]], source_by_id: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = {"correct": 0, "one_step_early": 0, "earlier_orbit": 0, "other": 0, "parse_failure": 0}
    for row in rows:
        if int(row["depth"]) != 2:
            continue
        prediction = row.get("prediction")
        source = source_by_id[str(row["id"])]
        orbit = [str(value).strip().upper() for value in source["orbit"]]
        if prediction is None:
            counts["parse_failure"] += 1
        elif bool(row["correct"]):
            counts["correct"] += 1
        elif prediction == orbit[-2]:
            counts["one_step_early"] += 1
        elif prediction in orbit[:-1]:
            counts["earlier_orbit"] += 1
        else:
            counts["other"] += 1
    return counts


def build_comparison(
    rows_by_label: dict[str, list[dict[str, Any]]],
    source_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = {spec["label"] for spec in CHECKPOINT_SPECS}
    if set(rows_by_label) != expected:
        raise RuntimeError(f"Checkpoint comparison is incomplete: {sorted(set(rows_by_label) ^ expected)}")
    source_by_id = {str(row.get("id") or row.get("instance_id")): row for row in source_rows}
    reference_ids = [str(row["id"]) for row in rows_by_label["B_step2000"]]
    settings = {label: summarize_rows(rows) for label, rows in rows_by_label.items()}
    within_arm = {
        arm: paired_delta(rows_by_label[f"{arm}_step4000"], rows_by_label[f"{arm}_step2000"])
        for arm in "BCD"
    }
    step4000 = {
        "C_minus_B": paired_delta(rows_by_label["C_step4000"], rows_by_label["B_step4000"]),
        "C_minus_D": paired_delta(rows_by_label["C_step4000"], rows_by_label["D_step4000"]),
        "D_minus_B": paired_delta(rows_by_label["D_step4000"], rows_by_label["B_step4000"]),
    }
    depth2 = {
        label: _depth2_classes(rows_by_label[label], source_by_id)
        for label in ("C_step2000", "C_step4000")
    }
    keyed = {
        label: {str(row["id"]): row for row in rows}
        for label, rows in rows_by_label.items()
    }
    paired_rows = []
    for row_id in reference_ids:
        reference = keyed["B_step2000"][row_id]
        if row_id not in source_by_id:
            raise RuntimeError(f"Frozen source row missing {row_id}")
        paired_rows.append(
            {
                "id": row_id,
                "depth": int(reference["depth"]),
                "target": reference["target"],
                "orbit": source_by_id[row_id]["orbit"],
                "predictions": {label: keyed[label][row_id].get("prediction") for label in sorted(keyed)},
                "correct": {label: bool(keyed[label][row_id]["correct"]) for label in sorted(keyed)},
            }
        )
    return {
        "kind": "stage5_phase_a_checkpoint_comparison",
        "settings": settings,
        "within_arm": within_arm,
        "step4000_pairwise": step4000,
        "depth2_error_classes": depth2,
    }, paired_rows


def _validate_checkpoint(spec: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(spec["checkpoint"])
    metadata_path = checkpoint / "stage5_dense_full_metadata.json"
    if not (checkpoint / "config.json").exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing dense checkpoint or metadata: {checkpoint}")
    metadata = read_json(metadata_path)
    config = metadata.get("config") or {}
    checks = {
        "step": int(metadata.get("step", -1)) == int(spec["step"]),
        "model_name": config.get("model_name") == spec["model_name"],
        "revision": config.get("revision") == MODEL_REVISIONS[spec["model_name"]],
        "surface": config.get("training_surface") == spec["surface"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Checkpoint metadata mismatch for {spec['label']}: {checks}")
    return {"checkpoint": str(checkpoint), "checks": checks, "metadata": metadata}


def build_repeatability_receipt(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    current_depth = current.get("by_depth") or {}
    reference_depth = reference.get("by_depth") or {}
    depth_keys_match = set(current_depth) == set(reference_depth)
    depth_total_checks = {
        depth: int(current_depth[depth].get("total", -1)) == int(reference_depth[depth].get("total", -2))
        for depth in sorted(set(current_depth) & set(reference_depth), key=int)
    }
    structural_checks = {
        "reader": current.get("reader") == reference.get("reader"),
        "total": int(current.get("total", -1)) == int(reference.get("total", -2)),
        "max_new_tokens": int(current.get("max_new_tokens", -1)) == int(reference.get("max_new_tokens", -2)),
        "depth_keys": depth_keys_match,
        "depth_totals": bool(depth_total_checks) and all(depth_total_checks.values()),
    }
    structural_checks_pass = all(structural_checks.values())
    correct_delta = int(current.get("correct", 0)) - int(reference.get("correct", 0))
    depth_correct_deltas = {
        depth: int(current_depth[depth].get("correct", 0)) - int(reference_depth[depth].get("correct", 0))
        for depth in sorted(set(current_depth) & set(reference_depth), key=int)
    }
    depth_parse_failure_deltas = {
        depth: int(current_depth[depth].get("parse_failures", 0))
        - int(reference_depth[depth].get("parse_failures", 0))
        for depth in sorted(set(current_depth) & set(reference_depth), key=int)
    }
    max_abs_depth_correct_delta = max((abs(value) for value in depth_correct_deltas.values()), default=0)
    max_abs_depth_parse_failure_delta = max(
        (abs(value) for value in depth_parse_failure_deltas.values()), default=0
    )
    exact = (
        structural_checks_pass
        and correct_delta == 0
        and max_abs_depth_correct_delta == 0
        and max_abs_depth_parse_failure_delta == 0
    )
    within_envelope = (
        structural_checks_pass
        and abs(correct_delta) <= GPU_REPEATABILITY_MAX_TOTAL_CORRECT_DELTA
        and max_abs_depth_correct_delta <= GPU_REPEATABILITY_MAX_DEPTH_CORRECT_DELTA
        and max_abs_depth_parse_failure_delta <= GPU_REPEATABILITY_MAX_DEPTH_PARSE_FAILURE_DELTA
    )
    status = (
        "exact"
        if exact
        else "within_gpu_repeatability_envelope"
        if within_envelope
        else "outside_gpu_repeatability_envelope"
    )
    return {
        "status": status,
        "exact": exact,
        "structural_checks_pass": structural_checks_pass,
        "structural_checks": structural_checks,
        "correct_delta": correct_delta,
        "accuracy_delta": float(current.get("accuracy", 0.0)) - float(reference.get("accuracy", 0.0)),
        "depth_correct_deltas": depth_correct_deltas,
        "depth_parse_failure_deltas": depth_parse_failure_deltas,
        "max_abs_depth_correct_delta": max_abs_depth_correct_delta,
        "max_abs_depth_parse_failure_delta": max_abs_depth_parse_failure_delta,
        "envelope": {
            "max_total_correct_delta": GPU_REPEATABILITY_MAX_TOTAL_CORRECT_DELTA,
            "max_depth_correct_delta": GPU_REPEATABILITY_MAX_DEPTH_CORRECT_DELTA,
            "max_depth_parse_failure_delta": GPU_REPEATABILITY_MAX_DEPTH_PARSE_FAILURE_DELTA,
        },
    }


def _step4000_repeatability_receipt(spec: dict[str, Any], current_path: Path) -> dict[str, Any] | None:
    reference_path = spec.get("reference_summary")
    if not reference_path:
        return None
    receipt = build_repeatability_receipt(read_json(current_path), read_json(reference_path))
    if receipt["status"] == "outside_gpu_repeatability_envelope":
        raise RuntimeError(f"Step-4000 repeatability failed for {spec['label']}: {receipt}")
    print(
        f"[receipt-ok] step4000_repeatability={spec['label']} status={receipt['status']} "
        f"correct_delta={receipt['correct_delta']} "
        f"max_depth_delta={receipt['max_abs_depth_correct_delta']}",
        flush=True,
    )
    return receipt


def _run_stream(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def _gzip_rows(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, destination.open("wb") as output_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output_handle, mtime=0) as compressed:
            shutil.copyfileobj(input_handle, compressed)


def _compress_completed_rows(eval_dir: Path) -> bool:
    """Finish an interrupted eval after summary/raw rows were already written."""
    summary = eval_dir / "summary.json"
    raw = eval_dir / "rows.jsonl"
    compressed = eval_dir / "rows.jsonl.gz"
    if not summary.exists() or not raw.exists() or compressed.exists():
        return False
    _gzip_rows(raw, compressed)
    raw.unlink()
    print(f"[resume] compressed completed rows for {eval_dir.name}", flush=True)
    return True


def _read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _publish(run_dir: Path, message: str) -> None:
    subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=ROOT, check=False)
    paths = list(run_dir.glob("eval/*/summary.json"))
    paths.extend(run_dir.glob("eval/*/rows.jsonl.gz"))
    paths.extend([run_dir / "summary.json", run_dir / "summary.md", run_dir / "paired_rows.jsonl"])
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


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Phase A Checkpoint Comparison - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        "",
        "| Setting | Correct | Total | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for label, summary in sorted((payload.get("comparison") or {}).get("settings", {}).items()):
        lines.append(f"| {label} | {summary['correct']} | {summary['total']} | {summary['accuracy']:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    if sha256_jsonl_content(EVAL_SOURCE) != EVAL_SHA256:
        raise RuntimeError("Phase A frozen eval rows do not match the locked content SHA256")
    run_id = os.environ.get("STAGE5_PHASE_A_COMPARISON_RUN_ID") or time.strftime(
        "stage5_phase_a_checkpoint_comparison_%Y%m%d_%H%M%S"
    )
    run_dir = ROOT / "outputs/stage5" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    payload = read_json(summary_path) if summary_path.exists() else {
        "kind": "stage5_phase_a_checkpoint_comparison_run",
        "run_id": run_id,
        "status": "started",
        "eval_source": path_for_cli(EVAL_SOURCE),
        "eval_sha256": EVAL_SHA256,
        "checkpoints": {},
    }
    payload.setdefault("repeatability_receipts", {})
    for spec in CHECKPOINT_SPECS:
        label = spec["label"]
        receipt = _validate_checkpoint(spec)
        payload["checkpoints"][label] = receipt
        eval_dir = run_dir / "eval" / label
        summary = eval_dir / "summary.json"
        compressed = eval_dir / "rows.jsonl.gz"
        completed_from_raw = _compress_completed_rows(eval_dir)
        evaluated = False
        if not summary.exists() or not compressed.exists():
            raw = eval_dir / "rows.jsonl"
            payload["status"] = f"evaluating_{label}"
            write_json(summary_path, payload)
            _run_stream(
                [
                    sys.executable,
                    "eval/eval_synthetic_depth_dense.py",
                    "--data_jsonl",
                    path_for_cli(EVAL_SOURCE),
                    "--checkpoint",
                    spec["checkpoint"],
                    "--output_jsonl",
                    path_for_cli(raw),
                    "--output_summary",
                    path_for_cli(summary),
                    "--batch_size",
                    str(spec["batch_size"]),
                    "--max_new_tokens",
                    str(spec["max_new_tokens"]),
                    "--dtype",
                    "bfloat16",
                    "--device",
                    "cuda",
                ]
            )
            _gzip_rows(raw, compressed)
            raw.unlink()
            evaluated = True
        repeatability = _step4000_repeatability_receipt(spec, summary)
        if repeatability is not None:
            payload["repeatability_receipts"][label] = repeatability
            write_json(summary_path, payload)
        if evaluated or completed_from_raw:
            _publish(run_dir, f"Record Phase A checkpoint evaluation {label} {run_id} [skip ci]")

    rows_by_label = {
        spec["label"]: _read_gzip_jsonl(run_dir / "eval" / spec["label"] / "rows.jsonl.gz")
        for spec in CHECKPOINT_SPECS
    }
    comparison, paired = build_comparison(rows_by_label, read_jsonl(EVAL_SOURCE))
    payload["status"] = "finished"
    payload["comparison"] = comparison
    write_json(summary_path, payload)
    (run_dir / "paired_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in paired), encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(_markdown(payload), encoding="utf-8")
    _publish(run_dir, f"Finish Phase A checkpoint comparison {run_id} [skip ci]")
    print(json.dumps(comparison, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
