"""Run corrected natural-surface robustness receipts and state probes.

This is the eval-only half of plan items 2-3. It deliberately does not open
the untouched depth-13-to-16 band. Every unit publishes independently and the
plan skips units already marked finished, making Colab restarts inexpensive.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(os.environ.get("STAGE5_ROOT", "/content/recurrent-qwen-svgd"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_natural_surface_receipts import (  # noqa: E402
    checkpoint_map,
    read_jsonl,
    run_active_eval,
)
from colab.run_stage5_natural_surface_transfer import (  # noqa: E402
    path_for_cli,
    publish_run,
    read_json,
    root_path,
    write_json,
    write_jsonl,
)
from eval.eval_synthetic_depth_active_labels import (  # noqa: E402
    candidates_for_row,
    prompt_for_row,
    single_token_candidate_ids,
)
from training.natural_surface_transfer import (  # noqa: E402
    CORRECTED_HELDOUT_SINGLE_TOKEN_NAMES,
    DEFAULT_NAME_SYMBOLS,
    build_verbal_rows,
    manifest_for_rows,
    verify_single_token_names,
)


SOURCE_RECEIPT_RUN_ID = "stage5_natural_surface_receipts_20260709_210151"
DEFAULT_EVAL_VARIANTS = (
    "corrected_relay_unseen_single_token_d1_12",
    "corrected_pointer_unseen_single_token_d1_12",
    "robust_relay_fronted_d1_12",
    "robust_pointer_fronted_d1_12",
    "robust_baton_fronted_d1_12",
    "robust_baton_passive_d1_12",
)
DEFAULT_PROBE_FAMILIES = ("paired_relay", "paired_pointer", "baton_default")


def csv_items(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def finished(record: Any) -> bool:
    return isinstance(record, dict) and record.get("status") == "finished"


def build_followup_plan(
    *,
    checkpoint_labels: list[str],
    eval_variants: list[str] | tuple[str, ...],
    probe_families: list[str] | tuple[str, ...],
    payload: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, str]]]:
    payload = payload or {}
    active = []
    probes = []
    for checkpoint in checkpoint_labels:
        for variant in eval_variants:
            if not finished(payload.get("active_evals", {}).get(checkpoint, {}).get(variant)):
                active.append({"checkpoint": checkpoint, "variant": variant})
        for family in probe_families:
            if not finished(payload.get("probes", {}).get(checkpoint, {}).get(family)):
                probes.append({"checkpoint": checkpoint, "family": family})
    return {"active": active, "probes": probes}


def stratified_depth_subset(rows: list[dict[str, Any]], *, depths: range, rows_per_depth: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for depth in depths:
        bucket = [row for row in rows if int(row["depth"]) == depth]
        if len(bucket) < rows_per_depth:
            raise ValueError(f"Need {rows_per_depth} rows at depth {depth}; found {len(bucket)}")
        selected.extend(bucket[:rows_per_depth])
    return selected


def verify_corrected_names(tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    verdict = verify_single_token_names(
        tokenizer,
        symbol_names=CORRECTED_HELDOUT_SINGLE_TOKEN_NAMES,
    )
    overlap = sorted(set(CORRECTED_HELDOUT_SINGLE_TOKEN_NAMES) & set(DEFAULT_NAME_SYMBOLS))
    if overlap:
        raise RuntimeError(f"Corrected held-out names overlap training names: {overlap}")
    if not verdict["all_single_token"]:
        raise RuntimeError(f"Corrected held-out names are not tokenizer-valid: {verdict}")
    prompt = prompt_for_row(rows[0], prediction_space="full_symbols", prompt_style="question_only")
    candidates = candidates_for_row(rows[0], prediction_space="full_symbols", value_prefix="name:")
    suffix_ids = single_token_candidate_ids(tokenizer, prompt, candidates)
    if suffix_ids is None or len(suffix_ids) != len(CORRECTED_HELDOUT_SINGLE_TOKEN_NAMES):
        raise RuntimeError("Corrected name set does not stay on the evaluator's single-token fast path")
    return {
        **verdict,
        "disjoint_from_training_names": True,
        "prompt_suffix_fast_path": True,
        "suffix_token_ids": suffix_ids,
    }


def materialize_data(out_dir: Path, source_receipt: dict[str, Any], *, seed: int) -> dict[str, Any]:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    files = dict(source_receipt["generated_data"]["files"])
    manifests: dict[str, Any] = {}

    corrected_rows: dict[str, list[dict[str, Any]]] = {}
    for family in ("relay", "pointer"):
        label = f"corrected_{family}_unseen_single_token_d1_12"
        rows = build_verbal_rows(
            family=family,  # type: ignore[arg-type]
            split="test",
            n_symbols=20,
            max_depth=12,
            rows_per_depth=128,
            seed=seed + (0 if family == "relay" else 1_000),
            max_target_loops=12,
            symbol_names=CORRECTED_HELDOUT_SINGLE_TOKEN_NAMES,
        )
        path = data_dir / f"{label}.jsonl"
        write_jsonl(path, rows)
        files[label] = path_for_cli(path)
        manifests[label] = manifest_for_rows(rows)
        corrected_rows[label] = rows

    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"))
    token_receipt = verify_corrected_names(tokenizer, corrected_rows["corrected_relay_unseen_single_token_d1_12"])

    probe_sources = {
        "paired_relay": "paired_relay_d1_12",
        "paired_pointer": "paired_pointer_d1_12",
        "baton_default": "robust_baton_default_d1_12",
    }
    probe_files: dict[str, str] = {}
    for family, source_label in probe_sources.items():
        rows = read_jsonl(files[source_label])
        subset = stratified_depth_subset(rows, depths=range(9, 13), rows_per_depth=32)
        path = data_dir / f"probe_{family}_d9_12_n128.jsonl"
        write_jsonl(path, subset)
        probe_files[family] = path_for_cli(path)
        manifests[f"probe_{family}_d9_12_n128"] = manifest_for_rows(subset)

    return {
        "files": files,
        "probe_files": probe_files,
        "manifests": manifests,
        "corrected_name_tokenizer_receipt": token_receipt,
        "untouched_depth_13_16_opened": False,
    }


def run_probe(
    out_dir: Path,
    *,
    checkpoint: str,
    checkpoint_label: str,
    family: str,
    data_jsonl: str,
    dtype: str,
) -> dict[str, Any]:
    probe_dir = out_dir / "probes" / checkpoint_label
    probe_dir.mkdir(parents=True, exist_ok=True)
    summary = probe_dir / f"{family}_probe_summary.json"
    log = probe_dir / f"{family}_probe.log"
    cmd = [
        sys.executable,
        "eval/eval_synthetic_depth_probe.py",
        "--data_jsonl",
        data_jsonl,
        "--checkpoint",
        checkpoint,
        "--output_summary",
        path_for_cli(summary),
        "--loop_counts",
        "1,4,8,12",
        "--target_steps",
        "1,4,8,12",
        "--n_symbols",
        "20",
        "--prompt_style",
        "question_only",
        "--value_prefix",
        "name:",
        "--permutations",
        os.environ.get("STAGE5_NATURAL_FOLLOWUP_PROBE_PERMUTATIONS", "20"),
        "--feature_transforms",
        "raw,unit_norm,rms_norm",
        "--envelope_fit_loop_max",
        "4",
        "--envelope_fit_depth_max",
        "12",
        "--split",
        "6,18",
        "--bridge_projection_mode",
        "split",
        "--lora_rank",
        "0",
        "--dtype",
        dtype,
        "--adapter_dtype",
        "float32",
        "--device",
        os.environ.get("DEVICE", "cuda"),
    ]
    print("$", " ".join(cmd), flush=True)
    process = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(process.stdout, encoding="utf-8")
    print(process.stdout, flush=True)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, cmd, output=process.stdout)
    probe = read_json(summary)
    return {
        "status": "finished",
        "family": family,
        "checkpoint": checkpoint,
        "data_jsonl": data_jsonl,
        "summary": path_for_cli(summary),
        "log": path_for_cli(log),
        "loop_index_probe": probe.get("loop_index_probe"),
        "state_envelope": probe.get("state_envelope"),
        "feature_transform_probes": probe.get("feature_transform_probes"),
        "router_leak_exclusion": probe.get("router_leak_exclusion"),
    }


def write_markdown(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Natural-Surface Follow-ups 2-3 - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Corrected tokenizer receipt: `{payload.get('data', {}).get('corrected_name_tokenizer_receipt')}`",
        f"- Untouched D13-16 opened: `{payload.get('data', {}).get('untouched_depth_13_16_opened')}`",
        "",
        "## Active Evals",
        "",
        f"`{payload.get('active_evals', {})}`",
        "",
        "## Probes",
        "",
        f"`{payload.get('probes', {})}`",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def publish(out_dir: Path, payload: dict[str, Any], message: str) -> None:
    write_json(out_dir / "summary.json", payload)
    write_markdown(out_dir, payload)
    publish_run(out_dir, message=message, update_pointer=False)


def main() -> int:
    run_id = os.environ.get("STAGE5_NATURAL_FOLLOWUP_RUN_ID", "stage5_natural_surface_followups_2_3_20260710")
    out_dir = ROOT / "outputs" / "stage5" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        payload = read_json(summary_path)
    else:
        payload = {
            "kind": "stage5_natural_surface_followups_2_3",
            "run_id": run_id,
            "status": "started",
            "source_receipt_run_id": os.environ.get("STAGE5_NATURAL_FOLLOWUP_SOURCE_RUN_ID", SOURCE_RECEIPT_RUN_ID),
            "active_evals": {},
            "probes": {},
        }
        publish(out_dir, payload, f"Start natural-surface follow-ups {run_id} [skip ci]")

    source_receipt_path = ROOT / "outputs" / "stage5" / payload["source_receipt_run_id"] / "summary.json"
    source_receipt = read_json(source_receipt_path)
    source_summary = read_json(source_receipt["source_summary"])
    backup_manifest = read_json(source_receipt["backup_manifest"])
    checkpoints = checkpoint_map(source_summary, backup_manifest)
    if not payload.get("data"):
        payload["data"] = materialize_data(
            out_dir,
            source_receipt,
            seed=int(os.environ.get("STAGE5_NATURAL_FOLLOWUP_SEED", "941337")),
        )
        payload["status"] = "data_verified"
        publish(out_dir, payload, f"Verify natural-surface follow-up data {run_id} [skip ci]")

    checkpoint_labels = csv_items(
        os.environ.get("STAGE5_NATURAL_FOLLOWUP_CHECKPOINTS", "frozen_n24,step_2000,step_4000,step_6000")
    )
    eval_variants = csv_items(
        os.environ.get("STAGE5_NATURAL_FOLLOWUP_VARIANTS", ",".join(DEFAULT_EVAL_VARIANTS))
    )
    probe_families = csv_items(
        os.environ.get("STAGE5_NATURAL_FOLLOWUP_PROBE_FAMILIES", ",".join(DEFAULT_PROBE_FAMILIES))
    )
    for label in checkpoint_labels:
        if label not in checkpoints:
            raise KeyError(f"Missing checkpoint {label!r}; have {sorted(checkpoints)}")
    plan = build_followup_plan(
        checkpoint_labels=checkpoint_labels,
        eval_variants=eval_variants,
        probe_families=probe_families,
        payload=payload,
    )
    payload["plan"] = plan
    payload["status"] = "running"
    publish(out_dir, payload, f"Record natural-surface follow-up plan {run_id} [skip ci]")
    dtype = os.environ.get("STAGE5_NATURAL_FOLLOWUP_DTYPE", "bfloat16")

    for item in plan["active"]:
        checkpoint_label = item["checkpoint"]
        variant = item["variant"]
        result = run_active_eval(
            out_dir / "active" / checkpoint_label,
            label=f"{checkpoint_label}_{variant}",
            checkpoint=checkpoints[checkpoint_label],
            data_jsonl=payload["data"]["files"][variant],
            loop_counts="1,2,3,4,5,6,7,8,9,10,11,12",
            value_prefix="name:",
            dtype=dtype,
        )
        payload.setdefault("active_evals", {}).setdefault(checkpoint_label, {})[variant] = {
            "status": "finished",
            **result,
        }
        payload["status"] = f"active_{checkpoint_label}_{variant}"
        publish(out_dir, payload, f"Record follow-up active eval {run_id} {checkpoint_label} {variant} [skip ci]")

    for item in plan["probes"]:
        checkpoint_label = item["checkpoint"]
        family = item["family"]
        result = run_probe(
            out_dir,
            checkpoint=checkpoints[checkpoint_label],
            checkpoint_label=checkpoint_label,
            family=family,
            data_jsonl=payload["data"]["probe_files"][family],
            dtype=dtype,
        )
        payload.setdefault("probes", {}).setdefault(checkpoint_label, {})[family] = result
        payload["status"] = f"probe_{checkpoint_label}_{family}"
        publish(out_dir, payload, f"Record follow-up probe {run_id} {checkpoint_label} {family} [skip ci]")

    payload["status"] = "finished"
    payload["plan"] = build_followup_plan(
        checkpoint_labels=checkpoint_labels,
        eval_variants=eval_variants,
        probe_families=probe_families,
        payload=payload,
    )
    publish(out_dir, payload, f"Finish natural-surface follow-ups {run_id} [skip ci]")
    print(json.dumps({"run_id": run_id, "summary": path_for_cli(summary_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
