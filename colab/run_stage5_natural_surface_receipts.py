"""Natural-surface transfer receipts and robustness diagnostics.

This runner consolidates the rung-zero result without doing more training:

* records whether pointer was held out from the training mix;
* freezes the untouched verbal tail band (depths 13-16) before keeper selection;
* generates robustness/diagnosis sets: third family, unseen names, fixed
  syntactic variants, and paired relay/pointer renderings;
* optionally evaluates backed-up checkpoints on those sets and publishes
  compact summaries incrementally.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("STAGE5_ROOT", "/content/recurrent-qwen-svgd"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colab.run_stage5_natural_surface_transfer import (  # noqa: E402
    DEFAULT_DATA_SUMMARY,
    compact_rows,
    env_flag,
    path_for_cli,
    publish_run,
    read_json,
    root_path,
    write_jsonl,
)
from training.natural_surface_transfer import (  # noqa: E402
    DEFAULT_NAME_SYMBOLS,
    HELDOUT_NAME_SYMBOLS,
    build_paired_verbal_rows,
    build_verbal_rows,
    manifest_for_rows,
)


SOURCE_RUN_ID = "stage5_natural_surface_transfer_rung0_fixed_prompt_20260709_133812"
CURVE_RUN_ID = "stage5_natural_surface_transfer_rung0_fixed_prompt_20260709_133812_checkpoint_curve_20260709_163008"
BACKUP_MANIFEST = f"outputs/stage5/{CURVE_RUN_ID}/drive_backup_manifest.json"


def redact(text: str) -> str:
    out = str(text)
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(name)
        if token:
            out = out.replace(token, "****")
    return out


def run(cmd: list[str | os.PathLike[str]], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> str:
    print("$", redact(" ".join(map(str, cmd))), flush=True)
    process = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        safe = redact(line)
        print(safe, end="", flush=True)
        chunks.append(safe)
    stdout = "".join(chunks)
    rc = process.wait()
    if rc:
        print("FAILED_COMMAND_TAIL_START", flush=True)
        print("\n".join(stdout.splitlines()[-160:]), flush=True)
        print("FAILED_COMMAND_TAIL_END", flush=True)
        raise subprocess.CalledProcessError(rc, list(map(str, cmd)), output=stdout)
    return stdout


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = root_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in root_path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def filter_depths(rows: list[dict[str, Any]], depths: range) -> list[dict[str, Any]]:
    depth_set = set(depths)
    return [row for row in rows if int(row["depth"]) in depth_set]


def family_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        family = row.get("verbal_surface_family") or row.get("curriculum_family") or "unknown"
        counts[str(family)] += 1
    return dict(sorted(counts.items()))


def checkpoint_map(summary: dict[str, Any], backup_manifest: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    init = summary.get("init_checkpoint_metadata") or {}
    if init.get("selected_checkpoint_source"):
        out["frozen_n24"] = str(init["selected_checkpoint_source"])
    for row in backup_manifest.get("checkpoint_files", []):
        out[f"step_{int(row['step'])}"] = str(row["dest"])
    return out


def materialize_datasets(out_dir: Path, *, seed: int) -> dict[str, Any]:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    manifests: dict[str, Any] = {}

    def add(label: str, rows: list[dict[str, Any]]) -> None:
        path = data_dir / f"{label}.jsonl"
        write_jsonl(path, rows)
        files[label] = path_for_cli(path)
        manifests[label] = manifest_for_rows(rows)

    untouched_relay = filter_depths(
        build_verbal_rows(
            family="relay",
            split="test",
            n_symbols=20,
            max_depth=16,
            rows_per_depth=128,
            seed=seed + 13_000,
            max_target_loops=16,
        ),
        range(13, 17),
    )
    untouched_pointer = filter_depths(
        build_verbal_rows(
            family="pointer",
            split="test",
            n_symbols=20,
            max_depth=16,
            rows_per_depth=128,
            seed=seed + 14_000,
            max_target_loops=16,
        ),
        range(13, 17),
    )
    add("untouched_relay_d13_16", untouched_relay)
    add("untouched_pointer_d13_16", untouched_pointer)

    for family in ("baton", "relay", "pointer"):
        if family == "baton":
            add(
                "robust_baton_default_d1_12",
                build_verbal_rows(
                    family="baton",
                    split="test",
                    n_symbols=20,
                    max_depth=12,
                    rows_per_depth=128,
                    seed=seed + 20_000,
                    max_target_loops=12,
                ),
            )
        for variant in ("passive", "fronted"):
            add(
                f"robust_{family}_{variant}_d1_12",
                build_verbal_rows(
                    family=family,  # type: ignore[arg-type]
                    split="test",
                    n_symbols=20,
                    max_depth=12,
                    rows_per_depth=64,
                    seed=seed + 30_000 + len(files) * 101,
                    max_target_loops=12,
                    template_variant=variant,  # type: ignore[arg-type]
                ),
            )

    for family in ("relay", "pointer"):
        add(
            f"robust_{family}_unseen_names_d1_12",
            build_verbal_rows(
                family=family,  # type: ignore[arg-type]
                split="test",
                n_symbols=20,
                max_depth=12,
                rows_per_depth=128,
                seed=seed + 40_000 + (0 if family == "relay" else 1_000),
                max_target_loops=12,
                symbol_names=HELDOUT_NAME_SYMBOLS,
            ),
        )

    paired = build_paired_verbal_rows(
        families=("relay", "pointer"),
        split="test",
        n_symbols=20,
        max_depth=12,
        rows_per_depth=64,
        seed=seed + 50_000,
        max_target_loops=12,
    )
    add("paired_relay_d1_12", paired["relay"])
    add("paired_pointer_d1_12", paired["pointer"])

    return {
        "files": files,
        "manifests": manifests,
        "untouched_policy": {
            "depths": [13, 14, 15, 16],
            "selection_hygiene": "Generated and hashed before keeper selection; do not evaluate until keepers are chosen.",
        },
        "symbol_sets": {
            "default": list(DEFAULT_NAME_SYMBOLS),
            "heldout": list(HELDOUT_NAME_SYMBOLS),
        },
    }


def summarize_active_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_depth: dict[str, dict[str, int]] = {}
    confusion: Counter[str] = Counter()
    paired: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row.get("active_cell"):
            continue
        if int(row.get("loop", -1)) != int(row["depth"]):
            continue
        depth = str(int(row["depth"]))
        bucket = by_depth.setdefault(depth, {"correct": 0, "total": 0})
        hit = bool(row.get("hit"))
        bucket["correct"] += int(hit)
        bucket["total"] += 1
        pred = str(row.get("prediction"))
        target = str(row.get("target"))
        if not hit:
            if pred in (row.get("symbol_names") or []):
                confusion["wrong_entity_name"] += 1
            elif pred:
                confusion["other_symbol"] += 1
            else:
                confusion["missing"] += 1
        if row.get("paired_instance_id"):
            paired[str(row["paired_instance_id"])] = {
                "id": row.get("id"),
                "depth": int(row["depth"]),
                "hit": hit,
                "prediction": pred,
                "target": target,
            }
    diag = {
        depth: counts["correct"] / counts["total"] if counts["total"] else 0.0
        for depth, counts in sorted(by_depth.items(), key=lambda item: int(item[0]))
    }
    return {
        "by_depth": {
            depth: {**counts, "accuracy": diag[depth]}
            for depth, counts in sorted(by_depth.items(), key=lambda item: int(item[0]))
        },
        "active_diagonal": diag,
        "active_min_1_8": min([diag[str(depth)] for depth in range(1, 9) if str(depth) in diag], default=0.0),
        "active_min_9_12": min([diag[str(depth)] for depth in range(9, 13) if str(depth) in diag], default=0.0),
        "failure_profile": dict(confusion),
        "paired_hits": paired,
    }


def run_active_eval(
    eval_dir: Path,
    *,
    label: str,
    checkpoint: str,
    data_jsonl: str,
    loop_counts: str,
    value_prefix: str,
    dtype: str,
    keep_rows: bool = False,
) -> dict[str, Any]:
    rows_path = eval_dir / f"{label}_active_rows.jsonl"
    summary_path = eval_dir / f"{label}_active_summary.json"
    log_path = eval_dir / f"{label}_active_eval.log"
    stdout = run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_active_labels.py",
            "--data_jsonl",
            data_jsonl,
            "--checkpoint",
            checkpoint,
            "--output_jsonl",
            path_for_cli(rows_path),
            "--output_summary",
            path_for_cli(summary_path),
            "--loop_counts",
            loop_counts,
            "--threshold",
            "0.71",
            "--prediction_space",
            "full_symbols",
            "--prompt_style",
            "question_only",
            "--value_prefix",
            value_prefix,
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
            "--progress_every",
            os.environ.get("STAGE5_NATURAL_RECEIPTS_PROGRESS_EVERY", "128"),
        ]
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(stdout, encoding="utf-8")
    rows = read_jsonl(rows_path)
    summary = read_json(summary_path)
    compact = summarize_active_rows(rows)
    if keep_rows:
        row_manifest = {"status": "kept_full", "full_rows_path": path_for_cli(rows_path), "full_rows_count": len(rows)}
    else:
        row_manifest = compact_rows(
            rows_path,
            sample_rows=int(os.environ.get("STAGE5_NATURAL_RECEIPTS_SAMPLE_ROWS", "128")),
        )
    return {
        "label": label,
        "checkpoint": checkpoint,
        "data_jsonl": data_jsonl,
        "active_summary": path_for_cli(summary_path),
        "eval_log": path_for_cli(log_path),
        "row_manifest": row_manifest,
        "active_diagonal": summary.get("active_diagonal", {}),
        "active_total": summary.get("active_total", {}),
        "above_diagonal": summary.get("above_diagonal", {}),
        "compact": {key: value for key, value in compact.items() if key != "paired_hits"},
        "paired_hits": compact.get("paired_hits", {}),
    }


def run_same_reader_eval(
    eval_dir: Path,
    *,
    label: str,
    checkpoint: str,
    data_jsonl: str,
    max_loops: int,
    value_prefix: str,
    dtype: str,
) -> dict[str, Any]:
    rows_path = eval_dir / f"{label}_same_reader_rows.jsonl"
    summary_path = eval_dir / f"{label}_same_reader_summary.json"
    stdout = run(
        [
            sys.executable,
            "eval/eval_synthetic_depth_final_symbol.py",
            "--data_jsonl",
            data_jsonl,
            "--checkpoint",
            checkpoint,
            "--output_jsonl",
            path_for_cli(rows_path),
            "--output_summary",
            path_for_cli(summary_path),
            "--max_loops",
            str(max_loops),
            "--threshold",
            "0.71",
            "--prompt_style",
            "question_only",
            "--value_prefix",
            value_prefix,
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
    )
    (eval_dir / f"{label}_same_reader_eval.log").write_text(stdout, encoding="utf-8")
    summary = read_json(summary_path)
    compact_rows(rows_path, sample_rows=int(os.environ.get("STAGE5_NATURAL_RECEIPTS_SAMPLE_ROWS", "128")))
    return {
        "label": label,
        "checkpoint": checkpoint,
        "data_jsonl": data_jsonl,
        "same_reader_summary": path_for_cli(summary_path),
        "same_reader_total": summary.get("same_reader_total", {}),
        "by_depth": summary.get("by_depth", {}),
        "all_depths_clear_threshold": summary.get("all_depths_clear_threshold"),
    }


def paired_mcnemar(relay_eval: dict[str, Any], pointer_eval: dict[str, Any]) -> dict[str, Any]:
    relay = relay_eval.get("paired_hits", {})
    pointer = pointer_eval.get("paired_hits", {})
    b = c = both = neither = 0
    by_depth: dict[str, Counter[str]] = defaultdict(Counter)
    for key in sorted(set(relay) & set(pointer)):
        r_hit = bool(relay[key]["hit"])
        p_hit = bool(pointer[key]["hit"])
        depth = str(relay[key]["depth"])
        if r_hit and p_hit:
            both += 1
            by_depth[depth]["both"] += 1
        elif r_hit and not p_hit:
            b += 1
            by_depth[depth]["relay_only"] += 1
        elif (not r_hit) and p_hit:
            c += 1
            by_depth[depth]["pointer_only"] += 1
        else:
            neither += 1
            by_depth[depth]["neither"] += 1
    z = (c - b) / max((b + c) ** 0.5, 1e-9)
    return {
        "both_correct": both,
        "relay_only_correct": b,
        "pointer_only_correct": c,
        "neither_correct": neither,
        "z_pointer_minus_relay": z,
        "by_depth": {depth: dict(counter) for depth, counter in sorted(by_depth.items(), key=lambda item: int(item[0]))},
    }


def write_markdown(out_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Natural-Surface Receipts - {payload['run_id']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Source run: `{payload['source_run_id']}`",
        f"- Curve run: `{payload['curve_run_id']}`",
        f"- Pointer holdout: `{payload['receipts'].get('pointer_holdout')}`",
        f"- Init checkpoint: `{payload['receipts'].get('init_checkpoint')}`",
        "",
        "## Frozen Data Manifests",
        "",
        f"`{payload.get('generated_data', {}).get('manifests', {})}`",
        "",
        "## Eval Results",
        "",
        f"`{payload.get('evals', {})}`",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def publish(out_dir: Path, payload: dict[str, Any], *, message: str) -> None:
    write_json(out_dir / "summary.json", payload)
    write_markdown(out_dir, payload)
    publish_run(out_dir, message=message, update_pointer=False)


def main() -> int:
    run_id = os.environ.get("STAGE5_NATURAL_RECEIPTS_RUN_ID") or time.strftime(
        "stage5_natural_surface_receipts_%Y%m%d_%H%M%S"
    )
    out_dir = ROOT / "outputs" / "stage5" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    source_run_id = os.environ.get("STAGE5_NATURAL_RECEIPTS_SOURCE_RUN_ID", SOURCE_RUN_ID)
    curve_run_id = os.environ.get("STAGE5_NATURAL_RECEIPTS_CURVE_RUN_ID", CURVE_RUN_ID)
    source_summary_path = ROOT / "outputs" / "stage5" / source_run_id / "summary.json"
    curve_summary_path = ROOT / "outputs" / "stage5" / curve_run_id / "summary.json"
    backup_manifest_path = root_path(os.environ.get("STAGE5_NATURAL_RECEIPTS_BACKUP_MANIFEST", BACKUP_MANIFEST))
    source_summary = read_json(source_summary_path)
    curve_summary = read_json(curve_summary_path)
    backup_manifest = read_json(backup_manifest_path)
    data_summary = read_json(source_summary.get("data_summary", DEFAULT_DATA_SUMMARY))
    data_summary_inner = read_json(data_summary["data_summary"])
    data_dir = root_path(data_summary["data_summary"]).parent
    train_rows = read_jsonl(data_dir / data_summary_inner["files"]["rung0_train_mix_chain_symbol_sft"])
    train_counts = family_counts(train_rows)
    pointer_rows_in_training = train_counts.get("pointer", 0) + train_counts.get("pointer_verbal", 0)
    pointer_holdout = {
        "pointer_rows_in_training": pointer_rows_in_training,
        "training_family_counts": train_counts,
        "verdict": "pointer_held_out" if pointer_rows_in_training == 0 else "pointer_present_in_training",
    }

    payload: dict[str, Any] = {
        "kind": "stage5_natural_surface_receipts",
        "run_id": run_id,
        "status": "started",
        "source_run_id": source_run_id,
        "curve_run_id": curve_run_id,
        "source_summary": path_for_cli(source_summary_path),
        "curve_summary": path_for_cli(curve_summary_path),
        "backup_manifest": path_for_cli(backup_manifest_path),
        "receipts": {
            "pointer_holdout": pointer_holdout,
            "init_checkpoint": source_summary.get("init_checkpoint_metadata", {}),
            "corrected_prompt_pipeline": {
                "source_run_id": source_run_id,
                "fixed_prompt_run": "fixed_prompt" in source_run_id,
                "artifact_check_pass": bool(
                    source_summary.get("frozen_baseline", {})
                    .get("n24", {})
                    .get("artifact_check", {})
                    .get("pass")
                ),
                "causal_prompt_masking_commit": "087fe65 Mask full rendered prompt in causal dataset",
            },
            "curve_best_by_metric": curve_summary.get("best_by_metric", {}),
            "drive_checkpoint_backup": {
                "backup_id": backup_manifest.get("backup_id"),
                "checkpoint_files": backup_manifest.get("checkpoint_files", []),
            },
            "canary_trace": {
                "status": "not_in_github_artifacts",
                "note": "Training log/summary was not published in the landed source run; checkpoint curve and eval artifacts are present.",
            },
        },
        "generated_data": {},
        "evals": {},
    }
    publish(out_dir, payload, message=f"Record natural-surface receipts start {run_id} [skip ci]")

    payload["generated_data"] = materialize_datasets(
        out_dir,
        seed=int(os.environ.get("STAGE5_NATURAL_RECEIPTS_SEED", "921337")),
    )
    payload["status"] = "generated_frozen_diagnostic_sets"
    publish(out_dir, payload, message=f"Record natural-surface frozen diagnostic sets {run_id} [skip ci]")

    if not env_flag("STAGE5_NATURAL_RECEIPTS_RUN_EVALS", "1"):
        payload["status"] = "finished_data_only"
        publish(out_dir, payload, message=f"Record natural-surface receipts data-only {run_id} [skip ci]")
        return 0

    ckpts = checkpoint_map(source_summary, backup_manifest)
    dtype = os.environ.get("STAGE5_NATURAL_RECEIPTS_DTYPE", "bfloat16")
    eval_specs = []
    variant_names = [
        item.strip()
        for item in os.environ.get(
            "STAGE5_NATURAL_RECEIPTS_VARIANTS",
            "robust_baton_default_d1_12,robust_relay_unseen_names_d1_12,robust_pointer_unseen_names_d1_12,"
            "robust_relay_passive_d1_12,robust_pointer_passive_d1_12,paired_relay_d1_12,paired_pointer_d1_12",
        ).split(",")
        if item.strip()
    ]
    files = payload["generated_data"]["files"]
    for variant in variant_names:
        eval_specs.append((variant, files[variant], "name:", "1,2,3,4,5,6,7,8,9,10,11,12"))
    if env_flag("STAGE5_NATURAL_RECEIPTS_RUN_FULL_SYNTHETIC", "1"):
        eval_specs.append(
            (
                "synthetic_frozen_v3_d1_12",
                "outputs/stage5/stage5_synthetic_depth_frozen_eval_v3_depth22_n24/data/test_chain_mcq.jsonl",
                "letter:",
                "1,2,3,4,5,6,7,8,9,10,11,12",
            )
        )

    checkpoint_labels = [
        item.strip()
        for item in os.environ.get("STAGE5_NATURAL_RECEIPTS_CHECKPOINTS", "frozen_n24,step_2000,step_4000,step_6000").split(",")
        if item.strip()
    ]
    for ckpt_label in checkpoint_labels:
        if ckpt_label not in ckpts:
            raise KeyError(f"Requested checkpoint {ckpt_label!r} not found in {sorted(ckpts)}")
        checkpoint = ckpts[ckpt_label]
        payload["evals"].setdefault(ckpt_label, {})
        for variant, data_jsonl, value_prefix, loop_counts in eval_specs:
            eval_label = f"{ckpt_label}_{variant}"
            keep_rows = variant.startswith("paired_")
            result = run_active_eval(
                out_dir / "eval" / ckpt_label,
                label=eval_label,
                checkpoint=checkpoint,
                data_jsonl=data_jsonl,
                loop_counts=loop_counts,
                value_prefix=value_prefix,
                dtype=dtype,
                keep_rows=keep_rows,
            )
            payload["evals"][ckpt_label][variant] = result
            if (
                "paired_relay_d1_12" in payload["evals"][ckpt_label]
                and "paired_pointer_d1_12" in payload["evals"][ckpt_label]
            ):
                payload["evals"][ckpt_label]["paired_relay_pointer_mcnemar"] = paired_mcnemar(
                    payload["evals"][ckpt_label]["paired_relay_d1_12"],
                    payload["evals"][ckpt_label]["paired_pointer_d1_12"],
                )
            payload["status"] = f"evaluated_{ckpt_label}_{variant}"
            publish(
                out_dir,
                payload,
                message=f"Record natural-surface receipt eval {run_id} {ckpt_label} {variant} [skip ci]",
            )

    if env_flag("STAGE5_NATURAL_RECEIPTS_RUN_SAME_READER", "1"):
        same_reader_specs = [
            ("relay_original", source_summary["data_paths"]["relay_test_chain_mcq"]),
            ("pointer_original", source_summary["data_paths"]["pointer_test_chain_mcq"]),
        ]
        payload["same_reader"] = {}
        for ckpt_label in [label for label in checkpoint_labels if label != "frozen_n24"]:
            payload["same_reader"].setdefault(ckpt_label, {})
            for family_label, data_jsonl in same_reader_specs:
                result = run_same_reader_eval(
                    out_dir / "same_reader" / ckpt_label,
                    label=f"{ckpt_label}_{family_label}",
                    checkpoint=ckpts[ckpt_label],
                    data_jsonl=data_jsonl,
                    max_loops=12,
                    value_prefix="name:",
                    dtype=dtype,
                )
                payload["same_reader"][ckpt_label][family_label] = result
                payload["status"] = f"same_reader_{ckpt_label}_{family_label}"
                publish(
                    out_dir,
                    payload,
                    message=f"Record natural-surface same-reader receipt {run_id} {ckpt_label} {family_label} [skip ci]",
                )

    payload["status"] = "finished"
    publish(out_dir, payload, message=f"Record natural-surface receipts final {run_id} [skip ci]")
    print(json.dumps({"run_id": run_id, "summary": path_for_cli(out_dir / "summary.json")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
