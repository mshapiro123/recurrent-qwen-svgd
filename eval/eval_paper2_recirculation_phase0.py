"""Run the governed Phase-0 gates for the paper-native recirculation probe."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import datasets
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_paper2_phase3_p34_task_trajectory import score_generation
from eval.eval_paper2_stage2bs_depth_study import _generation_rows
from models.recirculation import (
    PaperNativeRecirculationEvaluator,
    RecirculationConfig,
    graph_receipt,
)


KIND = "paper2_recirculation_phase0_v1"
LOCK_KIND = "paper2_recirculation_phase0_lock_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def file_receipt(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def adjudicate_battery_anchor(
    *,
    lock: Mapping[str, Any],
    v1_path: Path,
    rows_path: Path,
    v2_path: Path,
) -> dict[str, Any]:
    """Bind the retained alpha-zero rows to the comparator ruling without replay."""
    adjudication = lock["comparator_adjudication"]
    rows_receipt = file_receipt(rows_path)
    if rows_receipt != dict(adjudication["row_receipt"]):
        raise RuntimeError("ratified battery row receipt identity changed")
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    if v1.get("row_receipt") != rows_receipt:
        raise RuntimeError("v1 battery receipt no longer owns the retained rows")
    rows = read_jsonl(rows_path)
    observed_correct = sum(bool(row["augmented_correct"]) for row in rows)
    expected_rows = int(lock["gates"]["battery_anchor_rows"])
    expected_correct = int(adjudication["paper_native_correct"])
    if len(rows) != expected_rows or observed_correct != expected_correct:
        raise RuntimeError(
            "ratified paper-native comparator changed: "
            f"rows={len(rows)} correct={observed_correct}"
        )
    authority = next(
        (
            item
            for item in lock["authorities"]
            if item["filename"] == adjudication["authority_filename"]
        ),
        None,
    )
    if authority is None or authority["sha256"] != adjudication["authority_sha256"]:
        raise RuntimeError("comparator adjudication authority is missing or changed")
    receipt = {
        "kind": "paper2_recirculation_battery_anchor_v2_adjudicated",
        "status": "passed_by_strategy_adjudication",
        "rows": len(rows),
        "correct": observed_correct,
        "comparator": expected_correct,
        "row_receipt": rows_receipt,
        "source_v1_receipt": file_receipt(v1_path),
        "source_v1_passed": bool(v1.get("passed", False)),
        "source_v1_expected_correct": int(v1["expected_correct"]),
        "generation_replayed": False,
        "elapsed_seconds": float(v1["elapsed_seconds"]),
        "registered_bars": {
            "additive_delta": int(lock["gates"]["battery_additive_delta"]),
            "additive_threshold": int(lock["gates"]["battery_additive_threshold"]),
            "neutral_lower_delta": int(lock["gates"]["battery_neutral_lower_delta"]),
            "neutral_lower_threshold": int(
                lock["gates"]["battery_neutral_lower_threshold"]
            ),
        },
        "prior_comparator": {
            "correct": int(adjudication["prior_correct"]),
            "evaluator": adjudication["prior_evaluator"],
        },
        "authority": dict(authority),
        "passed": True,
    }
    if v2_path.is_file():
        if json.loads(v2_path.read_text(encoding="utf-8")) != receipt:
            raise RuntimeError("durable v2 battery adjudication receipt changed")
    else:
        write_json(v2_path, receipt)
    return receipt


def _string_leaves(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _string_leaves(child)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for child in value:
            yield from _string_leaves(child)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(words: Sequence[str], width: int) -> set[tuple[str, ...]]:
    return {tuple(words[index : index + width]) for index in range(len(words) - width + 1)}


def _battery_ngrams(panel: Sequence[Mapping[str, Any]], width: int) -> set[tuple[str, ...]]:
    grams: set[tuple[str, ...]] = set()
    for row in _generation_rows(panel):
        grams.update(_ngrams(_words(" ".join(_string_leaves(row))), width))
    return grams


def _document_windows(tokenizer: Any, text: str, *, width: int, maximum: int) -> list[list[int]]:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [
        token_ids[start : start + width]
        for start in range(0, min(len(token_ids), width * maximum), width)
        if len(token_ids[start : start + width]) == width
    ]


def prepare_corpus(
    *,
    lock: Mapping[str, Any],
    panel: Sequence[Mapping[str, Any]],
    qwen_tokenizer: Any,
    gemma_tokenizer: Any,
    private_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    token_path = private_dir / "corpus_token_windows.pt"
    receipt_path = private_dir / "corpus_receipt.json"
    if token_path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        observed = {"bytes": token_path.stat().st_size, "sha256": sha256_file(token_path)}
        if receipt.get("token_windows") != observed:
            raise RuntimeError("durable corpus token cache identity changed")
        saved = torch.load(token_path, map_location="cpu", weights_only=False)
        return saved["qwen"], saved["gemma"], receipt
    corpus = lock["corpus"]
    width = int(corpus["window_tokens"])
    maximum = int(corpus["maximum_windows_per_document"])
    qwen_target = int(corpus["qwen_timing_windows"])
    gemma_target = int(corpus["gemma_anchor_windows"])
    overlap_width = int(corpus["battery_overlap_ngram"])
    battery_grams = _battery_ngrams(panel, overlap_width)
    stream = load_dataset(
        corpus["id"],
        split=corpus["split"],
        revision=corpus["revision"],
        streaming=True,
    )
    qwen_windows: list[list[int]] = []
    gemma_windows: list[list[int]] = []
    manifest: list[dict[str, Any]] = []
    dropped_overlap = 0
    visited = 0
    for source_index, row in enumerate(stream):
        visited += 1
        text = str(row[corpus["text_field"]])
        words = _words(text)
        if battery_grams.intersection(_ngrams(words, overlap_width)):
            dropped_overlap += 1
            continue
        qwen_add = _document_windows(
            qwen_tokenizer, text, width=width, maximum=maximum
        )
        gemma_add = _document_windows(
            gemma_tokenizer, text, width=width, maximum=maximum
        )
        if not qwen_add and not gemma_add:
            continue
        qwen_take = qwen_add[: max(0, qwen_target - len(qwen_windows))]
        gemma_take = gemma_add[: max(0, gemma_target - len(gemma_windows))]
        if not qwen_take and not gemma_take:
            break
        qwen_windows.extend(qwen_take)
        gemma_windows.extend(gemma_take)
        manifest.append(
            {
                "source_index": source_index,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text_bytes": len(text.encode("utf-8")),
                "qwen_windows": len(qwen_take),
                "gemma_windows": len(gemma_take),
            }
        )
        if len(qwen_windows) >= qwen_target and len(gemma_windows) >= gemma_target:
            break
        if visited % 25 == 0:
            print(
                f"recirculation_corpus_progress documents={visited} "
                f"qwen_windows={len(qwen_windows)}/{qwen_target} "
                f"gemma_windows={len(gemma_windows)}/{gemma_target}",
                flush=True,
            )
    if len(qwen_windows) != qwen_target or len(gemma_windows) != gemma_target:
        raise RuntimeError(
            "fixed arXiv stream did not supply the registered complete windows: "
            f"qwen={len(qwen_windows)} gemma={len(gemma_windows)}"
        )
    manifest_path = private_dir / "corpus_manifest.jsonl"
    write_jsonl(manifest_path, manifest)
    torch.save(
        {
            "qwen": torch.tensor(qwen_windows, dtype=torch.long),
            "gemma": torch.tensor(gemma_windows, dtype=torch.long),
        },
        token_path,
    )
    receipt = {
        "kind": "paper2_recirculation_corpus_v1",
        "dataset": corpus["id"],
        "revision": corpus["revision"],
        "split": corpus["split"],
        "visited_documents": visited,
        "admitted_documents": len(manifest),
        "dropped_battery_13gram_overlap": dropped_overlap,
        "window_tokens": width,
        "maximum_windows_per_document": maximum,
        "qwen_windows": len(qwen_windows),
        "gemma_windows": len(gemma_windows),
        "manifest": {
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "token_windows": {
            "bytes": token_path.stat().st_size,
            "sha256": sha256_file(token_path),
        },
    }
    write_json(receipt_path, receipt)
    return (
        torch.tensor(qwen_windows, dtype=torch.long),
        torch.tensor(gemma_windows, dtype=torch.long),
        receipt,
    )


def load_model(spec: Mapping[str, Any], *, cache_dir: Path) -> Any:
    return AutoModelForCausalLM.from_pretrained(
        spec["id"],
        revision=spec["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        cache_dir=cache_dir,
    ).to("cuda").eval()


def score_nll(
    evaluator: PaperNativeRecirculationEvaluator,
    windows: torch.Tensor,
    *,
    recirculate: bool,
    batch_size: int,
    label: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    loss_sum = 0.0
    count = 0
    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start : start + batch_size].to(evaluator.device)
        mask = torch.ones_like(batch)
        batch_loss, batch_count = evaluator.sequence_nll(
            input_ids=batch,
            attention_mask=mask,
            recirculate=recirculate,
        )
        loss_sum += batch_loss
        count += batch_count
        print(
            f"recirculation_nll_progress label={label} "
            f"windows={min(start + batch.shape[0], windows.shape[0])}/{windows.shape[0]}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    mean_nll = loss_sum / count
    return {
        "label": label,
        "windows": int(windows.shape[0]),
        "window_tokens": int(windows.shape[1]),
        "predicted_tokens": count,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "elapsed_seconds": elapsed,
    }


def runtime_receipt() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_mib": int(properties.total_memory // (1024 * 1024)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "dtype": "bfloat16",
        "attention_backend": "sdpa",
    }


def validate_runtime(lock: Mapping[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 0 requires the pinned A100 runtime")
    receipt = runtime_receipt()
    expected = lock["runtime"]
    if receipt["gpu"] != expected["accelerator"]:
        raise RuntimeError(
            f"runtime identity changed: expected {expected['accelerator']}, observed {receipt['gpu']}"
        )
    if not str(receipt["torch"]).startswith(expected["torch_prefix"]):
        raise RuntimeError(
            f"torch runtime changed: expected {expected['torch_prefix']}, observed {receipt['torch']}"
        )
    if str(receipt["cuda"]) != "12.8":
        raise RuntimeError(f"CUDA runtime changed: expected 12.8, observed {receipt['cuda']}")
    for package in ("transformers", "datasets"):
        if receipt[package] != expected[package]:
            raise RuntimeError(
                f"{package} runtime changed: expected {expected[package]}, "
                f"observed {receipt[package]}"
            )
    return receipt


def projection_receipt(
    *, phase0_elapsed: float, qwen_pilot: Mapping[str, Any], battery_elapsed: float
) -> dict[str, Any]:
    num_layers = 24
    pilot_destination = 8
    pilot_equivalents = 1.0 + (num_layers - pilot_destination) / num_layers
    destinations = (2, 4, 6, 8, 10, 12, 14)
    offsets = (4, 6, 8, 10, 12)
    pairs = [
        (destination, destination + offset)
        for destination in destinations
        for offset in offsets
        if destination + offset <= 22
    ]
    seconds_per_pilot = float(qwen_pilot["recirculated"]["elapsed_seconds"])
    seconds_per_pair = {
        destination: seconds_per_pilot
        * (1.0 + (num_layers - destination) / num_layers)
        / pilot_equivalents
        for destination in destinations
    }
    a2_seconds = sum(seconds_per_pair[destination] * 3 for destination, _ in pairs)
    worst_refinement_seconds = max(seconds_per_pair.values())
    a3_perplexity_seconds = worst_refinement_seconds * (8 + 3 + 1 + 1)
    battery_seconds = 2.0 * battery_elapsed * max(seconds_per_pair.values()) / seconds_per_pilot
    total_seconds = phase0_elapsed + a2_seconds + a3_perplexity_seconds + battery_seconds
    return {
        "kind": "paper2_recirculation_cost_projection_v1",
        "coarse_pairs": len(pairs),
        "coarse_cells": len(pairs) * 3,
        "refinement_perplexity_cells": 13,
        "battery_cells": 2,
        "phase0_elapsed_seconds": phase0_elapsed,
        "projected_a2_seconds": a2_seconds,
        "projected_a3_perplexity_seconds": a3_perplexity_seconds,
        "projected_a3_battery_seconds": battery_seconds,
        "projected_total_seconds": total_seconds,
        "projected_total_a100_hours": total_seconds / 3600.0,
        "ceiling_a100_hours": 8.0,
        "within_ceiling": total_seconds / 3600.0 <= 8.0,
        "assumption": "destination_cost_scales_with_1_plus_L_minus_d_over_L_and_A3_uses_worst_destination",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--private_dir", type=Path, required=True)
    parser.add_argument("--model_cache", type=Path, required=True)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--nll_batch_size", type=int, default=32)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("kind") != LOCK_KIND or lock.get("phase_b_training_authorized") is not False:
        raise RuntimeError("recirculation Phase-0 lock is invalid or over-authorized")
    if lock.get("optimizer_steps_allowed") != 0:
        raise RuntimeError("Phase 0 must prohibit optimizer steps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runtime = validate_runtime(lock)
    write_json(args.output_dir / "runtime.json", runtime)
    panel = read_jsonl(args.panel)
    if len(panel) != 1024:
        raise RuntimeError("recirculation battery requires the frozen 1,024-row DEV panel")

    qwen_spec = lock["models"]["qwen"]
    gemma_spec = lock["models"]["gemma_anchor"]
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        qwen_spec["id"], revision=qwen_spec["revision"], cache_dir=args.model_cache
    )
    gemma_tokenizer = AutoTokenizer.from_pretrained(
        gemma_spec["id"], revision=gemma_spec["revision"], cache_dir=args.model_cache
    )
    qwen_windows, gemma_windows, corpus = prepare_corpus(
        lock=lock,
        panel=panel,
        qwen_tokenizer=qwen_tokenizer,
        gemma_tokenizer=gemma_tokenizer,
        private_dir=args.private_dir,
    )

    phase0: dict[str, Any] = {
        "kind": KIND,
        "status": "running",
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
        "phase_a_authorized": False,
        "runtime": runtime,
        "corpus": corpus,
    }
    write_json(args.output_dir / "phase0_status.json", phase0)

    qwen_model = load_model(qwen_spec, cache_dir=args.model_cache)
    identity_tokens = qwen_windows[:1, :32].to("cuda")
    identity = PaperNativeRecirculationEvaluator(
        qwen_model,
        RecirculationConfig(
            source_layer=int(qwen_spec["source_layer"]),
            destination_layer=int(qwen_spec["destination_layer"]),
            alpha=0.0,
        ),
    ).identity_receipt(
        input_ids=identity_tokens,
        attention_mask=torch.ones_like(identity_tokens),
    )
    if not identity["bit_exact"]:
        raise RuntimeError(f"Qwen recirculation identity gate failed: {identity}")
    write_json(args.output_dir / "identity_qwen.json", identity)
    graph = graph_receipt(
        sequence_length=32,
        num_layers=len(qwen_model.model.layers),
        destination_layer=int(qwen_spec["destination_layer"]),
    )
    write_json(args.output_dir / "graph_receipt.json", graph)

    battery_path = args.output_dir / "battery_anchor.json"
    battery_v2_path = args.output_dir / "battery_anchor_v2_adjudicated.json"
    battery_rows_path = args.private_dir / "battery_anchor_rows.jsonl"
    battery_resumed = battery_path.is_file() and battery_rows_path.is_file()
    if not battery_resumed and not bool(
        lock["comparator_adjudication"]["regenerate_anchor"]
    ):
        raise RuntimeError(
            "ratified comparator rows are missing; regeneration is prohibited"
        )
    if battery_resumed:
        battery_v1 = json.loads(battery_path.read_text(encoding="utf-8"))
        if battery_v1.get("row_receipt") != {
            "bytes": battery_rows_path.stat().st_size,
            "sha256": sha256_file(battery_rows_path),
        }:
            raise RuntimeError("durable battery row receipt identity changed")
        battery_elapsed = float(battery_v1["elapsed_seconds"])
        battery_graph = None
        print("recirculation_resume stage=battery_anchor", flush=True)
    else:
        battery_graph = PaperNativeRecirculationEvaluator(
            qwen_model,
            RecirculationConfig(
                source_layer=int(qwen_spec["source_layer"]),
                destination_layer=int(qwen_spec["destination_layer"]),
                alpha=0.0,
            ),
        )
        battery_started = time.perf_counter()
        battery_rows = score_generation(
            battery_graph,
            qwen_tokenizer,
            _generation_rows(panel),
            batch_size=args.generation_batch_size,
        )
        battery_elapsed = time.perf_counter() - battery_started
        write_jsonl(battery_rows_path, battery_rows)
        battery_v1 = {
            "kind": "paper2_recirculation_battery_anchor_v1",
            "rows": len(battery_rows),
            "correct": sum(bool(row["augmented_correct"]) for row in battery_rows),
            "expected_correct": int(lock["gates"]["battery_anchor_correct"]),
            "elapsed_seconds": battery_elapsed,
            "row_receipt": {
                "bytes": battery_rows_path.stat().st_size,
                "sha256": sha256_file(battery_rows_path),
            },
        }
        battery_v1["passed"] = (
            battery_v1["rows"] == int(lock["gates"]["battery_anchor_rows"])
            and battery_v1["correct"] == battery_v1["expected_correct"]
        )
        write_json(battery_path, battery_v1)
    battery = adjudicate_battery_anchor(
        lock=lock,
        v1_path=battery_path,
        rows_path=battery_rows_path,
        v2_path=battery_v2_path,
    )
    if not battery["passed"]:
        raise RuntimeError(f"paper-native battery comparator changed: {battery}")

    qwen_timing_path = args.output_dir / "qwen_timing.json"
    qwen_timing_resumed = qwen_timing_path.is_file()
    if qwen_timing_resumed:
        qwen_timing = json.loads(qwen_timing_path.read_text(encoding="utf-8"))
        qwen_probe = None
        print("recirculation_resume stage=qwen_timing", flush=True)
    else:
        qwen_probe = PaperNativeRecirculationEvaluator(
            qwen_model,
            RecirculationConfig(
                source_layer=int(qwen_spec["source_layer"]),
                destination_layer=int(qwen_spec["destination_layer"]),
                alpha=float(qwen_spec["alpha"]),
            ),
        )
        qwen_baseline = score_nll(
            qwen_probe,
            qwen_windows,
            recirculate=False,
            batch_size=args.nll_batch_size,
            label="qwen_intact_32k",
        )
        qwen_recirculated = score_nll(
            qwen_probe,
            qwen_windows,
            recirculate=True,
            batch_size=args.nll_batch_size,
            label="qwen_16_8_0p10_32k",
        )
        qwen_timing = {
            "kind": "paper2_recirculation_qwen_timing_v1",
            "cell": {"source": 16, "destination": 8, "alpha": 0.10},
            "intact": qwen_baseline,
            "recirculated": qwen_recirculated,
            "perplexity_reduction_percent": 100.0
            * (qwen_baseline["perplexity"] - qwen_recirculated["perplexity"])
            / qwen_baseline["perplexity"],
        }
        write_json(qwen_timing_path, qwen_timing)
    del qwen_probe, battery_graph, qwen_model
    gc.collect()
    torch.cuda.empty_cache()

    gemma_model = load_model(gemma_spec, cache_dir=args.model_cache)
    gemma_identity_tokens = gemma_windows[:1, :32].to("cuda")
    gemma_identity = PaperNativeRecirculationEvaluator(
        gemma_model,
        RecirculationConfig(source_layer=11, destination_layer=4, alpha=0.0),
    ).identity_receipt(
        input_ids=gemma_identity_tokens,
        attention_mask=torch.ones_like(gemma_identity_tokens),
    )
    if not gemma_identity["bit_exact"]:
        raise RuntimeError(f"Gemma recirculation identity gate failed: {gemma_identity}")
    write_json(args.output_dir / "identity_gemma.json", gemma_identity)
    gemma_anchor_path = args.output_dir / "gemma_anchor.json"
    gemma_anchor_resumed = gemma_anchor_path.is_file()
    if gemma_anchor_resumed:
        gemma_anchor = json.loads(gemma_anchor_path.read_text(encoding="utf-8"))
        gemma_probe = None
        print("recirculation_resume stage=gemma_anchor", flush=True)
    else:
        gemma_probe = PaperNativeRecirculationEvaluator(
            gemma_model,
            RecirculationConfig(source_layer=11, destination_layer=4, alpha=0.15),
        )
        gemma_baseline = score_nll(
            gemma_probe,
            gemma_windows,
            recirculate=False,
            batch_size=args.nll_batch_size,
            label="gemma_intact_128k",
        )
        gemma_recirculated = score_nll(
            gemma_probe,
            gemma_windows,
            recirculate=True,
            batch_size=args.nll_batch_size,
            label="gemma_11_4_0p15_128k",
        )
        gemma_anchor = {
            "kind": "paper2_recirculation_gemma_anchor_v1",
            "cell": {"source": 11, "destination": 4, "alpha": 0.15},
            "intact": gemma_baseline,
            "recirculated": gemma_recirculated,
            "perplexity_reduction_percent": 100.0
            * (gemma_baseline["perplexity"] - gemma_recirculated["perplexity"])
            / gemma_baseline["perplexity"],
        }
        gemma_anchor["passed_directional_gate"] = (
            gemma_recirculated["perplexity"] < gemma_baseline["perplexity"]
        )
        write_json(gemma_anchor_path, gemma_anchor)
    if not gemma_anchor["passed_directional_gate"]:
        raise RuntimeError(f"published Gemma directional anchor failed: {gemma_anchor}")
    del gemma_probe, gemma_model
    gc.collect()
    torch.cuda.empty_cache()

    resumed_elapsed = 0.0
    if battery_resumed:
        resumed_elapsed += float(battery["elapsed_seconds"])
    if qwen_timing_resumed:
        resumed_elapsed += float(qwen_timing["intact"]["elapsed_seconds"])
        resumed_elapsed += float(qwen_timing["recirculated"]["elapsed_seconds"])
    if gemma_anchor_resumed:
        resumed_elapsed += float(gemma_anchor["intact"]["elapsed_seconds"])
        resumed_elapsed += float(gemma_anchor["recirculated"]["elapsed_seconds"])
    phase0_elapsed = time.perf_counter() - started + resumed_elapsed
    projection = projection_receipt(
        phase0_elapsed=phase0_elapsed,
        qwen_pilot=qwen_timing,
        battery_elapsed=battery_elapsed,
    )
    write_json(args.output_dir / "cost_projection.json", projection)
    phase0.update(
        status=("phase0_pass_awaiting_relay" if projection["within_ceiling"] else "cost_ceiling_stop"),
        elapsed_seconds=phase0_elapsed,
        graph_receipt_sha256=sha256_file(args.output_dir / "graph_receipt.json"),
        identity_qwen=identity,
        identity_gemma=gemma_identity,
        battery_anchor_v1=battery_v1,
        battery_anchor=battery,
        qwen_timing=qwen_timing,
        gemma_anchor=gemma_anchor,
        cost_projection=projection,
    )
    write_json(args.output_dir / "phase0_status.json", phase0)
    print(json.dumps(phase0, indent=2, sort_keys=True), flush=True)
    # A fired cost ceiling is a successful governed stop, not an execution error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
