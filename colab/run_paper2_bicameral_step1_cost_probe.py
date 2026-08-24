"""Pinned-runtime, forward-only cost probe for Bicameral Step-1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM

from models.bicameral import (
    SEQUENTIAL_EXECUTION_SCHEDULE,
    BicameralTaskInferenceGraph,
)
from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM


KIND = "paper2_bicameral_w0_preflight_v1"
PINNED_GPU = "NVIDIA A100-SXM4-40GB"
PINNED_TORCH_PREFIX = "2.11.0"
PINNED_CUDA_PREFIX = "12.8"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def runtime_receipt() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Pinned Bicameral cost probe requires CUDA")
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 2**30
    receipt = {
        "gpu_name": properties.name,
        "gpu_total_gib": total_gib,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
    }
    if properties.name != PINNED_GPU or not 38.0 <= total_gib <= 42.0:
        raise RuntimeError(f"Runtime mismatch: {receipt}")
    if not torch.__version__.startswith(PINNED_TORCH_PREFIX):
        raise RuntimeError(f"Torch mismatch: {receipt}")
    if not str(torch.version.cuda).startswith(PINNED_CUDA_PREFIX):
        raise RuntimeError(f"CUDA mismatch: {receipt}")
    return receipt


def benchmark_cuda(
    operation: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.fmean(timings),
        "minimum_ms": min(timings),
        "maximum_ms": max(timings),
        "repeats": repeats,
    }


def make_tokens(batch: int, length: int, vocab_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260823 + batch * 10000 + length)
    input_ids = torch.randint(32, vocab_size, (batch, length), generator=generator)
    return input_ids.to(device), torch.ones((batch, length), dtype=torch.long, device=device)


def load_probe_batch(path: Path, device: torch.device, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows or any(not row.get("input_ids") for row in rows):
        raise RuntimeError("W0 probe batch is empty or malformed")
    width = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        input_ids[index, : len(values)] = values
        attention_mask[index, : len(values)] = 1
    return input_ids.to(device), attention_mask.to(device), [str(row["item_id"]) for row in rows]


def branch_divergence(branch_a: torch.Tensor, branch_b: torch.Tensor) -> dict[str, float]:
    left = branch_a.float().reshape(-1)
    right = branch_b.float().reshape(-1)
    difference = left - right
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    return {
        "l2_norm": float(torch.linalg.vector_norm(difference).cpu()),
        "rms": float(difference.square().mean().sqrt().cpu()),
        "maximum_absolute_difference": float(difference.abs().max().cpu()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(left, right, dim=0).cpu()),
        "centered_correlation": float(
            (centered_left * centered_right).sum()
            / (
                torch.linalg.vector_norm(centered_left)
                * torch.linalg.vector_norm(centered_right)
            ).clamp_min(1e-12)
        ),
    }


def actual_t2_contract(graph: BicameralTaskInferenceGraph, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, Any]:
    graph.core.zero_gates()
    for parameter in graph.core.parameters():
        parameter.grad = None
    output = graph(input_ids=input_ids[:1], attention_mask=attention_mask[:1])
    output.logits[:, -1].float().square().mean().backward()
    alive_names = (
        "callosum.gate_a",
        "callosum.gate_b",
        "bank_a.gate",
        "bank_b.gate",
        "combiner.mu",
    )
    parameters = dict(graph.core.named_parameters())
    alive = {
        name: {
            "finite": bool(parameters[name].grad is not None and torch.isfinite(parameters[name].grad).all()),
            "nonzero": bool(parameters[name].grad is not None and torch.count_nonzero(parameters[name].grad)),
        }
        for name in alive_names
    }
    exactly_zero = {
        name: bool(parameters[name].grad is not None and torch.count_nonzero(parameters[name].grad) == 0)
        for name in ("combiner.delta", "bank_a.gains", "bank_b.gains")
    }
    for parameter in graph.core.parameters():
        parameter.grad = None
    passed = all(item["finite"] and item["nonzero"] for item in alive.values()) and all(exactly_zero.values())
    return {"pass": passed, "alive": alive, "exactly_zero": exactly_zero}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-summary", type=Path, required=True)
    parser.add_argument("--probe-batch", type=Path, required=True)
    parser.add_argument("--initializer-seed-0", type=Path, required=True)
    parser.add_argument("--initializer-seed-1", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    receipt: dict[str, Any] = {
        "kind": KIND,
        "status": "running",
        "training_performed": False,
        "optimizer_constructed": False,
        "sealed_partition_touched": False,
        "runtime": runtime_receipt(),
        "model_id": args.model,
        "model_revision": MODEL_REVISION,
        "execution_schedule": SEQUENTIAL_EXECUTION_SCHEDULE,
        "authority_sha256": args.authority_sha256,
        "inputs": {
            "manifest_sha256": sha256_file(args.manifest),
            "manifest_summary_sha256": sha256_file(args.manifest_summary),
            "probe_batch_sha256": sha256_file(args.probe_batch),
            "initializer_seed_0_sha256": sha256_file(args.initializer_seed_0),
            "initializer_seed_1_sha256": sha256_file(args.initializer_seed_1),
        },
    }
    manifest_summary = json.loads(args.manifest_summary.read_text(encoding="utf-8"))
    if manifest_summary.get("manifest", {}).get("sha256") != receipt["inputs"]["manifest_sha256"]:
        raise RuntimeError("W0 manifest byte hash does not match its locked summary")
    if manifest_summary.get("probe_batch", {}).get("sha256") != receipt["inputs"]["probe_batch_sha256"]:
        raise RuntimeError("W0 probe-batch byte hash does not match its locked summary")
    if manifest_summary.get("execution_schedule") != SEQUENTIAL_EXECUTION_SCHEDULE:
        raise RuntimeError("W0 manifest evaluator schedule changed")
    atomic_json(args.output, receipt)

    load_started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    wrapper = RecurrentQwenForCausalLM(
        base,
        layer_split=LayerSplit(prelude_end=6, recurrent_end=18),
    ).eval()
    graph = BicameralTaskInferenceGraph(wrapper).eval()
    graph.core.zero_gates()
    receipt["model_load_seconds"] = time.perf_counter() - load_started
    receipt["model_config_commit"] = getattr(base.config, "_commit_hash", None)
    receipt["bicameral_parameter_count"] = sum(p.numel() for p in graph.core.parameters())
    receipt["step1_trainable_parameter_count"] = sum(
        p.numel() for p in graph.core.combiner.parameters()
    )

    vocab_size = int(base.config.vocab_size)
    identity_ids, identity_mask = make_tokens(1, 64, vocab_size, graph.device)
    with torch.inference_mode():
        base_output = graph.base_path(input_ids=identity_ids, attention_mask=identity_mask)
        bicameral_output = graph(input_ids=identity_ids, attention_mask=identity_mask)
    difference = (base_output.logits.float() - bicameral_output.logits.float()).abs()
    receipt["t1_real_substrate"] = {
        "exact_equal": bool(torch.equal(base_output.logits, bicameral_output.logits)),
        "maximum_absolute_logit_difference": float(difference.max()),
        "nonzero_logit_differences": int(torch.count_nonzero(difference)),
        "compared_values": int(difference.numel()),
    }

    probe_ids, probe_mask, probe_item_ids = load_probe_batch(
        args.probe_batch, graph.device, int(base.config.pad_token_id or base.config.eos_token_id)
    )
    receipt["probe_batch"] = {
        "rows": len(probe_item_ids),
        "item_ids": probe_item_ids,
        "padded_width": int(probe_ids.shape[1]),
    }
    receipt["t2_real_substrate"] = actual_t2_contract(graph, probe_ids, probe_mask)

    divergences = {}
    for seed, initializer in ((0, args.initializer_seed_0), (1, args.initializer_seed_1)):
        graph.core.zero_gates()
        graph.core.load_branch_initializers(initializer)
        graph.core.bind_strategy_operating_gates(source_receipt_sha256=args.authority_sha256)
        with torch.inference_mode():
            states = graph.cache_branch_states(input_ids=probe_ids, attention_mask=probe_mask)
        divergences[str(seed)] = {
            **branch_divergence(states.branch_a, states.branch_b),
            "initializer_sha256": sha256_file(initializer),
            "conditioning_receipt_sha256": graph.core.conditioning_receipt_sha256,
        }
    receipt["operating_point_divergence"] = divergences

    timings: list[dict[str, Any]] = []
    for batch, length in ((1, 64), (1, 128), (1, 256), (1, 512), (8, 256)):
        input_ids, attention_mask = make_tokens(batch, length, vocab_size, graph.device)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            measured = benchmark_cuda(
                lambda: graph.cache_branch_states(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            )
        measured.update(
            {
                "batch_size": batch,
                "sequence_length": length,
                "milliseconds_per_row": measured["median_ms"] / batch,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
        )
        timings.append(measured)
    receipt["branch_cache_timings"] = timings

    manifest_rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    manifest_lengths = sorted(int(row["native_input_token_length"]) for row in manifest_rows)
    torch.cuda.reset_peak_memory_stats()
    manifest_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(manifest_lengths), 8):
            local = manifest_lengths[start : start + 8]
            ids, mask = make_tokens(len(local), max(local), vocab_size, graph.device)
            graph.cache_branch_states(input_ids=ids, attention_mask=mask)
    torch.cuda.synchronize()
    manifest_seconds = time.perf_counter() - manifest_started
    receipt["bound_manifest_cache_measurement"] = {
        "rows": len(manifest_rows),
        "batch_size": 8,
        "minimum_sequence_length": min(manifest_lengths),
        "maximum_sequence_length": max(manifest_lengths),
        "mean_sequence_length": statistics.fmean(manifest_lengths),
        "one_seed_seconds": manifest_seconds,
        "two_seed_gpu_hours": 2 * manifest_seconds / 3600,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "manifest_sha256": sha256_file(args.manifest),
    }

    timing_256 = next(
        item for item in timings if item["batch_size"] == 8 and item["sequence_length"] == 256
    )
    per_row_seconds = timing_256["milliseconds_per_row"] / 1000.0
    receipt["planning_projection_at_batch8_seq256"] = {
        "seconds_per_row": per_row_seconds,
        "minutes_per_461_row_slice": per_row_seconds * 461 / 60,
        "minutes_per_2048_row_panel": per_row_seconds * 2048 / 60,
        "minutes_per_1000_training_rows": per_row_seconds * 1000 / 60,
        "two_seed_formula_gpu_hours": "2 * seconds_per_row * (training_rows + 461 + 2048) / 3600",
        "training_row_count": 256,
        "projected_two_seed_cache_gpu_hours_for_256_plus_461_plus_2048_rows": (
            2 * per_row_seconds * (256 + 461 + 2048) / 3600
        ),
    }

    torch.manual_seed(20260823)
    branch_a = torch.randn((256, 1, 896), device="cuda", dtype=torch.bfloat16)
    branch_b = torch.randn_like(branch_a)
    target = torch.randn_like(branch_a)
    torch.cuda.synchronize()
    fit_start = time.perf_counter()
    graph.core.combiner.fit_state_matching(branch_a, branch_b, target)
    torch.cuda.synchronize()
    receipt["closed_form_fit_256_rows_ms"] = (time.perf_counter() - fit_start) * 1000

    receipt["status"] = "complete"
    receipt["hard_gate_pass"] = bool(
        receipt["t1_real_substrate"]["exact_equal"]
        and receipt["t2_real_substrate"]["pass"]
        and all(item["l2_norm"] > 1e-8 for item in divergences.values())
        and receipt["execution_schedule"] == SEQUENTIAL_EXECUTION_SCHEDULE
    )
    atomic_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["hard_gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
