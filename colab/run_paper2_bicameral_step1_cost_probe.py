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

from models.bicameral import BicameralTaskInferenceGraph
from models.recurrent_wrapper import LayerSplit, RecurrentQwenForCausalLM


KIND = "paper2_bicameral_step1_cost_probe_v1"
PINNED_GPU = "NVIDIA A100-SXM4-40GB"
PINNED_TORCH_PREFIX = "2.11.0"
PINNED_CUDA_PREFIX = "12.8"


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
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
    }
    atomic_json(args.output, receipt)

    load_started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
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
        "training_row_count_unbound": True,
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
    receipt["hard_gate_pass"] = receipt["t1_real_substrate"]["exact_equal"]
    atomic_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["hard_gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
