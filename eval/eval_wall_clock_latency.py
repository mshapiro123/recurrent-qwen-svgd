"""Measure batch-1 wall-clock latency on the registered Phase A evaluation paths.

The recurrent prefill/decode split is a subtraction decomposition: a synchronized
one-loop reference call estimates prompt processing, while the actual forced-depth
call supplies total model latency.  The difference is reported as decode-side
recurrent work.  Dense arms use synchronized cached greedy decoding, with the
first forward counted as prefill and later cached forwards counted as decode.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SELECTED_DEPTHS = (1, 2, 4, 8, 11, 14)
STABILITY_DEPTHS = (4, 8, 12)
ARM_ORDER = ("A", "E", "C", "B", "D")
ARM_LABELS = {
    "A": "Arm A full-block recurrent",
    "E": "Arm E R16 recurrent",
    "C": "Arm C dense scratchpad",
    "B": "Arm B dense direct 0.5B",
    "D": "Arm D dense direct 1.5B",
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def row_id(row: dict[str, Any]) -> str:
    value = row.get("id") or row.get("instance_id")
    if value is None:
        raise KeyError("Latency row is missing id/instance_id")
    return str(value)


def build_observation_schedule(
    rows: list[dict[str, Any]],
    *,
    stability_depths: Iterable[int] = STABILITY_DEPTHS,
    stability_rows: int = 32,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_depth[int(row["depth"])].append(row)
    depths = sorted(by_depth)
    full: list[dict[str, Any]] = []
    for index in range(max(len(values) for values in by_depth.values())):
        for depth in depths:
            if index < len(by_depth[depth]):
                full.append({"phase": "full", "repeat": 0, "row": by_depth[depth][index]})
    stability: list[dict[str, Any]] = []
    for repeat in range(1, int(repeats) + 1):
        for depth in stability_depths:
            selected = by_depth[int(depth)][: int(stability_rows)]
            if len(selected) != int(stability_rows):
                raise ValueError(f"Depth {depth} has {len(selected)} rows; stability requires {stability_rows}")
            stability.extend({"phase": "stability", "repeat": repeat, "row": row} for row in selected)
    return full + stability


def observation_key(item: dict[str, Any]) -> str:
    return f"{item['phase']}|{int(item['repeat'])}|{row_id(item['row'])}"


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * float(fraction)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(value) for value in values]
    q1 = _percentile(data, 0.25)
    q3 = _percentile(data, 0.75)
    return {
        "n": len(data),
        "median": statistics.median(data) if data else 0.0,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": min(data) if data else 0.0,
        "max": max(data) if data else 0.0,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("total_ms", "prefill_ms", "decode_ms", "model_total_ms", "tokenization_ms", "generated_tokens")
    arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        arm_rows = [row for row in records if row.get("arm") == arm]
        if not arm_rows:
            continue
        full = [row for row in arm_rows if row.get("phase") == "full"]
        by_depth: dict[str, Any] = {}
        for depth in sorted({int(row["depth"]) for row in full}):
            selected = [row for row in full if int(row["depth"]) == depth]
            by_depth[str(depth)] = {
                **{metric: _stats(row[metric] for row in selected) for metric in metrics},
                "accuracy": sum(int(bool(row.get("correct"))) for row in selected) / len(selected),
            }
        stability: dict[str, Any] = {}
        stable_rows = [row for row in arm_rows if row.get("phase") == "stability"]
        for depth in sorted({int(row["depth"]) for row in stable_rows}):
            repeat_medians: dict[str, Any] = {}
            for repeat in sorted({int(row["repeat"]) for row in stable_rows if int(row["depth"]) == depth}):
                selected = [
                    row for row in stable_rows
                    if int(row["depth"]) == depth and int(row["repeat"]) == repeat
                ]
                repeat_medians[str(repeat)] = {
                    metric: statistics.median(float(row[metric]) for row in selected) for metric in metrics
                }
            decode_medians = [float(item["decode_ms"]) for item in repeat_medians.values()]
            stability[str(depth)] = {
                "repeat_medians": repeat_medians,
                "decode_median_spread_ms": max(decode_medians) - min(decode_medians) if decode_medians else 0.0,
            }
        arms[arm] = {
            "label": ARM_LABELS[arm],
            "n_full": len(full),
            "n_stability": len(stable_rows),
            "overall": {metric: _stats(row[metric] for row in full) for metric in metrics},
            "accuracy": sum(int(bool(row.get("correct"))) for row in full) / len(full),
            "by_depth": by_depth,
            "stability": stability,
        }
    return {
        "kind": "stage5_wall_clock_latency_descriptive",
        "status": "finished" if len(arms) == len(ARM_ORDER) else "partial",
        "arm_order": list(ARM_ORDER),
        "arms": arms,
        "scope": "single hardware configuration, batch size 1, registered evaluation paths",
    }


def build_markdown_table(summary: dict[str, Any], *, selected_depths: Iterable[int] = SELECTED_DEPTHS) -> str:
    arms = summary["arms"]
    present = [arm for arm in ARM_ORDER if arm in arms]
    headers = [f"Arm {arm}" for arm in present]
    lines = [
        "# Wall-Clock Latency Receipt",
        "",
        "Decode-side median milliseconds per row (batch size 1):",
        "",
        "| Depth | " + " | ".join(headers) + " |",
        "|---:|" + "---:|" * len(headers),
    ]
    for depth in selected_depths:
        cells = []
        for arm in present:
            value = arms[arm]["by_depth"][str(int(depth))]["decode_ms"]["median"]
            cells.append(f"{float(value):.2f}")
        lines.append(f"| {int(depth)} | " + " | ".join(cells) + " |")
    token_cells = [f"{float(arms[arm]['overall']['generated_tokens']['median']):.2f}" for arm in present]
    lines.append("| Median generated tokens | " + " | ".join(token_cells) + " |")
    lines.extend(
        [
            "",
            summary.get(
                "conditions_sentence",
                "Measured on one GPU configuration with bfloat16 weights, batch size 1, greedy decoding, and each arm's registered evaluation path.",
            ),
            "",
            "Recurrent decode-side time is the synchronized forced-depth call minus a synchronized one-loop reference; dense decode-side time is cached greedy decoding after the first prompt forward. This is an interactive-latency result, not a batched-throughput result.",
            "",
            "## Full Per-Depth Table",
            "",
            "| Arm | Depth | Total median (ms) | Total IQR | Prefill median | Decode median | Decode IQR | Tokens median |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in present:
        for depth, values in sorted(arms[arm]["by_depth"].items(), key=lambda item: int(item[0])):
            lines.append(
                f"| {arm} | {depth} | {values['total_ms']['median']:.2f} | {values['total_ms']['iqr']:.2f} | "
                f"{values['prefill_ms']['median']:.2f} | {values['decode_ms']['median']:.2f} | "
                f"{values['decode_ms']['iqr']:.2f} | {values['generated_tokens']['median']:.2f} |"
            )
    return "\n".join(lines) + "\n"


def update_claim_ledger(path: str | Path, *, evidence_path: str) -> None:
    ledger_path = Path(path)
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    claim = {
        "id": "wall_clock_latency_descriptive",
        "claim": "Batch-1 wall-clock latency was measured for recurrent and token-space systems on their registered evaluation paths.",
        "status": "descriptive",
        "scope": "single hardware configuration, batch size 1, registered evaluation paths",
        "evidence": [{"path": evidence_path, "locator": "per-arm and per-depth latency receipt"}],
    }
    claims = payload.setdefault("claims", [])
    for index, existing in enumerate(claims):
        if existing.get("id") == claim["id"]:
            claims[index] = claim
            break
    else:
        claims.append(claim)
    ledger_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def hardware_receipt() -> dict[str, Any]:
    import torch
    import transformers

    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    nvidia_smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout.strip()
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout.strip()
    process_lines = [line for line in processes.splitlines() if line.strip()]
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": None if gpu is None else gpu.name,
        "gpu_total_memory": None if gpu is None else gpu.total_memory,
        "nvidia_smi": nvidia_smi,
        "compute_processes_at_start": processes,
        "compute_process_count": len(process_lines),
        "gpu_exclusivity_observed": len(process_lines) <= 1,
        "gpu_exclusivity_note": "At most the current evaluator process was visible to nvidia-smi.",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _sync(device: str) -> None:
    import torch

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _timed_dense_greedy(model: Any, encoded: dict[str, Any], *, max_new_tokens: int, eos_token_id: int | None, device: str):
    import torch

    input_ids = encoded["input_ids"]
    model_kwargs = {key: value for key, value in encoded.items() if key != "input_ids"}
    model_kwargs["use_cache"] = True
    generated: list[int] = []
    prefill_ms = 0.0
    decode_ms = 0.0
    with torch.inference_mode():
        for step in range(int(max_new_tokens)):
            _sync(device)
            started = time.perf_counter_ns()
            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = model(**model_inputs, return_dict=True)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=False,
            )
            input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
            _sync(device)
            elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            if step == 0:
                prefill_ms += elapsed
            else:
                decode_ms += elapsed
            token = int(next_token.item())
            generated.append(token)
            if eos_token_id is not None and token == int(eos_token_id):
                break
    return generated, prefill_ms, decode_ms


def _assert_dense_equivalence(model: Any, tokenizer: Any, encoded: dict[str, Any], *, max_new_tokens: int, device: str):
    import torch

    manual, _, _ = _timed_dense_greedy(
        model,
        {key: value.clone() for key, value in encoded.items()},
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        device=device,
    )
    with torch.inference_mode():
        standard = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    expected = standard[0, encoded["input_ids"].shape[1] :].tolist()
    if manual != expected:
        raise RuntimeError(f"Manual cached greedy path differs from registered generate path: {manual[:16]} != {expected[:16]}")


def _measure_dense(model: Any, tokenizer: Any, row: dict[str, Any], *, max_new_tokens: int, device: str) -> dict[str, Any]:
    from eval.eval_synthetic_depth_dense import extract_final_symbol, prompt_for_row, candidates_for_row

    prompt = prompt_for_row(row)
    started = time.perf_counter_ns()
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    tokenization_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    generated, prefill_ms, decode_ms = _timed_dense_greedy(
        model,
        encoded,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        device=device,
    )
    continuation = tokenizer.decode(generated, skip_special_tokens=True)
    prediction = extract_final_symbol(continuation, candidates_for_row(row))
    model_total = prefill_ms + decode_ms
    return {
        "tokenization_ms": tokenization_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "model_total_ms": model_total,
        "total_ms": tokenization_ms + model_total,
        "generated_tokens": len(generated),
        "prediction": prediction,
    }


def _recurrent_call(wrapper: Any, encoded: dict[str, Any], candidate_ids: dict[str, int], *, loops: int, device: str):
    import torch
    from eval.eval_mcq import select_forced_loop_logits

    _sync(device)
    started = time.perf_counter_ns()
    with torch.inference_mode():
        output = wrapper(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=None,
            max_loops=int(loops),
            num_trajectories=1,
            particle_update_mode="none",
            use_cache=False,
            return_dict=True,
            return_loop_logits=True,
        )
        logits = select_forced_loop_logits(output, int(loops))[0, -1]
        prediction = max(candidate_ids, key=lambda name: float(logits[candidate_ids[name]].item()))
    _sync(device)
    return prediction, (time.perf_counter_ns() - started) / 1_000_000.0


def _measure_recurrent(wrapper: Any, tokenizer: Any, row: dict[str, Any], *, device: str) -> dict[str, Any]:
    from eval.eval_synthetic_depth_active_labels import candidates_for_row, prompt_for_row, single_token_candidate_ids

    prompt = prompt_for_row(row, prediction_space="full_symbols", prompt_style="question_only")
    candidates = candidates_for_row(row, prediction_space="full_symbols", value_prefix="letter:")
    candidate_ids = single_token_candidate_ids(tokenizer, prompt, candidates)
    if candidate_ids is None:
        raise RuntimeError(f"Registered recurrent reader is not single-token for row {row_id(row)}")
    started = time.perf_counter_ns()
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    tokenization_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    _, reference_ms = _recurrent_call(wrapper, encoded, candidate_ids, loops=1, device=device)
    prediction, forced_ms = _recurrent_call(
        wrapper,
        encoded,
        candidate_ids,
        loops=int(row["depth"]),
        device=device,
    )
    prefill_ms = min(reference_ms, forced_ms)
    decode_ms = max(forced_ms - prefill_ms, 0.0)
    return {
        "tokenization_ms": tokenization_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "model_total_ms": forced_ms,
        "total_ms": tokenization_ms + forced_ms,
        "generated_tokens": 1,
        "prediction": prediction,
        "one_loop_reference_ms": reference_ms,
    }


def _load_arm(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from eval.eval_identity import model_load_kwargs

    if args.arm in {"A", "E"}:
        from eval.eval_mcq import load_recurrent_wrapper

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        wrapper = load_recurrent_wrapper(args, args.checkpoint)
        return tokenizer, wrapper
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        **model_load_kwargs(args.dtype, args.attn_implementation, low_cpu_mem_usage=True),
    ).to(args.device)
    model.eval()
    return tokenizer, model


def _mirror(output_dir: Path, mirror_dir: Path | None) -> None:
    if mirror_dir is None:
        return
    import shutil

    mirror_dir.mkdir(parents=True, exist_ok=True)
    for name in ("raw_timings.jsonl", "status.json", "summary.json"):
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, mirror_dir / name)


def evaluate_arm(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir = Path(args.mirror_dir) if args.mirror_dir else None
    raw_path = output_dir / "raw_timings.jsonl"
    if not raw_path.exists() and mirror_dir is not None and (mirror_dir / raw_path.name).exists():
        import shutil

        shutil.copy2(mirror_dir / raw_path.name, raw_path)
    completed = {str(row["observation_key"]) for row in read_jsonl(raw_path)} if raw_path.exists() else set()
    rows = read_jsonl(args.data_jsonl)
    schedule = build_observation_schedule(
        rows,
        stability_depths=tuple(int(value) for value in args.stability_depths.split(",")),
        stability_rows=args.stability_rows,
        repeats=args.stability_repeats,
    )
    if len(completed) == len(schedule) and (output_dir / "summary.json").exists():
        print(f"latency_arm_already_complete={args.arm} observations={len(completed)}", flush=True)
        return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    tokenizer, model = _load_arm(args)
    warmup = max(rows, key=lambda row: int(row["depth"]))
    if args.arm in {"A", "E"}:
        _measure_recurrent(model, tokenizer, warmup, device=args.device)
    else:
        prompt = __import__("eval.eval_synthetic_depth_dense", fromlist=["prompt_for_row"]).prompt_for_row(warmup)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        _assert_dense_equivalence(
            model,
            tokenizer,
            encoded,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
    status = {
        "kind": "stage5_wall_clock_latency_arm_status",
        "arm": args.arm,
        "status": "running",
        "total_observations": len(schedule),
        "completed_observations": len(completed),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": args.checkpoint_sha256,
    }
    (output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    with raw_path.open("a", encoding="utf-8", buffering=1) as handle:
        for index, item in enumerate(schedule, start=1):
            key = observation_key(item)
            if key in completed:
                continue
            row = item["row"]
            if args.arm in {"A", "E"}:
                result = _measure_recurrent(model, tokenizer, row, device=args.device)
            else:
                result = _measure_dense(
                    model,
                    tokenizer,
                    row,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                )
            record = {
                "kind": "stage5_wall_clock_latency_observation",
                "observation_key": key,
                "arm": args.arm,
                "phase": item["phase"],
                "repeat": int(item["repeat"]),
                "row_id": row_id(row),
                "depth": int(row["depth"]),
                **result,
            }
            record["target"] = str(row.get("target", "")).strip().upper()
            record["correct"] = str(record.get("prediction") or "").strip().upper() == record["target"]
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            completed.add(key)
            if len(completed) % int(args.progress_every) == 0 or len(completed) == len(schedule):
                status["completed_observations"] = len(completed)
                status["updated_at_unix"] = time.time()
                (output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                _mirror(output_dir, mirror_dir)
                print(f"latency_progress arm={args.arm} observation={len(completed)}/{len(schedule)}", flush=True)
    records = read_jsonl(raw_path)
    arm_summary = summarize_records(records)
    arm_summary.update(
        {
            "arm": args.arm,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": args.checkpoint_sha256,
            "data_jsonl": args.data_jsonl,
            "precision": args.dtype,
            "batch_size": 1,
            "greedy": True,
            "hardware": hardware_receipt(),
            "timing_decomposition": (
                "forced-depth total minus synchronized one-loop reference"
                if args.arm in {"A", "E"}
                else "first cached greedy forward prefill; subsequent cached forwards decode"
            ),
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(arm_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status.update({"status": "finished", "completed_observations": len(completed), "updated_at_unix": time.time()})
    (output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    _mirror(output_dir, mirror_dir)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return arm_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARM_ORDER)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint_sha256", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mirror_dir")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--split", default="6,18")
    parser.add_argument("--bridge_projection_mode", default="split")
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--adapter_dtype", default="float32")
    parser.add_argument("--attn_implementation", default="default")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--stability_depths", default="4,8,12")
    parser.add_argument("--stability_rows", type=int, default=32)
    parser.add_argument("--stability_repeats", type=int, default=3)
    parser.add_argument("--progress_every", type=int, default=128)
    args = parser.parse_args()
    evaluate_arm(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
