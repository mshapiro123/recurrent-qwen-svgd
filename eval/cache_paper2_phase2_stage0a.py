"""Build the resumable DEV-only Phase-2 Stage 0A teacher lattice and state cache."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_identity import model_load_kwargs
from training.paper2_phase2_stage0a import (
    STAGE0A_CONFIG,
    build_sparse_union,
    coarse_lattice_metrics,
    post_block_hidden_state_indices,
    select_stage0a_samples,
    sha256_file,
    stable_fraction,
)


ROWS_PER_SHARD = 8
INFERENCE_LOGIT_CHUNK = 16
UNION_SCORE_CHUNK = 8
UNION_SCORE_SCHEMA_VERSION = 4
TOPK_LOG_PROB_EQUIVALENCE_TOLERANCE = 0.25
TOPK_PROB_EQUIVALENCE_TOLERANCE = 0.01
SPARSE_MASS_PROJECTION_MAX_OVERFLOW = 0.05
ACTIVE_CONFIG = STAGE0A_CONFIG


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
            count += 1
    os.replace(temporary, destination)
    return {"path": str(destination), "rows": count, "sha256": sha256_file(destination)}


def atomic_torch_save(payload: Any, destination: Path, *, staging_dir: Path) -> None:
    """Save locally, hash, then atomically replace the Drive destination."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_dir) as temporary_dir:
        local = Path(temporary_dir) / destination.name
        torch.save(payload, local)
        partial = destination.with_suffix(destination.suffix + ".partial")
        shutil.copy2(local, partial)
        if sha256_file(local) != sha256_file(partial):
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Stage 0A atomic copy hash mismatch: {destination}")
        os.replace(partial, destination)


def _sample_indices_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(str(int(value)) for value in values).encode("ascii")
    ).hexdigest()


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "count": len(values),
        "min": float(tensor.min()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "median": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
    }


def _manifest_paths(private_dir: Path) -> tuple[Path, Path]:
    return private_dir / "sample_manifest.jsonl", private_dir / "sample_manifest_summary.json"


def prepare_sample_manifest(
    *, data_path: Path, private_dir: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(data_path) != config["data_sha256"]:
        raise RuntimeError("Stage 0A DEV-C hash mismatch")
    rows = read_jsonl(data_path)
    sample_path, summary_path = _manifest_paths(private_dir)
    if sample_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("data_sha256") == config["data_sha256"]
            and summary.get("config_sha256") == _config_sha256(config)
            and summary.get("sample_manifest_sha256") == sha256_file(sample_path)
        ):
            samples = read_jsonl(sample_path)
            if len(samples) == int(config["boundary_sample_count"]):
                print(f"stage0a_manifest_resume={sample_path}", flush=True)
                return rows, samples, summary

    selected = select_stage0a_samples(
        rows,
        anchors_per_stratum={
            key: int(value) for key, value in config["anchors_per_stratum"].items()
        },
        horizons=tuple(config["horizons"]),
        seed=int(config["seed"]),
    )
    if selected["boundary_sample_count"] != int(config["boundary_sample_count"]):
        raise RuntimeError("Stage 0A boundary-sample count differs from the locked config")

    audit_count = round(
        selected["boundary_sample_count"] * float(config["full_logit_audit_fraction"])
    )
    ranked = sorted(
        selected["samples"],
        key=lambda row: stable_fraction(
            "full_logit_audit", row["sample_key"], seed=int(config["seed"])
        ),
    )
    audit_keys = {row["sample_key"] for row in ranked[:audit_count]}
    samples = []
    for row in selected["samples"]:
        samples.append({**row, "full_logit_audit": row["sample_key"] in audit_keys})
    receipt = write_jsonl(sample_path, samples)
    summary = {
        "kind": "paper2_phase2_stage0a_sample_manifest",
        "status": "complete_development_only",
        "data_sha256": config["data_sha256"],
        "config_sha256": _config_sha256(config),
        "sample_manifest_sha256": receipt["sha256"],
        "position_key_sha256": selected["position_key_sha256"],
        "anchor_count": selected["anchor_count"],
        "boundary_sample_count": selected["boundary_sample_count"],
        "counts_by_stratum": selected["counts_by_stratum"],
        "full_logit_audit_samples": audit_count,
        "full_logit_audit_sample_keys_sha256": hashlib.sha256(
            ("\n".join(sorted(audit_keys)) + "\n").encode("utf-8")
        ).hexdigest(),
        "document_isolated": True,
        "span_boundaries_available": sum(
            len(row.get("span_boundaries") or row.get("trace_boundaries") or [])
            for row in rows
        ),
        "span_boundary_policy": "up_to_four_when_present; none synthesized",
        "training_started": False,
        "optimizer_steps": 0,
        "frozen_evaluation_partitions_touched": [],
    }
    write_json(summary_path, summary)
    return rows, samples, summary


def _config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _samples_by_row(samples: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        result[int(sample["row_index"])].append(sample)
    for values in result.values():
        values.sort(key=lambda row: int(row["sample_index"]))
    return dict(result)


def _shard_path(cache_root: Path, model_key: str, start: int, stop: int) -> Path:
    return cache_root / model_key / f"rows_{start:06d}_{stop:06d}.pt"


def completed_model_shard(
    path: Path,
    *,
    model_key: str,
    revision: str,
    expected_sample_indices: Sequence[int],
    position_key_sha256: str,
) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return (
        payload.get("kind") == "paper2_phase2_stage0a_model_shard"
        and payload.get("model_key") == model_key
        and payload.get("revision") == revision
        and payload.get("position_key_sha256") == position_key_sha256
        and payload.get("sample_indices_sha256")
        == _sample_indices_sha256(expected_sample_indices)
        and payload.get("sample_indices", torch.empty(0, dtype=torch.long)).tolist()
        == list(expected_sample_indices)
    )


def _vocabulary_receipt(tokenizer: Any) -> dict[str, Any]:
    vocab = {str(token): int(token_id) for token, token_id in tokenizer.get_vocab().items()}
    ids = sorted(vocab.values())
    if ids != list(range(len(ids))):
        raise RuntimeError("Stage 0A requires a contiguous shared tokenizer vocabulary")
    return {
        "vocabulary_size": len(vocab),
        "vocabulary_sha256": hashlib.sha256(
            json.dumps(vocab, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _validate_shared_vocabulary(receipt: dict[str, Any], private_dir: Path) -> None:
    path = private_dir / "shared_vocabulary.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != receipt:
            raise RuntimeError(
                "Stage 0A teacher/student tokenizers are not exactly ID-aligned"
            )
    else:
        write_json(path, receipt)


def _load_model(
    *,
    model_name: str,
    revision: str,
    device: str,
    dtype: str,
    attn_implementation: str,
    cpu_offload: bool = False,
    offload_dir: Path | None = None,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    kwargs = model_load_kwargs(
        dtype,
        attn_implementation,
        revision=revision,
        low_cpu_mem_usage=True,
    )
    if device.startswith("cuda"):
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        if cpu_offload:
            if offload_dir is None:
                raise ValueError("32B CPU offload requires a local offload directory")
            offload_dir.mkdir(parents=True, exist_ok=True)
            gpu_total_gib = int(
                torch.cuda.get_device_properties(index).total_memory // 1024**3
            )
            cpu_total_gib = int(
                os.sysconf("SC_PHYS_PAGES")
                * os.sysconf("SC_PAGE_SIZE")
                // 1024**3
            )
            kwargs.update(
                {
                    "device_map": "auto",
                    "max_memory": {
                        index: f"{max(8, gpu_total_gib - 4)}GiB",
                        "cpu": f"{max(8, cpu_total_gib - 16)}GiB",
                    },
                    "offload_folder": str(offload_dir),
                    "offload_state_dict": True,
                    "offload_buffers": True,
                }
            )
        else:
            kwargs["device_map"] = {"": index}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return tokenizer, model


def _model_device(model: Any) -> torch.device:
    hook = getattr(model, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        return torch.device(execution_device)
    device_map = getattr(model, "hf_device_map", None) or {}
    for value in device_map.values():
        if value not in {"cpu", "disk"}:
            return torch.device(f"cuda:{value}" if isinstance(value, int) else value)
    return next(model.parameters()).device


def _materialize_weight(module: Any, name: str = "weight") -> torch.Tensor:
    weight = getattr(module, name)
    if weight.device.type != "meta":
        return weight
    hook = getattr(module, "_hf_hook", None)
    weights_map = getattr(hook, "weights_map", None)
    if weights_map is None:
        raise RuntimeError(f"Cannot materialize offloaded parameter {name}")
    return weights_map[name]


def _topk_statistics(
    model: Any, hidden: torch.Tensor, *, shared_vocab_size: int, top_k: int
) -> dict[str, torch.Tensor]:
    top_ids: list[torch.Tensor] = []
    top_log_probs: list[torch.Tensor] = []
    log_partitions: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for start in range(0, hidden.shape[0], INFERENCE_LOGIT_CHUNK):
        local = hidden[start : start + INFERENCE_LOGIT_CHUNK]
        logits = model.lm_head(local).float()[..., :shared_vocab_size]
        log_partition = torch.logsumexp(logits, dim=-1)
        log_probs = logits - log_partition.unsqueeze(-1)
        values, ids = torch.topk(log_probs, k=top_k, dim=-1)
        probabilities = log_probs.exp()
        entropy = -(probabilities * log_probs).sum(dim=-1)
        top_ids.append(ids.to(torch.int32).cpu())
        top_log_probs.append(values.to(torch.bfloat16).cpu())
        log_partitions.append(log_partition.cpu())
        entropies.append(entropy.cpu())
        del logits, log_probs, probabilities
    return {
        "topk_ids": torch.cat(top_ids),
        "topk_log_probs": torch.cat(top_log_probs),
        "log_partition": torch.cat(log_partitions),
        "entropy": torch.cat(entropies),
    }


def _save_lm_head(
    *, model: Any, model_key: str, model_spec: dict[str, Any], shared_vocab_size: int,
    cache_root: Path, staging_dir: Path
) -> dict[str, Any]:
    destination = cache_root / model_key / "lm_head.pt"
    if destination.exists():
        payload = torch.load(destination, map_location="cpu", weights_only=False)
        if (
            payload.get("kind") == "paper2_phase2_stage0a_lm_head"
            and payload.get("revision") == model_spec["revision"]
            and int(payload.get("shared_vocab_size", 0)) == shared_vocab_size
        ):
            return {
                "path": str(destination),
                "sha256": sha256_file(destination),
                "shape": list(payload["weight_bfloat16"].shape),
            }
    weight = (
        _materialize_weight(model.lm_head)[:shared_vocab_size]
        .detach()
        .to(torch.bfloat16)
        .cpu()
    )
    payload = {
        "kind": "paper2_phase2_stage0a_lm_head",
        "model_key": model_key,
        "model": model_spec["model"],
        "revision": model_spec["revision"],
        "shared_vocab_size": shared_vocab_size,
        "weight_bfloat16": weight,
    }
    atomic_torch_save(payload, destination, staging_dir=staging_dir)
    return {"path": str(destination), "sha256": sha256_file(destination), "shape": list(weight.shape)}


@torch.inference_mode()
def cache_model_pass(
    *,
    model_key: str,
    rows: Sequence[dict[str, Any]],
    samples: Sequence[dict[str, Any]],
    position_key_sha256: str,
    private_dir: Path,
    staging_dir: Path,
    allowed_sample_indices: set[int] | None,
    device: str,
    dtype: str,
    attn_implementation: str,
    offload_32b: bool = False,
    offload_dir: Path | None = None,
) -> dict[str, Any]:
    model_spec = ACTIVE_CONFIG["models"][model_key]
    samples_by_row = _samples_by_row(samples)
    cache_root = private_dir / "model_cache"
    expected_by_shard: list[tuple[int, int, list[int]]] = []
    for start in range(0, len(rows), ROWS_PER_SHARD):
        stop = min(len(rows), start + ROWS_PER_SHARD)
        indices = [
            int(sample["sample_index"])
            for row_index in range(start, stop)
            for sample in samples_by_row.get(row_index, [])
            if allowed_sample_indices is None
            or int(sample["sample_index"]) in allowed_sample_indices
        ]
        expected_by_shard.append((start, stop, indices))

    if all(
        completed_model_shard(
            _shard_path(cache_root, model_key, start, stop),
            model_key=model_key,
            revision=model_spec["revision"],
            expected_sample_indices=indices,
            position_key_sha256=position_key_sha256,
        )
        for start, stop, indices in expected_by_shard
    ) and (cache_root / model_key / "summary.json").exists():
        print(f"stage0a_model_resume_complete={model_key}", flush=True)
        return json.loads((cache_root / model_key / "summary.json").read_text(encoding="utf-8"))

    tokenizer, model = _load_model(
        model_name=model_spec["model"],
        revision=model_spec["revision"],
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        cpu_offload=bool(offload_32b and model_key == "teacher_32b"),
        offload_dir=(offload_dir / model_key if offload_dir is not None else None),
    )
    execution_mode = (
        "accelerate_cpu_disk_offload_cuda_execution"
        if offload_32b and model_key == "teacher_32b"
        else "fully_resident_cuda"
    )
    vocab_receipt = _vocabulary_receipt(tokenizer)
    _validate_shared_vocabulary(vocab_receipt, private_dir)
    shared_vocab_size = int(vocab_receipt["vocabulary_size"])
    if int(model.config.vocab_size) < shared_vocab_size:
        raise RuntimeError("Stage 0A model output vocabulary is smaller than the tokenizer")

    collect_teacher_states = model_key == ACTIVE_CONFIG["teacher_state_model"]["key"]
    if collect_teacher_states:
        expected = ACTIVE_CONFIG["teacher_state_model"]
        if int(model.config.hidden_size) != int(expected["hidden_size"]):
            raise RuntimeError("Stage 0A 14B hidden width differs from the locked design")
        if int(model.config.num_hidden_layers) != int(expected["num_hidden_layers"]):
            raise RuntimeError("Stage 0A 14B layer count differs from the locked design")
        hidden_indices = post_block_hidden_state_indices(
            num_hidden_layers=int(model.config.num_hidden_layers),
            ordinals_one_based=ACTIVE_CONFIG["selected_layer_ordinals_one_based"],
        )
    else:
        hidden_indices = ()

    head_receipt = _save_lm_head(
        model=model,
        model_key=model_key,
        model_spec=model_spec,
        shared_vocab_size=shared_vocab_size,
        cache_root=cache_root,
        staging_dir=staging_dir,
    )
    rows_forwarded = 0
    samples_cached = 0
    shard_seconds_per_sample: list[float] = []
    started_at = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for shard_number, (start, stop, expected_indices) in enumerate(expected_by_shard, start=1):
        destination = _shard_path(cache_root, model_key, start, stop)
        if completed_model_shard(
            destination,
            model_key=model_key,
            revision=model_spec["revision"],
            expected_sample_indices=expected_indices,
            position_key_sha256=position_key_sha256,
        ):
            samples_cached += len(expected_indices)
            print(
                f"stage0a_model_resume model={model_key} shard={shard_number}/{len(expected_by_shard)} "
                f"samples={samples_cached}",
                flush=True,
            )
            continue
        shard_started_at = time.perf_counter()
        payload_rows: list[dict[str, torch.Tensor | int]] = []
        for row_index in range(start, stop):
            selected = [
                sample
                for sample in samples_by_row.get(row_index, [])
                if allowed_sample_indices is None
                or int(sample["sample_index"]) in allowed_sample_indices
            ]
            if not selected:
                continue
            values = torch.tensor(
                [rows[row_index]["input_ids"]], dtype=torch.long, device=_model_device(model)
            )
            attention = torch.ones_like(values)
            output = model.model(
                input_ids=values,
                attention_mask=attention,
                use_cache=False,
                output_hidden_states=collect_teacher_states,
                return_dict=True,
            )
            prediction_positions = torch.tensor(
                [int(sample["prediction_position"]) for sample in selected],
                dtype=torch.long,
                device=values.device,
            )
            final_hidden = output.last_hidden_state[0].index_select(
                0, prediction_positions
            )
            statistics = _topk_statistics(
                model,
                final_hidden,
                shared_vocab_size=shared_vocab_size,
                top_k=int(ACTIVE_CONFIG["top_k"]),
            )
            row_payload: dict[str, Any] = {
                "row_index": row_index,
                "sample_indices": torch.tensor(
                    [int(sample["sample_index"]) for sample in selected], dtype=torch.long
                ),
                "final_hidden_bfloat16": final_hidden.to(torch.bfloat16).cpu(),
                **statistics,
            }
            if collect_teacher_states:
                state_positions = torch.tensor(
                    [int(sample["state_position"]) for sample in selected],
                    dtype=torch.long,
                    device=values.device,
                )
                assert output.hidden_states is not None
                state_layers = [
                    output.hidden_states[index][0].index_select(0, state_positions)
                    for index in hidden_indices
                ]
                row_payload["teacher_states_bfloat16"] = torch.stack(
                    state_layers, dim=1
                ).to(torch.bfloat16).cpu()
            payload_rows.append(row_payload)
            rows_forwarded += 1
            samples_cached += len(selected)
            del output, final_hidden, statistics
        payload = {
            "kind": "paper2_phase2_stage0a_model_shard",
            "model_key": model_key,
            "model": model_spec["model"],
            "revision": model_spec["revision"],
            "position_key_sha256": position_key_sha256,
            "row_start": start,
            "row_stop": stop,
            "sample_indices": torch.tensor(expected_indices, dtype=torch.long),
            "sample_indices_sha256": _sample_indices_sha256(expected_indices),
            "rows": payload_rows,
            "shared_vocab_size": shared_vocab_size,
            "teacher_state_layer_ordinals_one_based": list(hidden_indices),
        }
        atomic_torch_save(payload, destination, staging_dir=staging_dir)
        shard_elapsed = time.perf_counter() - shard_started_at
        if expected_indices:
            shard_seconds_per_sample.append(shard_elapsed / len(expected_indices))
        print(
            f"stage0a_model_progress model={model_key} shard={shard_number}/{len(expected_by_shard)} "
            f"samples={samples_cached}",
            flush=True,
        )

    shard_receipts = [
        {
            "path": str(_shard_path(cache_root, model_key, start, stop)),
            "sha256": sha256_file(_shard_path(cache_root, model_key, start, stop)),
            "samples": len(indices),
        }
        for start, stop, indices in expected_by_shard
    ]
    elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
    summary = {
        "kind": "paper2_phase2_stage0a_model_cache",
        "status": "complete_development_only",
        "model_key": model_key,
        "model": model_spec["model"],
        "revision": model_spec["revision"],
        "tokenizer": vocab_receipt,
        "position_key_sha256": position_key_sha256,
        "samples": sum(row["samples"] for row in shard_receipts),
        "shards": shard_receipts,
        "lm_head": head_receipt,
        "teacher_states_collected": collect_teacher_states,
        "teacher_state_layer_ordinals_one_based": list(hidden_indices),
        "teacher_forward_passes": 1,
        "forward_rows_this_invocation": rows_forwarded,
        "elapsed_seconds_this_invocation": elapsed_seconds,
        "samples_per_second_this_invocation": samples_cached / elapsed_seconds,
        "shard_seconds_per_sample_this_invocation": _quantiles(
            shard_seconds_per_sample
        ),
        "peak_gpu_memory_bytes_this_invocation": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "execution_mode": execution_mode,
        "hf_device_map": {
            str(key): str(value)
            for key, value in (getattr(model, "hf_device_map", None) or {}).items()
        },
        "resume_policy": "completed shards never re-forwarded; an interrupted partial shard may replay",
        "training_started": False,
        "optimizer_steps": 0,
        "frozen_evaluation_partitions_touched": [],
    }
    write_json(cache_root / model_key / "summary.json", summary)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _load_flat_shard(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = payload["rows"]
    if not rows:
        return {
            "sample_indices": torch.empty(0, dtype=torch.long),
            "topk_ids": torch.empty((0, ACTIVE_CONFIG["top_k"]), dtype=torch.int32),
            "topk_log_probs": torch.empty((0, ACTIVE_CONFIG["top_k"]), dtype=torch.bfloat16),
            "log_partition": torch.empty(0),
            "entropy": torch.empty(0),
            "final_hidden_bfloat16": torch.empty((0, 0), dtype=torch.bfloat16),
        }
    keys = (
        "sample_indices",
        "topk_ids",
        "topk_log_probs",
        "log_partition",
        "entropy",
        "final_hidden_bfloat16",
    )
    result = {key: torch.cat([row[key] for row in rows], dim=0) for key in keys}
    if "teacher_states_bfloat16" in rows[0]:
        result["teacher_states_bfloat16"] = torch.cat(
            [row["teacher_states_bfloat16"] for row in rows], dim=0
        )
    return result


def build_32b_cascade(
    *, rows: Sequence[dict[str, Any]], samples: Sequence[dict[str, Any]], private_dir: Path
) -> tuple[set[int], dict[str, Any]]:
    cache_root = private_dir / "model_cache"
    selected: set[int] = set()
    disagreements = 0
    verifier_selected = 0
    audit_selected = 0
    for start in range(0, len(rows), ROWS_PER_SHARD):
        stop = min(len(rows), start + ROWS_PER_SHARD)
        seven = _load_flat_shard(_shard_path(cache_root, "teacher_7b", start, stop))
        fourteen = _load_flat_shard(_shard_path(cache_root, "teacher_14b", start, stop))
        if seven["sample_indices"].tolist() != fourteen["sample_indices"].tolist():
            raise RuntimeError("Stage 0A 7B/14B sample alignment failed")
        seven_argmax = seven["topk_ids"][:, 0]
        fourteen_argmax = fourteen["topk_ids"][:, 0]
        for offset, sample_index in enumerate(seven["sample_indices"].tolist()):
            sample = samples[sample_index]
            disagreement = int(seven_argmax[offset]) != int(fourteen_argmax[offset])
            verifier = bool(sample.get("verifier_available"))
            audit = bool(sample["full_logit_audit"])
            if disagreement or verifier or audit:
                selected.add(sample_index)
                disagreements += int(disagreement)
                verifier_selected += int(verifier and not disagreement)
                audit_selected += int(audit and not disagreement and not verifier)
    ordered = sorted(selected)
    path = private_dir / "teacher_32b_cascade_indices.json"
    receipt = {
        "kind": "paper2_phase2_stage0a_32b_cascade",
        "status": "complete_development_only",
        "selected_samples": len(ordered),
        "selected_fraction": len(ordered) / len(samples),
        "seven_fourteen_argmax_disagreements": disagreements,
        "verifier_only": verifier_selected,
        "stable_audit_only": audit_selected,
        "sample_indices": ordered,
        "selection_sha256": _sample_indices_sha256(ordered),
        "training_started": False,
        "optimizer_steps": 0,
    }
    write_json(path, receipt)
    return selected, receipt


def build_union_shards(
    *, rows: Sequence[dict[str, Any]], private_dir: Path, staging_dir: Path
) -> dict[str, Any]:
    model_root = private_dir / "model_cache"
    union_root = private_dir / "union"
    summary_path = union_root / "summary.json"
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            shards = existing.get("shards", [])
            complete = (
                existing.get("kind") == "paper2_phase2_stage0a_union"
                and existing.get("samples")
                == int(ACTIVE_CONFIG["boundary_sample_count"])
                and len(shards) == math.ceil(len(rows) / ROWS_PER_SHARD)
                and all(
                    Path(receipt["path"]).is_file()
                    and sha256_file(receipt["path"]) == receipt["sha256"]
                    for receipt in shards
                )
            )
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            complete = False
        if complete:
            print(f"stage0a_union_resume_complete={summary_path}", flush=True)
            return existing
    receipts = []
    for start in range(0, len(rows), ROWS_PER_SHARD):
        stop = min(len(rows), start + ROWS_PER_SHARD)
        destination = union_root / f"rows_{start:06d}_{stop:06d}.pt"
        sources = {
            key: _load_flat_shard(_shard_path(model_root, key, start, stop))
            for key in ("student_0p5b", "teacher_7b", "teacher_14b", "teacher_32b")
        }
        base_indices = sources["student_0p5b"]["sample_indices"]
        if not (
            base_indices.tolist() == sources["teacher_7b"]["sample_indices"].tolist()
            == sources["teacher_14b"]["sample_indices"].tolist()
        ):
            raise RuntimeError("Stage 0A base lattice model sample alignment failed")
        n = int(base_indices.numel())
        topk_inputs = [
            sources[key]["topk_ids"].long()
            for key in ("student_0p5b", "teacher_7b", "teacher_14b")
        ]
        thirty_two_topk = torch.full(
            (n, int(ACTIVE_CONFIG["top_k"])), -1, dtype=torch.long
        )
        thirty_two_present = torch.zeros(n, dtype=torch.bool)
        base_lookup = {value: offset for offset, value in enumerate(base_indices.tolist())}
        for offset, sample_index in enumerate(sources["teacher_32b"]["sample_indices"].tolist()):
            target = base_lookup[sample_index]
            thirty_two_topk[target] = sources["teacher_32b"]["topk_ids"][offset].long()
            thirty_two_present[target] = True
        topk_inputs.append(thirty_two_topk)
        union_ids, union_mask = build_sparse_union(topk_inputs)
        payload = {
            "kind": "paper2_phase2_stage0a_union_shard",
            "row_start": start,
            "row_stop": stop,
            "sample_indices": base_indices,
            "union_ids": union_ids.to(torch.int32),
            "union_mask": union_mask,
            "teacher_32b_present": thirty_two_present,
            "topk_ids": {
                "student_0p5b": topk_inputs[0].to(torch.int32),
                "teacher_7b": topk_inputs[1].to(torch.int32),
                "teacher_14b": topk_inputs[2].to(torch.int32),
                "teacher_32b": topk_inputs[3].to(torch.int32),
            },
        }
        atomic_torch_save(payload, destination, staging_dir=staging_dir)
        receipts.append(
            {
                "path": str(destination),
                "sha256": sha256_file(destination),
                "samples": n,
                "max_union_width": int(union_ids.shape[1]) if n else 0,
            }
        )
        print(f"stage0a_union_progress rows={stop}/{len(rows)}", flush=True)
    summary = {
        "kind": "paper2_phase2_stage0a_union",
        "status": "complete",
        "shards": receipts,
        "samples": sum(row["samples"] for row in receipts),
        "max_union_width": max((row["max_union_width"] for row in receipts), default=0),
    }
    write_json(summary_path, summary)
    return summary


def _score_candidates(
    *, hidden: torch.Tensor, head: torch.Tensor, candidate_ids: torch.Tensor,
    candidate_mask: torch.Tensor, log_partition: torch.Tensor, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.full(candidate_ids.shape, float("-inf"), dtype=torch.float32)
    for start in range(0, hidden.shape[0], UNION_SCORE_CHUNK):
        stop = min(hidden.shape[0], start + UNION_SCORE_CHUNK)
        local_ids = candidate_ids[start:stop].long()
        local_mask = candidate_mask[start:stop]
        clamped = local_ids.clamp_min(0).to(device)
        embeddings = head.index_select(0, clamped.reshape(-1)).view(
            stop - start, local_ids.shape[1], -1
        )
        local_hidden = hidden[start:stop].to(device=device, dtype=head.dtype)
        logits = torch.einsum("bd,bkd->bk", local_hidden, embeddings).float()
        log_probs = logits - log_partition[start:stop].to(device).unsqueeze(-1)
        log_probs = log_probs.masked_fill(~local_mask.to(device), float("-inf"))
        output[start:stop] = log_probs.cpu()
        del embeddings, logits, log_probs
    candidate_mass = torch.where(candidate_mask, output.exp(), torch.zeros_like(output)).sum(dim=-1)
    tail_mass = (1.0 - candidate_mass).clamp(min=1e-30, max=1.0)
    return output, tail_mass.log()


def completed_union_score_shard(
    path: Path,
    *,
    model_key: str,
    row_start: int,
    row_stop: int,
    union_sha256: str,
) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return (
        payload.get("kind") == "paper2_phase2_stage0a_union_score_shard"
        and payload.get("score_schema_version") == UNION_SCORE_SCHEMA_VERSION
        and payload.get("model_key") == model_key
        and payload.get("row_start") == row_start
        and payload.get("row_stop") == row_stop
        and payload.get("union_sha256") == union_sha256
    )


def apply_authoritative_topk(
    *,
    candidate_log_probs: torch.Tensor,
    union_ids: torch.Tensor,
    union_mask: torch.Tensor,
    cached_topk_ids: torch.Tensor,
    cached_topk_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Anchor to cached top-k and reconcile the sparse distribution on the simplex.

    The extra union candidates are rescored from bfloat16 cached hidden states and
    LM-head weights, while top-k values come from the original model forward. The
    cached top-k log probabilities are themselves bfloat16, so their rounded mass
    may slightly exceed one. In that case a proportional simplex projection keeps
    their relative probabilities; any remaining overflow then scales only the
    approximate extras into the residual mass.
    """

    valid_union = union_ids[union_mask].long()
    cached_ids = cached_topk_ids.long()
    positions = torch.searchsorted(valid_union, cached_ids)
    if bool((positions >= valid_union.numel()).any()) or not torch.equal(
        valid_union[positions], cached_ids
    ):
        raise RuntimeError("Stage 0A union omitted a cached top-k token")
    corrected = candidate_log_probs.float().clone()
    rescored = corrected[positions].float()
    cached = cached_topk_log_probs.float()
    diagnostics = {
        "log_probability_max_abs_error": float((rescored - cached).abs().max()),
        "probability_max_abs_error": float(
            (rescored.exp() - cached.exp()).abs().max()
        ),
    }
    corrected[positions] = cached

    topk_mask = torch.zeros_like(union_mask, dtype=torch.bool)
    topk_mask[positions] = True
    extra_mask = union_mask & ~topk_mask
    raw_topk_mass = cached.exp().sum()
    topk_mass_overflow = max(0.0, float(raw_topk_mass - 1.0))
    if topk_mass_overflow > SPARSE_MASS_PROJECTION_MAX_OVERFLOW:
        raise RuntimeError(
            "Stage 0A cached top-k mass projection is too large: "
            f"overflow={topk_mass_overflow} raw_topk_mass={float(raw_topk_mass)}"
        )
    topk_projection_scale = 1.0
    if topk_mass_overflow > 0.0:
        topk_projection_scale = float(1.0 / raw_topk_mass)
        corrected[positions] += math.log(topk_projection_scale)
    topk_mass = corrected[positions].exp().sum()
    available_extra_mass = (1.0 - topk_mass).clamp(min=0.0)
    raw_extra_mass = corrected[extra_mask].exp().sum()
    raw_candidate_mass = topk_mass + raw_extra_mass
    mass_overflow = max(0.0, float(raw_candidate_mass - 1.0))
    if mass_overflow > SPARSE_MASS_PROJECTION_MAX_OVERFLOW:
        raise RuntimeError(
            "Stage 0A sparse union mass projection is too large: "
            f"overflow={mass_overflow} raw_candidate_mass={float(raw_candidate_mass)}"
        )
    projection_scale = 1.0
    if float(raw_extra_mass) > float(available_extra_mass) and extra_mask.any():
        if float(available_extra_mass) <= 0.0:
            corrected[extra_mask] = float("-inf")
            projection_scale = 0.0
        else:
            projection_scale = float(available_extra_mass / raw_extra_mass)
            corrected[extra_mask] += math.log(projection_scale)

    candidate_mass = corrected[union_mask].exp().sum()
    tail_log_prob = (1.0 - candidate_mass).clamp(min=1e-30, max=1.0).log()
    diagnostics.update(
        {
            "raw_candidate_mass": float(raw_candidate_mass),
            "raw_topk_mass": float(raw_topk_mass),
            "topk_mass_overflow": topk_mass_overflow,
            "topk_mass_projection_scale": topk_projection_scale,
            "mass_overflow": mass_overflow,
            "extra_mass_projection_scale": projection_scale,
        }
    )
    return corrected, tail_log_prob, diagnostics


def score_union_for_model(
    *, model_key: str, rows: Sequence[dict[str, Any]], samples: Sequence[dict[str, Any]],
    private_dir: Path, staging_dir: Path, device: str
) -> dict[str, Any]:
    model_root = private_dir / "model_cache"
    union_root = private_dir / "union"
    score_root = private_dir / "union_scores" / model_key
    head_payload = torch.load(
        model_root / model_key / "lm_head.pt", map_location="cpu", weights_only=False
    )
    head = head_payload["weight_bfloat16"].to(device)
    receipts = []
    sample_lookup = {int(sample["sample_index"]): sample for sample in samples}
    maximum_topk_equivalence_error = 0.0
    maximum_topk_probability_error = 0.0
    maximum_mass_overflow = 0.0
    maximum_topk_mass_overflow = 0.0
    minimum_topk_mass_projection_scale = 1.0
    topk_projected_sample_count = 0
    minimum_extra_mass_projection_scale = 1.0
    projected_sample_count = 0
    for start in range(0, len(rows), ROWS_PER_SHARD):
        stop = min(len(rows), start + ROWS_PER_SHARD)
        destination = score_root / f"rows_{start:06d}_{stop:06d}.pt"
        union_path = union_root / f"rows_{start:06d}_{stop:06d}.pt"
        union_sha256 = sha256_file(union_path)
        if completed_union_score_shard(
            destination,
            model_key=model_key,
            row_start=start,
            row_stop=stop,
            union_sha256=union_sha256,
        ):
            payload = torch.load(destination, map_location="cpu", weights_only=False)
            receipts.append(
                {
                    "path": str(destination),
                    "sha256": sha256_file(destination),
                    "samples_present": int(payload["present"].sum()),
                    "audit_samples": int(payload["audit_sample_indices"].numel()),
                    "topk_equivalence_max_abs_error": float(
                        payload["topk_equivalence_max_abs_error"]
                    ),
                    "topk_probability_max_abs_error": float(
                        payload["topk_probability_max_abs_error"]
                    ),
                    "maximum_mass_overflow": float(payload["maximum_mass_overflow"]),
                    "maximum_topk_mass_overflow": float(
                        payload["maximum_topk_mass_overflow"]
                    ),
                    "minimum_topk_mass_projection_scale": float(
                        payload["minimum_topk_mass_projection_scale"]
                    ),
                    "topk_projected_sample_count": int(
                        payload["topk_projected_sample_count"]
                    ),
                    "minimum_extra_mass_projection_scale": float(
                        payload["minimum_extra_mass_projection_scale"]
                    ),
                    "projected_sample_count": int(payload["projected_sample_count"]),
                }
            )
            maximum_topk_equivalence_error = max(
                maximum_topk_equivalence_error,
                float(payload["topk_equivalence_max_abs_error"]),
            )
            maximum_topk_probability_error = max(
                maximum_topk_probability_error,
                float(payload["topk_probability_max_abs_error"]),
            )
            maximum_mass_overflow = max(
                maximum_mass_overflow, float(payload["maximum_mass_overflow"])
            )
            maximum_topk_mass_overflow = max(
                maximum_topk_mass_overflow,
                float(payload["maximum_topk_mass_overflow"]),
            )
            minimum_topk_mass_projection_scale = min(
                minimum_topk_mass_projection_scale,
                float(payload["minimum_topk_mass_projection_scale"]),
            )
            topk_projected_sample_count += int(
                payload["topk_projected_sample_count"]
            )
            minimum_extra_mass_projection_scale = min(
                minimum_extra_mass_projection_scale,
                float(payload["minimum_extra_mass_projection_scale"]),
            )
            projected_sample_count += int(payload["projected_sample_count"])
            print(
                f"stage0a_union_score_resume model={model_key} rows={stop}/{len(rows)}",
                flush=True,
            )
            continue
        union = torch.load(
            union_path,
            map_location="cpu",
            weights_only=False,
        )
        model_cache = _load_flat_shard(_shard_path(model_root, model_key, start, stop))
        base_indices = union["sample_indices"].tolist()
        model_indices = model_cache["sample_indices"].tolist()
        model_lookup = {value: offset for offset, value in enumerate(model_indices)}
        present = torch.tensor([value in model_lookup for value in base_indices], dtype=torch.bool)
        hidden_size = int(head.shape[1])
        hidden = torch.zeros((len(base_indices), hidden_size), dtype=torch.bfloat16)
        log_partition = torch.zeros(len(base_indices), dtype=torch.float32)
        entropy = torch.full((len(base_indices),), float("nan"), dtype=torch.float32)
        greedy = torch.full((len(base_indices),), -1, dtype=torch.int32)
        for base_offset, sample_index in enumerate(base_indices):
            if sample_index not in model_lookup:
                continue
            source_offset = model_lookup[sample_index]
            hidden[base_offset] = model_cache["final_hidden_bfloat16"][source_offset]
            log_partition[base_offset] = model_cache["log_partition"][source_offset]
            entropy[base_offset] = model_cache["entropy"][source_offset]
            greedy[base_offset] = model_cache["topk_ids"][source_offset, 0]
        candidate_log_probs = torch.full(
            union["union_ids"].shape, float("-inf"), dtype=torch.float32
        )
        tail_log_probs = torch.full((len(base_indices),), float("-inf"), dtype=torch.float32)
        if present.any():
            local_scores, local_tail = _score_candidates(
                hidden=hidden[present],
                head=head,
                candidate_ids=union["union_ids"][present],
                candidate_mask=union["union_mask"][present],
                log_partition=log_partition[present],
                device=device,
            )
            candidate_log_probs[present] = local_scores
            tail_log_probs[present] = local_tail

        equivalence_error = 0.0
        probability_error = 0.0
        shard_maximum_mass_overflow = 0.0
        shard_maximum_topk_mass_overflow = 0.0
        shard_minimum_topk_projection_scale = 1.0
        shard_topk_projected_sample_count = 0
        shard_minimum_projection_scale = 1.0
        shard_projected_sample_count = 0
        for base_offset, sample_index in enumerate(base_indices):
            if not present[base_offset]:
                continue
            source_offset = model_lookup[sample_index]
            corrected, corrected_tail, diagnostics = apply_authoritative_topk(
                candidate_log_probs=candidate_log_probs[base_offset],
                union_ids=union["union_ids"][base_offset],
                union_mask=union["union_mask"][base_offset],
                cached_topk_ids=model_cache["topk_ids"][source_offset],
                cached_topk_log_probs=model_cache["topk_log_probs"][source_offset],
            )
            candidate_log_probs[base_offset] = corrected
            tail_log_probs[base_offset] = corrected_tail
            equivalence_error = max(
                equivalence_error,
                diagnostics["log_probability_max_abs_error"],
            )
            probability_error = max(
                probability_error,
                diagnostics["probability_max_abs_error"],
            )
            shard_maximum_mass_overflow = max(
                shard_maximum_mass_overflow, diagnostics["mass_overflow"]
            )
            shard_maximum_topk_mass_overflow = max(
                shard_maximum_topk_mass_overflow,
                diagnostics["topk_mass_overflow"],
            )
            shard_minimum_topk_projection_scale = min(
                shard_minimum_topk_projection_scale,
                diagnostics["topk_mass_projection_scale"],
            )
            if diagnostics["topk_mass_projection_scale"] < 1.0:
                shard_topk_projected_sample_count += 1
            shard_minimum_projection_scale = min(
                shard_minimum_projection_scale,
                diagnostics["extra_mass_projection_scale"],
            )
            if diagnostics["extra_mass_projection_scale"] < 1.0:
                shard_projected_sample_count += 1
        if (
            equivalence_error > TOPK_LOG_PROB_EQUIVALENCE_TOLERANCE
            or probability_error > TOPK_PROB_EQUIVALENCE_TOLERANCE
        ):
            print(
                "stage0a_union_score_preanchor_diagnostic "
                f"model={model_key} rows={start}:{stop} "
                f"log_max_abs_error={equivalence_error} "
                f"probability_max_abs_error={probability_error}",
                flush=True,
            )
        maximum_topk_equivalence_error = max(
            maximum_topk_equivalence_error, equivalence_error
        )
        maximum_topk_probability_error = max(
            maximum_topk_probability_error, probability_error
        )
        maximum_mass_overflow = max(
            maximum_mass_overflow, shard_maximum_mass_overflow
        )
        maximum_topk_mass_overflow = max(
            maximum_topk_mass_overflow, shard_maximum_topk_mass_overflow
        )
        minimum_topk_mass_projection_scale = min(
            minimum_topk_mass_projection_scale,
            shard_minimum_topk_projection_scale,
        )
        topk_projected_sample_count += shard_topk_projected_sample_count
        minimum_extra_mass_projection_scale = min(
            minimum_extra_mass_projection_scale, shard_minimum_projection_scale
        )
        projected_sample_count += shard_projected_sample_count

        audit_offsets = [
            offset
            for offset, sample_index in enumerate(base_indices)
            if present[offset] and bool(sample_lookup[sample_index]["full_logit_audit"])
        ]
        full_log_probs = []
        for offset in audit_offsets:
            logits = torch.mv(
                head, hidden[offset].to(device=device, dtype=head.dtype)
            ).float()
            full_log_probs.append(torch.log_softmax(logits, dim=0).to(torch.bfloat16).cpu())
        payload = {
            "kind": "paper2_phase2_stage0a_union_score_shard",
            "score_schema_version": UNION_SCORE_SCHEMA_VERSION,
            "model_key": model_key,
            "row_start": start,
            "row_stop": stop,
            "union_sha256": union_sha256,
            "sample_indices": union["sample_indices"],
            "present": present,
            "candidate_log_probs": candidate_log_probs,
            "tail_log_probs": tail_log_probs,
            "entropy": entropy,
            "greedy_token_ids": greedy,
            "audit_sample_indices": torch.tensor(
                [base_indices[offset] for offset in audit_offsets], dtype=torch.long
            ),
            "full_log_probs_bfloat16": (
                torch.stack(full_log_probs)
                if full_log_probs
                else torch.empty((0, head.shape[0]), dtype=torch.bfloat16)
            ),
            "topk_equivalence_max_abs_error": equivalence_error,
            "topk_probability_max_abs_error": probability_error,
            "topk_values_source": "cached_forward_pass_anchor_simplex_reconciled",
            "maximum_mass_overflow": shard_maximum_mass_overflow,
            "maximum_topk_mass_overflow": shard_maximum_topk_mass_overflow,
            "minimum_topk_mass_projection_scale": shard_minimum_topk_projection_scale,
            "topk_projected_sample_count": shard_topk_projected_sample_count,
            "minimum_extra_mass_projection_scale": shard_minimum_projection_scale,
            "projected_sample_count": shard_projected_sample_count,
            "mass_reconciliation": "fixed-topk KL projection of approximate extras",
        }
        atomic_torch_save(payload, destination, staging_dir=staging_dir)
        receipts.append(
            {
                "path": str(destination),
                "sha256": sha256_file(destination),
                "samples_present": int(present.sum()),
                "audit_samples": len(audit_offsets),
                "topk_equivalence_max_abs_error": equivalence_error,
                "topk_probability_max_abs_error": probability_error,
                "maximum_mass_overflow": shard_maximum_mass_overflow,
                "maximum_topk_mass_overflow": shard_maximum_topk_mass_overflow,
                "minimum_topk_mass_projection_scale": shard_minimum_topk_projection_scale,
                "topk_projected_sample_count": shard_topk_projected_sample_count,
                "minimum_extra_mass_projection_scale": shard_minimum_projection_scale,
                "projected_sample_count": shard_projected_sample_count,
            }
        )
        print(f"stage0a_union_score_progress model={model_key} rows={stop}/{len(rows)}", flush=True)
    del head, head_payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    summary = {
        "kind": "paper2_phase2_stage0a_union_scores",
        "status": "complete",
        "model_key": model_key,
        "shards": receipts,
        "samples_present": sum(row["samples_present"] for row in receipts),
        "full_logit_audit_samples": sum(row["audit_samples"] for row in receipts),
        "topk_equivalence_max_abs_error": maximum_topk_equivalence_error,
        "topk_equivalence_tolerance": TOPK_LOG_PROB_EQUIVALENCE_TOLERANCE,
        "topk_probability_max_abs_error": maximum_topk_probability_error,
        "topk_probability_tolerance": TOPK_PROB_EQUIVALENCE_TOLERANCE,
        "topk_equivalence_role": "discarded_pre_anchor_reconstruction_diagnostic",
        "topk_log_reference_exceeded": (
            maximum_topk_equivalence_error > TOPK_LOG_PROB_EQUIVALENCE_TOLERANCE
        ),
        "topk_probability_reference_exceeded": (
            maximum_topk_probability_error > TOPK_PROB_EQUIVALENCE_TOLERANCE
        ),
        "topk_values_source": "cached_forward_pass_anchor_simplex_reconciled",
        "maximum_mass_overflow": maximum_mass_overflow,
        "maximum_topk_mass_overflow": maximum_topk_mass_overflow,
        "minimum_topk_mass_projection_scale": minimum_topk_mass_projection_scale,
        "topk_projected_sample_count": topk_projected_sample_count,
        "mass_projection_max_overflow": SPARSE_MASS_PROJECTION_MAX_OVERFLOW,
        "minimum_extra_mass_projection_scale": minimum_extra_mass_projection_scale,
        "projected_sample_count": projected_sample_count,
        "mass_reconciliation": "fixed-topk KL projection of approximate extras",
        "resume_policy": "completed union score shards are hash-validated and never rescored",
    }
    write_json(score_root / "summary.json", summary)
    return summary


def _scale_coherence(vectors: Sequence[torch.Tensor]) -> float | None:
    if len(vectors) != 3:
        return None
    seven, fourteen, thirty_two = [vector.float() for vector in vectors]
    first = (fourteen - seven)
    second = (thirty_two - fourteen)
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.norm() * second.norm()
    if float(denominator) <= 1e-12:
        return None
    return float(torch.dot(first, second) / denominator)


def _bucket(student: int, seven: int, fourteen: int, thirty_two: int | None) -> str:
    if thirty_two is None:
        if seven == fourteen == student:
            return "A_consensus_easy"
        if seven == fourteen:
            return "B_consensus_challenging"
        return "E_unresolved_without_32b"
    if seven == fourteen == thirty_two == student:
        return "A_consensus_easy"
    if seven == fourteen == thirty_two:
        return "B_consensus_challenging"
    if fourteen == thirty_two and seven != fourteen:
        return "C_14b_32b_convergence"
    if seven == fourteen and thirty_two != fourteen:
        return "D_32b_only_emergence"
    return "E_all_disagree_or_nonmonotonic"


def finalize_lattice(
    *, rows: Sequence[dict[str, Any]], samples: Sequence[dict[str, Any]], private_dir: Path,
    output_summary: Path, staging_dir: Path, manifest_summary: dict[str, Any],
    model_summaries: dict[str, Any], union_score_summaries: dict[str, Any],
    cascade_summary: dict[str, Any]
) -> dict[str, Any]:
    union_root = private_dir / "union"
    scores_root = private_dir / "union_scores"
    final_root = private_dir / "lattice"
    bucket_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    agreements: list[float] = []
    gaps: list[float] = []
    teachabilities: list[float] = []
    coherences: list[float] = []
    audit_candidate_errors: dict[str, list[float]] = defaultdict(list)
    audit_tail_errors: dict[str, list[float]] = defaultdict(list)
    audit_mass_errors: dict[str, list[float]] = defaultdict(list)
    final_receipts = []
    for start in range(0, len(rows), ROWS_PER_SHARD):
        stop = min(len(rows), start + ROWS_PER_SHARD)
        filename = f"rows_{start:06d}_{stop:06d}.pt"
        union = torch.load(union_root / filename, map_location="cpu", weights_only=False)
        scores = {
            key: torch.load(scores_root / key / filename, map_location="cpu", weights_only=False)
            for key in ACTIVE_CONFIG["models"]
        }
        base_lookup = {
            int(sample_index): offset
            for offset, sample_index in enumerate(union["sample_indices"].tolist())
        }
        for model_key, payload in scores.items():
            for audit_row, sample_index in enumerate(
                payload["audit_sample_indices"].tolist()
            ):
                offset = base_lookup[int(sample_index)]
                valid = union["union_mask"][offset]
                candidate_ids = union["union_ids"][offset][valid].long()
                full = payload["full_log_probs_bfloat16"][audit_row].float()
                full = full - torch.logsumexp(full, dim=0)
                expected_candidates = full[candidate_ids]
                actual_candidates = payload["candidate_log_probs"][offset][valid].float()
                audit_candidate_errors[model_key].append(
                    float((actual_candidates - expected_candidates).abs().max())
                )
                expected_tail = (
                    1.0 - expected_candidates.exp().sum()
                ).clamp(min=1e-30, max=1.0)
                actual_tail = payload["tail_log_probs"][offset].float().exp()
                audit_tail_errors[model_key].append(
                    float((actual_tail - expected_tail).abs())
                )
                represented_mass = actual_candidates.exp().sum() + actual_tail
                audit_mass_errors[model_key].append(
                    float((represented_mass - 1.0).abs())
                )
        records = []
        for offset, sample_index in enumerate(union["sample_indices"].tolist()):
            sample = samples[sample_index]
            valid = union["union_mask"][offset]
            distributions: dict[str, torch.Tensor] = {}
            for key, payload in scores.items():
                if not bool(payload["present"][offset]):
                    continue
                candidates = payload["candidate_log_probs"][offset][valid].float()
                distributions[key] = torch.cat(
                    [candidates, payload["tail_log_probs"][offset : offset + 1].float()]
                )
            teachers = [
                distributions[key]
                for key in ("teacher_7b", "teacher_14b", "teacher_32b")
                if key in distributions
            ]
            student_top_ids = set(
                int(value)
                for value in union["topk_ids"]["student_0p5b"][offset].tolist()
                if int(value) >= 0
            )
            union_ids = union["union_ids"][offset][valid].tolist()
            student_mask = torch.tensor(
                [int(value) in student_top_ids for value in union_ids] + [False],
                dtype=torch.bool,
            )
            metric = coarse_lattice_metrics(
                student_log_probs=distributions["student_0p5b"],
                teacher_log_probs=teachers,
                student_topk_mask=student_mask,
            )
            teacher_vectors = [
                distributions[key]
                for key in ("teacher_7b", "teacher_14b", "teacher_32b")
                if key in distributions
            ]
            coherence = _scale_coherence(teacher_vectors)
            greedy = {
                key: int(payload["greedy_token_ids"][offset])
                for key, payload in scores.items()
                if bool(payload["present"][offset])
            }
            bucket = _bucket(
                greedy["student_0p5b"],
                greedy["teacher_7b"],
                greedy["teacher_14b"],
                greedy.get("teacher_32b"),
            )
            record = {
                "sample_index": sample_index,
                "sample_key": sample["sample_key"],
                "row_index": sample["row_index"],
                "stratum": sample["stratum"],
                "horizon": sample["horizon"],
                "prediction_position": sample["prediction_position"],
                "state_position": sample["state_position"],
                "observed_next_token_id": sample["observed_next_token_id"],
                "verifier_available": sample["verifier_available"],
                "verifier_semantics": "observed token only unless an external verifier is present",
                "teacher_32b_present": "teacher_32b" in distributions,
                "bucket": bucket,
                "greedy_token_ids": greedy,
                **metric,
                "scale_coherence_cosine": coherence,
            }
            records.append(record)
            bucket_counts[bucket] += 1
            stratum_counts[str(sample["stratum"])] += 1
            agreements.append(float(metric["normalized_teacher_agreement"]))
            gaps.append(float(metric["student_gap_coarse_kl"]))
            teachabilities.append(float(metric["teachability_student_topk"]))
            if coherence is not None:
                coherences.append(coherence)
        destination = final_root / filename
        payload = {
            "kind": "paper2_phase2_stage0a_lattice_shard",
            "row_start": start,
            "row_stop": stop,
            "sample_indices": union["sample_indices"],
            "union_ids": union["union_ids"],
            "union_mask": union["union_mask"],
            "model_candidate_log_probs": {
                key: value["candidate_log_probs"] for key, value in scores.items()
            },
            "model_tail_log_probs": {
                key: value["tail_log_probs"] for key, value in scores.items()
            },
            "model_entropy": {key: value["entropy"] for key, value in scores.items()},
            "records": records,
        }
        atomic_torch_save(payload, destination, staging_dir=staging_dir)
        final_receipts.append(
            {"path": str(destination), "sha256": sha256_file(destination), "samples": len(records)}
        )
        print(f"stage0a_finalize_progress rows={stop}/{len(rows)}", flush=True)

    fourteen_summary = model_summaries["teacher_14b"]
    summary = {
        "kind": "paper2_phase2_stage0a",
        "status": "complete_development_only",
        "config": ACTIVE_CONFIG,
        "config_sha256": _config_sha256(ACTIVE_CONFIG),
        "manifest": manifest_summary,
        "model_caches": model_summaries,
        "union_scores": union_score_summaries,
        "cascade_32b": cascade_summary,
        "lattice": {
            "samples": sum(row["samples"] for row in final_receipts),
            "top_k": ACTIVE_CONFIG["top_k"],
            "candidate_space": (
                "cached-forward own-model top-k anchor plus approximately rescored "
                "cross-model candidates and simplex-reconciled tail"
            ),
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "stratum_counts": dict(sorted(stratum_counts.items())),
            "normalized_teacher_agreement": _quantiles(agreements),
            "student_gap_coarse_kl": _quantiles(gaps),
            "teachability_student_topk": _quantiles(teachabilities),
            "scale_coherence_cosine": _quantiles(coherences),
            "shards": final_receipts,
        },
        "full_logit_audit": {
            key: {
                "samples": len(audit_candidate_errors[key]),
                "candidate_log_probability_max_abs_error": _quantiles(
                    audit_candidate_errors[key]
                ),
                "tail_probability_abs_error": _quantiles(audit_tail_errors[key]),
                "represented_mass_abs_error": _quantiles(audit_mass_errors[key]),
            }
            for key in ACTIVE_CONFIG["models"]
        },
        "sparse_reconstruction_scope": (
            "own-model top-k values come from the original forward; cross-model union "
            "candidate values are reconstructed from cached bfloat16 hidden states and "
            "LM-head weights; full-logit audit rows quantify reconstruction error"
        ),
        "teacher_states": {
            "model": fourteen_summary["model"],
            "revision": fourteen_summary["revision"],
            "samples": fourteen_summary["samples"],
            "layer_ordinals_one_based": fourteen_summary[
                "teacher_state_layer_ordinals_one_based"
            ],
            "dtype": "bfloat16",
            "raw_states_retained_until_experiment_0a": True,
        },
        "teacher_forward_passes": {
            key: int(value["teacher_forward_passes"]) for key, value in model_summaries.items()
        },
        "evaluation_partition_touched": False,
        "frozen_evaluation_partitions_touched": [],
        "training_started": False,
        "optimizer_steps": 0,
        "do_not_claim": [
            "development-only lattice metrics are confirmatory evidence",
            "teacher agreement is correctness",
            "teacher scale differences are internal reasoning steps",
            "Stage 0A selects the whitening alpha before matched module pilots",
        ],
    }
    write_json(output_summary, summary)
    write_json(final_root / "summary.json", summary)
    return summary


def main() -> int:
    global ACTIVE_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--private_dir", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--staging_dir", default="/content/stage0a_staging")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument(
        "--config_json",
        help="Optional locked config; defaults to the original Stage 0A config.",
    )
    parser.add_argument(
        "--offload_32b",
        action="store_true",
        help="Keep the pinned 32B bf16 pass on CUDA via Accelerate CPU/disk offload.",
    )
    parser.add_argument("--offload_dir")
    args = parser.parse_args()

    if args.config_json:
        ACTIVE_CONFIG = json.loads(Path(args.config_json).read_text(encoding="utf-8"))

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Stage 0A requires an A100-class CUDA runtime")
    private_dir = Path(args.private_dir)
    staging_dir = Path(args.staging_dir)
    private_dir.mkdir(parents=True, exist_ok=True)
    rows, samples, manifest_summary = prepare_sample_manifest(
        data_path=Path(args.data_jsonl),
        private_dir=private_dir,
        config=ACTIVE_CONFIG,
    )
    model_summaries: dict[str, Any] = {}
    for model_key in ("student_0p5b", "teacher_7b", "teacher_14b"):
        model_summaries[model_key] = cache_model_pass(
            model_key=model_key,
            rows=rows,
            samples=samples,
            position_key_sha256=manifest_summary["position_key_sha256"],
            private_dir=private_dir,
            staging_dir=staging_dir,
            allowed_sample_indices=None,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            offload_32b=args.offload_32b,
            offload_dir=Path(args.offload_dir) if args.offload_dir else None,
        )
    cascade_indices, cascade_summary = build_32b_cascade(
        rows=rows, samples=samples, private_dir=private_dir
    )
    model_summaries["teacher_32b"] = cache_model_pass(
        model_key="teacher_32b",
        rows=rows,
        samples=samples,
        position_key_sha256=manifest_summary["position_key_sha256"],
        private_dir=private_dir,
        staging_dir=staging_dir,
        allowed_sample_indices=cascade_indices,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        offload_32b=args.offload_32b,
        offload_dir=Path(args.offload_dir) if args.offload_dir else None,
    )
    build_union_shards(rows=rows, private_dir=private_dir, staging_dir=staging_dir)
    union_score_summaries: dict[str, Any] = {}
    for model_key in ACTIVE_CONFIG["models"]:
        union_score_summaries[model_key] = score_union_for_model(
            model_key=model_key,
            rows=rows,
            samples=samples,
            private_dir=private_dir,
            staging_dir=staging_dir,
            device=args.device,
        )
    summary = finalize_lattice(
        rows=rows,
        samples=samples,
        private_dir=private_dir,
        output_summary=Path(args.output_summary),
        staging_dir=staging_dir,
        manifest_summary=manifest_summary,
        model_summaries=model_summaries,
        union_score_summaries=union_score_summaries,
        cascade_summary=cascade_summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
