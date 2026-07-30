"""Read-only DC1 residual-stream scale-response probe on frozen DEV-C."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import load_drafter, read_jsonl
from eval.eval_paper2_dc0_depth_by_append import (
    group_batches,
    parameter_fingerprint,
    transition_counts,
)
from eval.eval_paper2_dc1_preflight import select_probe_indices
from eval.eval_speculative_depth_d0_floor import load_partition_cache
from models.coconut_composite import CoconutRecurrentQwen
from training.paper2_dc1 import PREFLIGHT_POSITION_BUDGET, scale_interpolation_schedule
from training.paper2_dc1_followups import scale_response_reading, summarize_values
from training.speculative_depth_d0_corpus import sha256_file


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


@torch.inference_mode()
def k0_reference_batches(
    composite: CoconutRecurrentQwen,
    rows: list[dict[str, Any]],
    *,
    vocab_size: int,
    device: str,
    batch_size: int,
    resume_dir: Path,
) -> dict[int, dict[str, torch.Tensor]]:
    outputs: dict[int, dict[str, torch.Tensor]] = {}
    batches = group_batches(rows, batch_size)
    resume_dir.mkdir(parents=True, exist_ok=True)
    for number, indices in enumerate(batches, start=1):
        path = resume_dir / f"batch_{number:06d}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
        else:
            values = torch.tensor([rows[index]["input_ids"] for index in indices], device=device)
            result = composite.depth_by_append(
                input_ids=values,
                append_steps=0,
                feedback_mode="raw",
                prediction_vocab_size=vocab_size,
                capture_state_diagnostics=True,
            )
            if result.position_hidden_states is None:
                raise RuntimeError("k0 scale-response reference did not expose hidden states")
            payload = {
                "indices": indices,
                "predictions": result.predictions,
                "position_hidden_states": result.position_hidden_states,
            }
            temporary = path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, path)
        for local, row_index in enumerate(payload["indices"]):
            outputs[int(row_index)] = {
                "predictions": payload["predictions"][local],
                "position_hidden_states": payload["position_hidden_states"][local],
            }
        if number == 1 or number % 16 == 0 or number == len(batches):
            print(f"dc1_scale_k0 batches={number}/{len(batches)}", flush=True)
    if len(outputs) != len(rows):
        raise RuntimeError("scale-response k0 reference left rows missing")
    return outputs


@torch.inference_mode()
def evaluate_scale(
    composite: CoconutRecurrentQwen,
    rows: list[dict[str, Any]],
    references: dict[int, dict[str, torch.Tensor]],
    *,
    spec: dict[str, Any],
    embedding_rms: float,
    raw_rms: float,
    neutral_token_id: int,
    vocab_size: int,
    device: str,
    batch_size: int,
    resume_dir: Path,
) -> dict[str, torch.Tensor]:
    predictions: list[torch.Tensor | None] = [None] * len(rows)
    cos_fed: list[torch.Tensor | None] = [None] * len(rows)
    cos_k0: list[torch.Tensor | None] = [None] * len(rows)
    layer_cosines: list[torch.Tensor | None] = [None] * len(rows)
    batches = group_batches(rows, batch_size)
    destination = resume_dir / str(spec["label"])
    destination.mkdir(parents=True, exist_ok=True)
    capture_layerwise = str(spec["label"]) in {"10x", "raw"}
    for number, indices in enumerate(batches, start=1):
        path = destination / f"batch_{number:06d}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("spec") != spec:
                raise RuntimeError(f"scale-response resume spec mismatch: {path}")
        else:
            values = torch.tensor([rows[index]["input_ids"] for index in indices], device=device)
            reference = torch.stack(
                [references[index]["position_hidden_states"] for index in indices]
            )
            mode = str(spec["feedback_mode"])
            reference_rms = (
                raw_rms
                if spec.get("reference_basis") == "raw"
                else embedding_rms
                if mode == "scaled"
                else None
            )
            result = composite.depth_by_append(
                input_ids=values,
                append_steps=1,
                feedback_mode=mode,
                reference_rms=reference_rms,
                feedback_scale=spec.get("feedback_scale"),
                neutral_token_id=neutral_token_id,
                prediction_vocab_size=vocab_size,
                capture_state_diagnostics=True,
                reference_position_states=reference,
                capture_layerwise_state_diagnostics=capture_layerwise,
            )
            if result.slot_cosine_to_fed is None or result.slot_cosine_to_k0 is None:
                raise RuntimeError("scale-response arm did not expose state cosines")
            payload = {
                "indices": indices,
                "spec": spec,
                "predictions": result.predictions,
                "cosine_to_fed": result.slot_cosine_to_fed,
                "cosine_to_k0": result.slot_cosine_to_k0,
                "layer_cosine_to_fed": result.slot_layer_cosine_to_fed,
            }
            temporary = path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, path)
        for local, row_index in enumerate(payload["indices"]):
            predictions[int(row_index)] = payload["predictions"][local]
            cos_fed[int(row_index)] = payload["cosine_to_fed"][local]
            cos_k0[int(row_index)] = payload["cosine_to_k0"][local]
            if payload.get("layer_cosine_to_fed") is not None:
                layer_cosines[int(row_index)] = payload["layer_cosine_to_fed"][local]
        if number == 1 or number % 16 == 0 or number == len(batches):
            print(
                f"dc1_scale_progress label={spec['label']} batches={number}/{len(batches)}",
                flush=True,
            )
    if any(value is None for value in predictions + cos_fed + cos_k0):
        raise RuntimeError(f"scale-response arm {spec['label']} left rows missing")
    result = {
        "predictions": torch.cat([value for value in predictions if value is not None], dim=0),
        "cosine_to_fed": torch.cat([value for value in cos_fed if value is not None], dim=0),
        "cosine_to_k0": torch.cat([value for value in cos_k0 if value is not None], dim=0),
    }
    available_layers = [value for value in layer_cosines if value is not None]
    if available_layers:
        result["layer_cosine_to_fed"] = torch.cat(available_layers, dim=0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--teacher_cache_summary", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected_checkpoint_sha256", required=True)
    parser.add_argument("--preflight_summary", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--private_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("scale-response checkpoint SHA mismatch")
    rows = read_jsonl(args.data_jsonl)
    selected_indices = select_probe_indices(
        rows, position_budget=PREFLIGHT_POSITION_BUDGET
    )
    selected_rows = [rows[index] for index in selected_indices]
    teacher_summary = json.loads(Path(args.teacher_cache_summary).read_text(encoding="utf-8"))
    teacher_rows = load_partition_cache(teacher_summary, "teacher_7b", "dev_c")
    targets = torch.cat(
        [teacher_rows[index]["teacher_greedy_token_id"].long() for index in selected_indices]
    )

    _tokenizer, wrapper, resize, _original_vocab = load_drafter(
        checkpoint=Path(args.checkpoint),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    wrapper.eval()
    composite = CoconutRecurrentQwen(
        wrapper, latent_token_id=int(resize.control_token_ids[2])
    ).eval()
    before = parameter_fingerprint(composite)
    private = Path(args.private_dir)

    preflight = json.loads(Path(args.preflight_summary).read_text(encoding="utf-8"))
    interpolation = preflight["scale_interpolation"]
    embedding_rms = float(interpolation["embedding_rms"])
    raw_rms = float(interpolation["raw_hidden_rms"])
    references = k0_reference_batches(
        composite,
        selected_rows,
        vocab_size=resize.original_tokenizer_size,
        device=args.device,
        batch_size=args.batch_size,
        resume_dir=private / "k0_reference",
    )
    schedule = scale_interpolation_schedule(embedding_rms=embedding_rms, raw_rms=raw_rms)
    schedule.extend(
        [
            {
                "label": f"raw_{str(multiplier).replace('.', 'p')}x",
                "feedback_mode": "scaled",
                "feedback_scale": float(multiplier),
                "reference_basis": "raw",
                "target_rms": raw_rms * float(multiplier),
            }
            for multiplier in (1.5, 2.0)
        ]
    )
    schedule.sort(key=lambda row: float(row["target_rms"]))
    baseline = torch.cat([references[index]["predictions"][:, 0] for index in range(len(selected_rows))])
    scale_rows = []
    for spec in schedule:
        values = evaluate_scale(
            composite,
            selected_rows,
            references,
            spec=spec,
            embedding_rms=embedding_rms,
            raw_rms=raw_rms,
            neutral_token_id=int(resize.control_token_ids[2]),
            vocab_size=resize.original_tokenizer_size,
            device=args.device,
            batch_size=args.batch_size,
            resume_dir=private / "scale_batches",
        )
        after = values["predictions"][:, 1]
        row = {
            **spec,
            "target_to_raw_rms_ratio": float(spec["target_rms"]) / raw_rms,
            "transition": transition_counts(baseline, after, targets),
            "cosine_to_fed": summarize_values(values["cosine_to_fed"]),
            "cosine_to_k0": summarize_values(values["cosine_to_k0"]),
        }
        if "layer_cosine_to_fed" in values:
            layer_values = values["layer_cosine_to_fed"].float().reshape(
                -1, values["layer_cosine_to_fed"].shape[-1]
            )
            row["layerwise_cosine_to_fed"] = [
                {"layer_state_index": index, **summarize_values(layer_values[:, index])}
                for index in range(layer_values.shape[1])
            ]
        scale_rows.append(row)

    after_fingerprint = parameter_fingerprint(composite)
    if before != after_fingerprint:
        raise RuntimeError("read-only scale-response probe mutated the checkpoint")
    summary = {
        "kind": "paper2_dc1_scale_response_probe",
        "status": "complete_descriptive_non_gating",
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "checkpoint_mutated": False,
        "population": {
            "partition": "DEV-C reusable development surface",
            "rows": len(selected_rows),
            "positions": len(targets),
            "data_jsonl_sha256": sha256_file(args.data_jsonl),
            "teacher_cache_summary_sha256": sha256_file(args.teacher_cache_summary),
            "banked_preflight_summary_sha256": sha256_file(args.preflight_summary),
            "evaluation_c_touched": False,
        },
        "embedding_rms": embedding_rms,
        "raw_hidden_rms": raw_rms,
        "rows": scale_rows,
        "reading": scale_response_reading(scale_rows),
        "mechanism_language_authorized": False,
        "training_started": False,
        "optimizer_steps": 0,
        "evaluation_c_touched": False,
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
