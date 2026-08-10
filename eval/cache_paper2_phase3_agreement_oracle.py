"""Cache strict agreement directions and deployable Phase 3 forecast features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from eval.cache_paper2_phase2_stage0a import _load_flat_shard
from eval.eval_paper2_phase3_p32_coverage import _resolve, _samples, sha256_file
from models.paper2_dc2_student import Phase3StudentModules
from training.paper2_phase3_migration import tensor_state_digest
from training.paper2_phase3_p32 import (
    batched_oracle_directions,
    oracle_batch_equivalence,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_strict_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if bool(row["flip_candidate_14b"]) and bool(row["cross_scale_consistent"]):
                records.append(row)
    if not records:
        raise RuntimeError("P3.2 coverage index has no strict concurrent flip positions")
    ids = [str(row["record_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("strict P3.2 oracle record ids are not unique")
    return records


def _student_receipts(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(summary["model_caches"]["student_0p5b"]["shards"])


def _lm_head(
    sources: Mapping[str, tuple[Path, Path]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    weights = []
    receipts = []
    for source, (summary_path, private_root) in sources.items():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        receipt = summary["model_caches"]["student_0p5b"]["lm_head"]
        path = _resolve(receipt["path"], private_root)
        observed = sha256_file(path)
        if observed != receipt["sha256"]:
            raise RuntimeError(f"P3.2 student LM-head hash mismatch: {source}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        weights.append(payload["weight_bfloat16"].contiguous())
        receipts.append(
            {
                "source": source,
                "path": str(path),
                "sha256": observed,
                "revision": payload["revision"],
                "shape": list(payload["weight_bfloat16"].shape),
            }
        )
    if not torch.equal(weights[0], weights[1]):
        raise RuntimeError("old and new P3.2 populations use different student LM heads")
    return weights[0], {
        "sources": receipts,
        "cross_population_bit_exact": True,
    }


def load_selected_anchor_hidden(
    *,
    records: list[dict[str, Any]],
    sources: Mapping[str, tuple[Path, Path]],
) -> tuple[torch.Tensor, torch.Tensor, dict[tuple[str, int], int], dict[str, Any]]:
    anchor_keys = sorted({(str(row["source"]), int(row["anchor_index"])) for row in records})
    anchor_lookup = {key: index for index, key in enumerate(anchor_keys)}
    hidden = torch.empty((len(anchor_keys), 4, 896), dtype=torch.bfloat16)
    seen = torch.zeros((len(anchor_keys), 4), dtype=torch.bool)
    source_receipts = []
    for source, (summary_path, private_root) in sources.items():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        samples = _samples(private_root)
        needed = {
            anchor for record_source, anchor in anchor_keys if record_source == source
        }
        files = []
        for receipt in _student_receipts(summary):
            path = _resolve(receipt["path"], private_root)
            observed = sha256_file(path)
            if observed != receipt["sha256"]:
                raise RuntimeError(f"P3.2 student hidden shard hash mismatch: {path}")
            shard = _load_flat_shard(path)
            for offset, sample_index in enumerate(shard["sample_indices"].tolist()):
                sample = samples[sample_index]
                anchor = int(sample["anchor_index"])
                if anchor not in needed:
                    continue
                destination = anchor_lookup[(source, anchor)]
                horizon = int(sample["horizon"]) - 1
                hidden[destination, horizon] = shard["final_hidden_bfloat16"][offset]
                seen[destination, horizon] = True
            files.append({"path": str(path), "sha256": observed})
        source_receipts.append(
            {
                "source": source,
                "summary_path": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "student_hidden_files_sha256": hashlib.sha256(
                    json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "selected_anchors": len(needed),
            }
        )
    if not bool(seen.all()):
        missing = torch.where(~seen)
        raise RuntimeError(
            "P3.2 selected anchor hidden cache is incomplete: "
            f"missing_cells={int(missing[0].numel())}"
        )
    record_anchor = torch.tensor(
        [anchor_lookup[(str(row["source"]), int(row["anchor_index"]))] for row in records],
        dtype=torch.long,
    )
    return hidden, record_anchor, anchor_lookup, {
        "selected_anchors": len(anchor_keys),
        "four_horizons_complete": True,
        "sources": source_receipts,
    }


def analytic_oracle_directions(
    *, lm_head_weight: torch.Tensor, source_tokens: torch.Tensor, target_tokens: torch.Tensor
) -> torch.Tensor:
    source = lm_head_weight.index_select(0, source_tokens.long()).float()
    target = lm_head_weight.index_select(0, target_tokens.long()).float()
    gradients = target - source
    return gradients / gradients.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def actual_head_equivalence(
    *,
    hidden: torch.Tensor,
    source_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    lm_head_weight: torch.Tensor,
    rows: int = 8,
) -> dict[str, Any]:
    count = min(int(rows), int(hidden.shape[0]))
    states = hidden[:count].float().unsqueeze(1)
    source = source_tokens[:count].long()
    target = target_tokens[:count].long()
    positions = torch.zeros(count, dtype=torch.long)
    weight = lm_head_weight.float()

    def forward(values: torch.Tensor) -> torch.Tensor:
        return values @ weight.T

    batched = batched_oracle_directions(
        insertion_states=states,
        forward_from_insertion=forward,
        prediction_positions=positions,
        source_tokens=source,
        target_tokens=target,
    )
    analytic = analytic_oracle_directions(
        lm_head_weight=weight,
        source_tokens=source,
        target_tokens=target,
    )
    single = oracle_batch_equivalence(
        insertion_states=states,
        forward_from_insertion=forward,
        prediction_positions=positions,
        source_tokens=source,
        target_tokens=target,
    )
    maximum = float((batched.directions - analytic).abs().max())
    return {
        "kind": "paper2_phase3_actual_lm_head_oracle_equivalence_v1",
        "rows": count,
        "optimized_vs_autograd_max_abs_difference": maximum,
        "batched_vs_single": single,
        "passed": bool(maximum <= 1e-6 and single["maximum_direction_difference"] <= 1e-6),
    }


def _load_phase3_module(
    *, checkpoint: Path, embedding_weight: torch.Tensor, device: str
) -> tuple[Phase3StudentModules, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("trainable_state")
    if not isinstance(state, Mapping):
        raise RuntimeError("migrated Phase 3 checkpoint lacks trainable_state")
    embedding = nn.Embedding.from_pretrained(embedding_weight.float(), freeze=True)
    module = Phase3StudentModules(
        tied_embedding=embedding,
        hidden_size=896,
        rms_cap=float(payload["migration_receipt"]["state_rms_cap"]),
    ).float()
    trainable = {name: value for name, value in module.named_parameters() if value.requires_grad}
    if set(trainable) != set(state):
        raise RuntimeError("migrated Phase 3 trainable-state schema changed")
    with torch.no_grad():
        for name, value in trainable.items():
            value.copy_(state[name].to(dtype=value.dtype))
    module.to(device).eval()
    return module, {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "source_seed": int(payload["source_seed"]),
        "trainable_state_sha256": tensor_state_digest(state),
        "optimizer_state_absent": payload.get("optimizer_state") is None,
    }


@torch.inference_mode()
def scratch_states(
    module: Phase3StudentModules,
    hidden: torch.Tensor,
    *,
    steps: int,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    output = torch.empty((hidden.shape[0], 8, 128), dtype=torch.bfloat16)
    for start in range(0, hidden.shape[0], int(batch_size)):
        stop = min(hidden.shape[0], start + int(batch_size))
        hidden4 = hidden[start:stop].to(device=device, dtype=torch.float32)
        dummy = torch.zeros_like(hidden4[:, :1])
        values = torch.cat([dummy, hidden4], dim=1)
        attention = torch.ones(values.shape[:2], dtype=torch.bool, device=device)
        attention[:, 0] = False
        initial = module.initializer(values, attention)
        context = values.float().mean(dim=1)
        flow = module.flow(initial, context, steps=int(steps))
        output[start:stop] = flow.state.to(torch.bfloat16).cpu()
        if start == 0 or stop == hidden.shape[0] or stop % (int(batch_size) * 64) == 0:
            print(
                f"phase3_oracle_scratch_progress steps={steps} anchors={stop}/{hidden.shape[0]}",
                flush=True,
            )
    return output


def build_cache(
    *,
    coverage_index: Path,
    sources: Mapping[str, tuple[Path, Path]],
    migrated_checkpoints: Iterable[Path],
    output_dir: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    records = read_strict_records(coverage_index)
    lm_head, lm_head_receipt = _lm_head(sources)
    hidden, record_anchor, _anchor_lookup, hidden_receipt = load_selected_anchor_hidden(
        records=records, sources=sources
    )
    record_horizon = torch.tensor([int(row["horizon"]) - 1 for row in records], dtype=torch.long)
    source_tokens = torch.tensor([int(row["student_top1"]) for row in records], dtype=torch.long)
    target_tokens = torch.tensor([int(row["teacher_14b_top1"]) for row in records], dtype=torch.long)
    selected_hidden = hidden[record_anchor, record_horizon]
    directions = analytic_oracle_directions(
        lm_head_weight=lm_head,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
    )
    equivalence = actual_head_equivalence(
        hidden=selected_hidden,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        lm_head_weight=lm_head,
    )
    if not equivalence["passed"]:
        raise RuntimeError(f"actual LM-head oracle equivalence failed: {equivalence}")

    output_dir.mkdir(parents=True, exist_ok=True)
    direction_path = output_dir / "agreement_oracle_directions.pt"
    direction_receipt_path = direction_path.with_suffix(".receipt.json")
    coverage_sha256 = sha256_file(coverage_index)
    direction_payload = {
        "kind": "paper2_phase3_agreement_oracle_direction_cache_v1",
        "status": "complete_strict_concurrence_threshold_neutral",
        "record_ids": [str(row["record_id"]) for row in records],
        "documents": [str(row["document_id"]) for row in records],
        "directions": directions.to(torch.bfloat16),
        "teachability": torch.tensor([float(row["teachability"]) for row in records]),
        "horizons": record_horizon + 1,
        "sources": [str(row["source"]) for row in records],
        "strata": [str(row["stratum"]) for row in records],
        "source_tokens": source_tokens.to(torch.int32),
        "target_tokens": target_tokens.to(torch.int32),
        "coverage_index_sha256": coverage_sha256,
        "lm_head": lm_head_receipt,
        "actual_head_equivalence": equivalence,
        "optimizer_steps": 0,
        "p33_training_authorized": False,
    }
    reuse_direction = False
    if direction_path.is_file() and direction_receipt_path.is_file():
        existing_direction = json.loads(
            direction_receipt_path.read_text(encoding="utf-8")
        )
        reuse_direction = bool(
            existing_direction.get("sha256") == sha256_file(direction_path)
            and existing_direction.get("coverage_index_sha256") == coverage_sha256
            and int(existing_direction.get("rows", -1)) == len(records)
        )
    if reuse_direction:
        print(f"phase3_oracle_direction_resume path={direction_path}", flush=True)
    else:
        atomic_torch_save(direction_payload, direction_path)
        write_json(
            direction_receipt_path,
            {
                "path": str(direction_path),
                "sha256": sha256_file(direction_path),
                "coverage_index_sha256": coverage_sha256,
                "rows": len(records),
                "direction_dimension": 896,
            },
        )

    feature_receipts = []
    checkpoint_receipts = []
    for checkpoint in migrated_checkpoints:
        module, checkpoint_receipt = _load_phase3_module(
            checkpoint=checkpoint, embedding_weight=lm_head, device=device
        )
        checkpoint_receipts.append(checkpoint_receipt)
        seed = int(checkpoint_receipt["source_seed"])
        for steps in (1, 2, 3, 4):
            path = output_dir / f"agreement_features_seed_{seed}_loop_{steps}.pt"
            receipt_path = path.with_suffix(".receipt.json")
            direction_sha256 = sha256_file(direction_path)
            if path.is_file() and receipt_path.is_file():
                existing = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (
                    existing.get("sha256") == sha256_file(path)
                    and existing.get("checkpoint_sha256") == checkpoint_receipt["sha256"]
                    and existing.get("direction_cache_sha256") == direction_sha256
                    and int(existing.get("rows", -1)) == len(records)
                    and int(existing.get("feature_dimension", -1)) == 1920
                ):
                    print(
                        f"phase3_oracle_feature_resume seed={seed} loop={steps} path={path}",
                        flush=True,
                    )
                    feature_receipts.append(existing)
                    continue
            scratch = scratch_states(
                module,
                hidden,
                steps=steps,
                device=device,
                batch_size=batch_size,
            )
            selected_scratch = scratch.index_select(0, record_anchor).flatten(1)
            features = torch.cat([selected_hidden, selected_scratch], dim=1).to(torch.bfloat16)
            atomic_torch_save(
                {
                    "kind": "paper2_phase3_agreement_forecast_feature_cache_v1",
                    "record_ids": direction_payload["record_ids"],
                    "features": features,
                    "feature_definition": "concat(h_p_896,flatten(S_8x128))",
                    "source_seed": seed,
                    "loop_index": steps,
                    "checkpoint_sha256": checkpoint_receipt["sha256"],
                    "direction_cache_sha256": direction_sha256,
                    "optimizer_steps": 0,
                    "p33_training_authorized": False,
                },
                path,
            )
            feature_receipt = {
                "seed": seed,
                "loop_index": steps,
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": int(features.shape[0]),
                "feature_dimension": int(features.shape[1]),
                "checkpoint_sha256": checkpoint_receipt["sha256"],
                "direction_cache_sha256": direction_sha256,
            }
            write_json(receipt_path, feature_receipt)
            feature_receipts.append(feature_receipt)
            del scratch, selected_scratch, features
        del module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "kind": "paper2_phase3_agreement_oracle_cache_summary_v1",
        "status": "complete_agreement_oracle_and_forecast_features_no_training",
        "strict_concurrent_positions": len(records),
        "selected_anchors": hidden_receipt["selected_anchors"],
        "teachability_threshold_selected": False,
        "direction_cache": {
            "path": str(direction_path),
            "sha256": sha256_file(direction_path),
            "rows": len(records),
            "direction_dimension": 896,
        },
        "feature_caches": feature_receipts,
        "checkpoint_sources": checkpoint_receipts,
        "hidden_sources": hidden_receipt,
        "actual_head_equivalence": equivalence,
        "assertions": {
            "strict_14b_32b_concurrence_only": all(
                bool(row["cross_scale_consistent"]) for row in records
            ),
            "teacher_student_disagreements_only": all(
                bool(row["flip_candidate_14b"]) for row in records
            ),
            "numeric_teachability_threshold_not_selected": True,
            "actual_lm_head_equivalence": bool(equivalence["passed"]),
            "both_seed_lineages_present": {row["source_seed"] for row in checkpoint_receipts}
            == {0, 1},
            "all_four_loops_cached_per_seed": len(feature_receipts) == 8,
            "optimizer_absent": all(row["optimizer_state_absent"] for row in checkpoint_receipts),
            "training_steps_zero": True,
            "confirm_unscored": True,
        },
        "p33_training_authorized": False,
        "optimizer_steps": 0,
    }
    failed = [name for name, passed in summary["assertions"].items() if not passed]
    if failed:
        raise RuntimeError(f"P3.2 oracle cache assertions failed: {failed}")
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage_index", type=Path, required=True)
    parser.add_argument("--old_summary", type=Path, required=True)
    parser.add_argument("--old_private", type=Path, required=True)
    parser.add_argument("--new_summary", type=Path, required=True)
    parser.add_argument("--new_private", type=Path, required=True)
    parser.add_argument("--migrated_checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    result = build_cache(
        coverage_index=args.coverage_index,
        sources={
            "old": (args.old_summary, args.old_private),
            "new": (args.new_summary, args.new_private),
        },
        migrated_checkpoints=args.migrated_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
