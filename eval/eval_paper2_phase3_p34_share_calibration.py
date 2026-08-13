"""Measure and solve the ratified P3.4 step-zero loss-share contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from eval.cache_paper2_phase2_stage0a import _load_flat_shard
from eval.eval_paper2_phase3_p34_task_trajectory import load_condition
from training.paper2_phase3_p32 import GateLabel
from training.paper2_phase3_p34 import (
    P34_LOSS_NAMES,
    P34_SLOT_LOSS_NAMES,
    SlotSupervisionLift,
    loss_gradient_bundle,
    p34_forward_losses,
    postclip_gradient_norms_from_bundle,
    set_p34_trainable,
    slot_supervision_loss,
    solve_static_loss_weights_from_bundles,
)
from training.run_paper2_phase2_matched_alpha import _local_source, _parallel_receipts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_jsonl_sha256(path: Path) -> str:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _grow_last(value: torch.Tensor, width: int, fill: int | float | bool) -> torch.Tensor:
    if width <= value.shape[-1]:
        return value
    output = torch.full((*value.shape[:-1], width), fill, dtype=value.dtype)
    output[..., : value.shape[-1]] = value
    return output


def _selected_source_cache(
    *,
    source: str,
    records: Sequence[Mapping[str, Any]],
    summary_path: Path,
    private_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_path = private_root / "sample_manifest.jsonl"
    manifest = read_jsonl(manifest_path)
    needed_anchors = sorted(
        {int(row["anchor_index"]) for row in records if str(row["source"]) == source}
    )
    if not needed_anchors:
        raise RuntimeError(f"P3.4 calibration has no {source} anchors")
    anchor_lookup = {anchor: index for index, anchor in enumerate(needed_anchors)}
    sample_anchor = torch.tensor([int(row["anchor_index"]) for row in manifest])
    sample_horizon = torch.tensor([int(row["horizon"]) for row in manifest])
    positions = torch.zeros(len(needed_anchors), dtype=torch.long)
    for row in manifest:
        anchor = int(row["anchor_index"])
        if anchor in anchor_lookup and int(row["horizon"]) == 1:
            positions[anchor_lookup[anchor]] = int(row["prediction_position"])

    lattice_receipts = list(summary["lattice"]["shards"])
    student_receipts = _parallel_receipts(summary, "student_0p5b")
    if len(lattice_receipts) != len(student_receipts):
        raise RuntimeError("P3.4 lattice and student receipt counts differ")
    width = 1
    hidden = torch.empty((len(needed_anchors), 4, 896), dtype=torch.bfloat16)
    candidate_ids = torch.full((len(needed_anchors), 4, width), -1, dtype=torch.int32)
    candidate_mask = torch.zeros((len(needed_anchors), 4, width), dtype=torch.bool)
    base_candidates = torch.full(
        (len(needed_anchors), 4, width), float("-inf"), dtype=torch.bfloat16
    )
    teacher_candidates = torch.full_like(base_candidates, float("-inf"))
    base_tail = torch.empty((len(needed_anchors), 4), dtype=torch.bfloat16)
    teacher_tail = torch.empty((len(needed_anchors), 4), dtype=torch.bfloat16)
    seen = torch.zeros((len(needed_anchors), 4), dtype=torch.bool)
    files = []
    needed = set(needed_anchors)
    for lattice_receipt, student_receipt in zip(lattice_receipts, student_receipts):
        lattice_path = _local_source(lattice_receipt["path"], private_root)
        student_path = _local_source(student_receipt["path"], private_root)
        if not lattice_path.is_file() and not student_path.is_file():
            continue
        if not lattice_path.is_file() or not student_path.is_file():
            raise RuntimeError("P3.4 selected source shard pair is incomplete")
        for path, receipt in ((lattice_path, lattice_receipt), (student_path, student_receipt)):
            observed = sha256_file(path)
            if observed != receipt["sha256"]:
                raise RuntimeError(f"P3.4 selected source shard hash changed: {path}")
            files.append({"path": str(path), "sha256": observed})
        lattice = torch.load(lattice_path, map_location="cpu", weights_only=False)
        student = _load_flat_shard(student_path)
        indices = lattice["sample_indices"].long()
        if not torch.equal(indices, student["sample_indices"].long()):
            raise RuntimeError("P3.4 lattice and student sample indices differ")
        anchors = sample_anchor.index_select(0, indices)
        take = torch.tensor([int(value) in needed for value in anchors], dtype=torch.bool)
        if not bool(take.any()):
            raise RuntimeError("P3.4 staged shard contains no selected anchor")
        indices = indices[take]
        anchors = anchors[take]
        horizons = sample_horizon.index_select(0, indices) - 1
        destinations = torch.tensor([anchor_lookup[int(value)] for value in anchors])
        shard_width = int(lattice["union_ids"].shape[1])
        if shard_width > width:
            candidate_ids = _grow_last(candidate_ids, shard_width, -1)
            candidate_mask = _grow_last(candidate_mask, shard_width, False)
            base_candidates = _grow_last(base_candidates, shard_width, float("-inf"))
            teacher_candidates = _grow_last(
                teacher_candidates, shard_width, float("-inf")
            )
            width = shard_width
        hidden[destinations, horizons] = student["final_hidden_bfloat16"][take]
        candidate_ids[destinations, horizons, :shard_width] = lattice["union_ids"][take].to(
            torch.int32
        )
        candidate_mask[destinations, horizons, :shard_width] = lattice["union_mask"][take]
        base_candidates[destinations, horizons, :shard_width] = lattice[
            "model_candidate_log_probs"
        ]["student_0p5b"][take].to(torch.bfloat16)
        teacher_candidates[destinations, horizons, :shard_width] = lattice[
            "model_candidate_log_probs"
        ]["teacher_14b"][take].to(torch.bfloat16)
        base_tail[destinations, horizons] = lattice["model_tail_log_probs"]["student_0p5b"][
            take
        ].to(torch.bfloat16)
        teacher_tail[destinations, horizons] = lattice["model_tail_log_probs"]["teacher_14b"][
            take
        ].to(torch.bfloat16)
        seen[destinations, horizons] = True
    if not bool(seen.all()):
        raise RuntimeError(
            f"P3.4 selected cache is incomplete for {source}: missing={int((~seen).sum())}"
        )
    payload = {
        "anchors": needed_anchors,
        "anchor_lookup": anchor_lookup,
        "positions": positions,
        "hidden4": hidden,
        "candidate_ids": candidate_ids,
        "candidate_mask": candidate_mask,
        "base_candidates": base_candidates,
        "base_tail": base_tail,
        "teacher_candidates": teacher_candidates,
        "teacher_tail": teacher_tail,
    }
    receipt = {
        "source": source,
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selected_anchors": len(needed_anchors),
        "verified_files": len(files),
        "verified_file_ledger_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    return payload, receipt


def build_selected_batch(
    *,
    records: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    direction_cache: Path,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    width = max(int(source["candidate_ids"].shape[-1]) for source in sources.values())
    direction_payload = torch.load(direction_cache, map_location="cpu", weights_only=False)
    if direction_payload.get("kind") != "paper2_phase3_agreement_oracle_direction_cache_v1":
        raise RuntimeError("P3.4 direction cache kind changed")
    direction_lookup = {
        str(record_id): index for index, record_id in enumerate(direction_payload["record_ids"])
    }
    selected: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "hidden4",
            "candidate_ids",
            "candidate_mask",
            "base_candidates",
            "base_tail",
            "teacher_candidates",
            "teacher_tail",
            "position_bucket",
            "gate_labels",
            "oracle_directions",
            "kl_mask",
            "ce_mask",
        )
    }
    from eval.eval_paper2_phase3_retention_step0 import position_buckets

    for row in records:
        source = sources[str(row["source"])]
        anchor = source["anchor_lookup"][int(row["anchor_index"])]
        horizon = int(row["horizon"]) - 1
        selected["hidden4"].append(source["hidden4"][anchor])
        for key, fill in (
            ("candidate_ids", -1),
            ("candidate_mask", False),
            ("base_candidates", float("-inf")),
            ("teacher_candidates", float("-inf")),
        ):
            value = _grow_last(source[key][anchor], width, fill)
            selected[key].append(value)
        selected["base_tail"].append(source["base_tail"][anchor])
        selected["teacher_tail"].append(source["teacher_tail"][anchor])
        selected["position_bucket"].append(position_buckets(source["positions"][anchor : anchor + 1])[0])
        labels = torch.full((4,), int(GateLabel.IGNORED), dtype=torch.long)
        labels[horizon] = int(row["gate_label"])
        selected["gate_labels"].append(labels)
        oracle = torch.zeros((4, 896), dtype=torch.float32)
        if int(row["gate_label"]) == int(GateLabel.POSITIVE):
            record_id = str(row["record_id"])
            if record_id not in direction_lookup:
                raise RuntimeError(f"P3.4 positive row lacks oracle direction: {record_id}")
            oracle[horizon] = direction_payload["directions"][direction_lookup[record_id]].float()
        selected["oracle_directions"].append(oracle)
        kl_mask = torch.zeros(4, dtype=torch.bool)
        kl_mask[horizon] = True
        selected["kl_mask"].append(kl_mask)
        ce_mask = torch.zeros(4, dtype=torch.bool)
        ce_mask[horizon] = int(row["gate_label"]) == int(GateLabel.POSITIVE)
        selected["ce_mask"].append(ce_mask)
    batch = {key: torch.stack(values).to(device) for key, values in selected.items()}
    teacher_logits = batch["teacher_candidates"].masked_fill(~batch["candidate_mask"], float("-inf"))
    teacher_token_index = teacher_logits.argmax(dim=-1)
    teacher_tokens = batch["candidate_ids"].gather(
        -1, teacher_token_index.unsqueeze(-1)
    ).squeeze(-1)
    if bool((teacher_tokens < 0).any()):
        raise RuntimeError("P3.4 selected cache cannot locate teacher greedy tokens")
    batch["teacher_token_index"] = teacher_token_index
    batch["teacher_tokens"] = teacher_tokens
    batch["teacher_mask"] = torch.ones_like(teacher_tokens, dtype=torch.bool)
    return batch, {
        "rows": len(records),
        "direction_cache_sha256": sha256_file(direction_cache),
        "positive_rows": sum(int(row["gate_label"]) == int(GateLabel.POSITIVE) for row in records),
        "negative_rows": sum(int(row["gate_label"]) == int(GateLabel.NEGATIVE) for row in records),
    }


def arm_read(
    *,
    module: Any,
    batch: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    trainable = set_p34_trainable(module)
    parameters = list(trainable.values())
    module.train()
    losses, metrics = p34_forward_losses(
        module=module,
        tied_embedding=module.draft.tied_embedding,
        teacher_candidates=batch["teacher_candidates"],
        teacher_tail=batch["teacher_tail"],
        teacher_token_index=batch["teacher_token_index"],
        kl_mask=batch["kl_mask"],
        ce_mask=batch["ce_mask"],
        hidden4=batch["hidden4"],
        candidate_ids=batch["candidate_ids"],
        candidate_mask=batch["candidate_mask"],
        base_candidates=batch["base_candidates"],
        base_tail=batch["base_tail"],
        gate_labels=batch["gate_labels"],
        oracle_directions=batch["oracle_directions"],
        position_bucket=batch["position_bucket"],
    )
    bundle = loss_gradient_bundle(
        losses=losses,
        module=module,
        parameters=parameters,
    )
    attribution = postclip_gradient_norms_from_bundle(bundle)
    return {
        "slot_arm": False,
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        **attribution,
        "slot_telemetry": None,
    }, bundle, metrics


def extend_main_bundle_with_slot(
    *,
    main_bundle: Mapping[str, Any],
    slot_lift: SlotSupervisionLift,
    slot_loss: torch.Tensor,
) -> dict[str, Any]:
    """Add the zero-init slot gradient without repeating the sidecar forward."""

    if tuple(main_bundle["names"]) != P34_LOSS_NAMES:
        raise RuntimeError("P3.4 slot extension requires a complete main bundle")
    slot_parameters = [parameter for parameter in slot_lift.parameters() if parameter.requires_grad]
    slot_gradients = torch.autograd.grad(slot_loss, slot_parameters, allow_unused=False)
    gradients = {
        name: dict(main_bundle["gradients"][name]) for name in P34_LOSS_NAMES
    }
    gradients["slot"] = {
        id(parameter): gradient.detach()
        for parameter, gradient in zip(slot_parameters, slot_gradients)
    }
    parameter_groups = dict(main_bundle["parameter_groups"])
    parameter_groups.update({id(parameter): ("heads", 1.0) for parameter in slot_parameters})
    return {
        "names": P34_SLOT_LOSS_NAMES,
        "parameter_ids": [*main_bundle["parameter_ids"], *(id(p) for p in slot_parameters)],
        "gradients": gradients,
        "parameter_groups": parameter_groups,
    }


def slot_read_from_main(
    *,
    module: Any,
    batch: Mapping[str, torch.Tensor],
    main_read: Mapping[str, Any],
    main_bundle: Mapping[str, Any],
    flow_states: tuple[torch.Tensor, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    slot_lift = SlotSupervisionLift().to(next(module.parameters()).device)
    slot, telemetry = slot_supervision_loss(
        lift=slot_lift,
        # The zero-initialized lift is the only slot-arm parameter at step zero.
        flow_states=tuple(state.detach() for state in flow_states),
        tied_weight=module.draft.tied_embedding.weight,
        teacher_tokens=batch["teacher_tokens"],
        teacher_mask=batch["teacher_mask"],
    )
    bundle = extend_main_bundle_with_slot(
        main_bundle=main_bundle,
        slot_lift=slot_lift,
        slot_loss=slot,
    )
    attribution = postclip_gradient_norms_from_bundle(bundle)
    return {
        "slot_arm": True,
        "losses": {**dict(main_read["losses"]), "slot": float(slot.detach())},
        **attribution,
        "slot_telemetry": telemetry,
    }, bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--selection_receipt", type=Path, required=True)
    parser.add_argument("--old_summary", type=Path)
    parser.add_argument("--old_private", type=Path)
    parser.add_argument("--new_summary", type=Path)
    parser.add_argument("--new_private", type=Path)
    parser.add_argument("--compact_batch", type=Path)
    parser.add_argument("--compact_batch_sha256")
    parser.add_argument("--lm_head", type=Path, required=True)
    parser.add_argument("--lm_head_sha256", required=True)
    parser.add_argument("--direction_cache", type=Path)
    parser.add_argument("--migrated", type=Path, required=True)
    parser.add_argument("--migrated_sha256", required=True)
    parser.add_argument("--p33", type=Path, required=True)
    parser.add_argument("--p33_sha256", required=True)
    parser.add_argument("--i1", type=Path, required=True)
    parser.add_argument("--i1_sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--main_only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    records = read_jsonl(args.rows)
    selection = json.loads(args.selection_receipt.read_text(encoding="utf-8"))
    if len(records) != 256 or selection["rows"] != 256:
        raise RuntimeError("P3.4 step-zero calibration requires 128 rows per stratum")
    if args.compact_batch is not None:
        if not args.compact_batch_sha256 or sha256_file(args.compact_batch) != args.compact_batch_sha256:
            raise RuntimeError("P3.4 compact calibration batch hash changed")
        compact = torch.load(args.compact_batch, map_location="cpu", weights_only=False)
        if compact.get("kind") != "paper2_phase3_p34_compact_share_batch_v2":
            raise RuntimeError("P3.4 compact calibration batch kind changed")
        if compact["selection_receipt_canonical_sha256"] != canonical_json_sha256(
            args.selection_receipt
        ):
            raise RuntimeError("P3.4 compact batch selection receipt changed")
        if compact["rows_file_canonical_sha256"] != canonical_jsonl_sha256(args.rows):
            raise RuntimeError("P3.4 compact batch rows changed")
        batch = {key: value.to(args.device) for key, value in compact["batch"].items()}
        batch_receipt = dict(compact["batch_receipt"])
        batch_receipt["compact_batch_sha256"] = args.compact_batch_sha256
        source_receipts = list(compact["source_receipts"])
    else:
        legacy = (args.old_summary, args.old_private, args.new_summary, args.new_private)
        if any(value is None for value in legacy):
            raise RuntimeError("P3.4 calibration needs either a compact batch or both source caches")
        sources = {}
        source_receipts = []
        for name, summary, private in (
            ("old", args.old_summary, args.old_private),
            ("new", args.new_summary, args.new_private),
        ):
            sources[name], receipt = _selected_source_cache(
                source=name,
                records=records,
                summary_path=summary,
                private_root=private,
            )
            source_receipts.append(receipt)
        if args.direction_cache is None:
            raise RuntimeError("P3.4 legacy cache path needs the direction cache")
        batch, batch_receipt = build_selected_batch(
            records=records,
            sources=sources,
            direction_cache=args.direction_cache,
            device=args.device,
        )
    if sha256_file(args.lm_head) != args.lm_head_sha256:
        raise RuntimeError("P3.4 tied-head hash changed")
    embedding = torch.load(args.lm_head, map_location="cpu", weights_only=False)
    if isinstance(embedding, Mapping):
        embedding = embedding.get(
            "weight_bfloat16", embedding.get("weight", embedding.get("lm_head"))
        )
    if not isinstance(embedding, torch.Tensor) or embedding.shape != (151665, 896):
        raise RuntimeError("P3.4 tied-head tensor shape changed")
    module, checkpoint_receipts = load_condition(
        embedding_weight=embedding,
        migrated=args.migrated,
        migrated_sha256=args.migrated_sha256,
        p33=args.p33,
        p33_sha256=args.p33_sha256,
        i1=args.i1,
        i1_sha256=args.i1_sha256,
    )

    strata = sorted({str(row["stratum"]) for row in records})
    arm_specs = [("main", False)]
    if not args.main_only:
        arm_specs.append(("slot", True))
    reads: dict[str, dict[str, Any]] = {arm: {} for arm, _slot in arm_specs}
    bundles: dict[str, list[dict[str, Any]]] = {arm: [] for arm, _slot in arm_specs}
    for stratum in strata:
        print(f"p34_share_stratum_start seed={args.seed} stratum={stratum}", flush=True)
        indexes = torch.tensor(
            [index for index, row in enumerate(records) if str(row["stratum"]) == stratum],
            device=args.device,
        )
        stratum_batch = {key: value.index_select(0, indexes) for key, value in batch.items()}
        main_read, main_bundle, metrics = arm_read(module=module, batch=stratum_batch)
        reads["main"][stratum] = main_read
        bundles["main"].append(main_bundle)
        if not args.main_only:
            slot_read, slot_bundle = slot_read_from_main(
                module=module,
                batch=stratum_batch,
                main_read=main_read,
                main_bundle=main_bundle,
                flow_states=metrics["flow_states"],
            )
            reads["slot"][stratum] = slot_read
            bundles["slot"].append(slot_bundle)
        print(f"p34_share_stratum_complete seed={args.seed} stratum={stratum}", flush=True)
    solved = {}
    for arm, slot_arm in arm_specs:
        print(f"p34_share_solve_start seed={args.seed} arm={arm}", flush=True)
        solved[arm] = solve_static_loss_weights_from_bundles(
            bundles[arm], slot_arm=slot_arm
        )
        print(
            f"p34_share_solve_complete seed={args.seed} arm={arm} "
            f"iterations={solved[arm]['iterations']}",
            flush=True,
        )
    result = {
        "kind": "paper2_phase3_p34_step0_share_calibration_v1",
        "status": "complete_read_only_no_optimizer",
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_authorized": False,
        "calibration_seed": args.seed,
        "calibrated_arms": [arm for arm, _slot in arm_specs],
        "selection": selection,
        "selection_receipt_sha256": sha256_file(args.selection_receipt),
        "rows_file_sha256": sha256_file(args.rows),
        "source_receipts": source_receipts,
        "batch_receipt": batch_receipt,
        "checkpoint_receipts": checkpoint_receipts,
        "lm_head_sha256": args.lm_head_sha256,
        "stratum_reads": reads,
        "solved": solved,
        "estimator": (
            "mean across fixed code and general 128-row strata of unit-weight independent "
            "loss gradient norms after the registered combined bridge/head clipping factors"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
