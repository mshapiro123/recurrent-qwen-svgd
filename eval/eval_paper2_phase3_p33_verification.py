"""Verify P3.3 selectivity and re-emit its metrics with the BF16 serving reader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from eval.cache_paper2_phase3_agreement_oracle import _lm_head
from models.paper2_dc2_student import Phase3StudentModules
from training.paper2_phase3_migration import tensor_state_digest
from training.paper2_phase3_p33_prep import sha256_file
from training.paper2_phase3_p33_verification import (
    P33_CANONICAL_READER_NAME,
    P33_GATE_OPEN_THRESHOLD,
    P33_VERIFICATION_AUDIT_RADIUS,
    canonical_logits,
    fixed_pair_margin,
    verification_verdict,
)
from training.run_paper2_phase3_p33 import (
    AUDIT_BATCH_SIZE,
    _direction_lookup,
    _model_components,
    _ratio_bootstrap,
    load_audit_material,
    read_jsonl,
    write_json,
    write_jsonl,
)


RUN_KIND = "paper2_phase3_p33_zero_verification_v1"
EVALUATION_PATH_ID = "p33_model_components_deployed_hidden_bf16_reader_v1"
STATE_RMS_CAP = 0.5508932316303252
EXPECTED_CHECKPOINT_SHA256 = {
    0: "84dc0fb2d1f69114b20888acd95101d6b31c810974a536dc36358b69fe13c70e",
    1: "e80ad205eb3c4712fdee5303a4887260488f67ff858a2b4b005d724675e52067",
}


def _load_final_module(
    *, checkpoint: Path, embedding_weight: torch.Tensor, seed: int, device: str
) -> tuple[Phase3StudentModules, dict[str, Any]]:
    observed = sha256_file(checkpoint)
    if observed != EXPECTED_CHECKPOINT_SHA256[int(seed)]:
        raise RuntimeError(f"P3.3 final checkpoint SHA mismatch for seed {seed}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("kind") != "paper2_phase3_p33_aimed_writeback_seed_v1":
        raise RuntimeError("P3.3 verification requires a final P3.3 checkpoint")
    if int(payload.get("seed", -1)) != int(seed) or int(payload.get("step", -1)) != 1000:
        raise RuntimeError("P3.3 verification checkpoint seed or step mismatch")
    state = payload.get("trainable_state")
    embedding = nn.Embedding.from_pretrained(embedding_weight.float(), freeze=True)
    module = Phase3StudentModules(
        tied_embedding=embedding, hidden_size=896, rms_cap=STATE_RMS_CAP
    ).float()
    target = {name: value for name, value in module.named_parameters() if value.requires_grad}
    if set(target) != set(state):
        raise RuntimeError("P3.3 verification trainable-state schema changed")
    with torch.no_grad():
        for name, value in target.items():
            value.copy_(state[name].to(dtype=value.dtype))
    module.bridge.set_gate_ceiling(0.02)
    module.to(device).eval()
    return module, {
        "path": str(checkpoint),
        "sha256": observed,
        "seed": int(seed),
        "step": 1000,
        "trainable_state_sha256": tensor_state_digest(state),
    }


def _row_metrics(
    *,
    records: list[dict[str, Any]],
    hidden_all: torch.Tensor,
    module: Phase3StudentModules,
    directions: torch.Tensor,
    direction_index: Mapping[str, int],
    population: str,
    seed: int,
    device: str,
) -> list[dict[str, Any]]:
    embedding = module.draft.tied_embedding.weight
    output: list[dict[str, Any]] = []
    for start in range(0, len(records), AUDIT_BATCH_SIZE):
        stop = min(len(records), start + AUDIT_BATCH_SIZE)
        local = records[start:stop]
        hidden4 = hidden_all[start:stop].to(device=device, dtype=torch.float32)
        positions = torch.tensor(
            [int(row["prediction_position"]) for row in local], device=device
        )
        horizons = torch.tensor([int(row["horizon"]) - 1 for row in local], device=device)
        index = torch.arange(len(local), device=device)
        components = _model_components(module, hidden4, positions)
        base = hidden4[index, horizons]
        trained_delta = components["delta"][index, horizons]
        deployed = components["deployed_hidden"][index, horizons]
        gate_unclamped = components["gate_unclamped"][index, horizons]
        gate_deployed = components["gate_deployed"][index, horizons]
        forced_trained = base + P33_VERIFICATION_AUDIT_RADIUS * trained_delta

        states = [base, deployed, forced_trained]
        oracle = None
        if population == "positive":
            oracle = torch.stack(
                [directions[direction_index[str(row["record_id"])]] for row in local]
            ).to(device)
            reference = base.square().mean(dim=-1).sqrt().clamp_max(module.bridge.rms_cap)
            oracle_scaled = oracle / oracle.square().mean(dim=-1).sqrt().clamp_min(1e-8).unsqueeze(-1)
            oracle_scaled = oracle_scaled * reference.unsqueeze(-1)
            states.extend(
                [
                    base + P33_VERIFICATION_AUDIT_RADIUS * oracle_scaled,
                    base + gate_deployed.unsqueeze(-1) * oracle_scaled,
                ]
            )

        logits = canonical_logits(torch.cat(states, dim=0), embedding).float().split(len(local))
        base_logits, deployed_logits, forced_logits = logits[:3]
        top2 = base_logits.topk(2, dim=-1).indices
        winner, runner_up = top2[:, 0], top2[:, 1]
        base_margin = fixed_pair_margin(base_logits, winner, runner_up)
        deployed_margin = fixed_pair_margin(deployed_logits, winner, runner_up)
        base_top1 = base_logits.argmax(dim=-1)
        deployed_top1 = deployed_logits.argmax(dim=-1)
        forced_top1 = forced_logits.argmax(dim=-1)
        hidden_delta = deployed - base

        for offset, source in enumerate(local):
            row: dict[str, Any] = {
                "seed": int(seed),
                "population": population,
                "record_id": str(source["record_id"]),
                "document_id": str(source["document_id"]),
                "horizon": int(source["horizon"]),
                "evaluation_path_id": EVALUATION_PATH_ID,
                "reader": P33_CANONICAL_READER_NAME,
                "base_top1": int(base_top1[offset]),
                "cached_student_top1": int(source["student_top1"]),
                "base_reader_matches_cached_student": (
                    int(base_top1[offset]) == int(source["student_top1"])
                ),
                "deployed_top1": int(deployed_top1[offset]),
                "collateral_change": bool(base_top1[offset] != deployed_top1[offset]),
                "forced_open_top1": int(forced_top1[offset]),
                "forced_open_collateral_change": bool(base_top1[offset] != forced_top1[offset]),
                "base_pair_margin": float(base_margin[offset]),
                "deployed_pair_margin": float(deployed_margin[offset]),
                "margin_delta": float(deployed_margin[offset] - base_margin[offset]),
                "hidden_delta_rms": float(hidden_delta[offset].square().mean().sqrt()),
                "base_hidden_rms": float(base[offset].square().mean().sqrt()),
                "gate_unclamped": float(gate_unclamped[offset]),
                "gate_deployed": float(gate_deployed[offset]),
                "gate_predicted_open": bool(
                    gate_unclamped[offset] >= P33_GATE_OPEN_THRESHOLD
                ),
            }
            if population == "positive":
                assert oracle is not None
                teacher = int(source["teacher_14b_top1"])
                forced_oracle_logits, deployed_oracle_logits = logits[3:5]
                row.update(
                    {
                        "teacher_top1": teacher,
                        "teachability_decile": int(source["teachability_decile"]),
                        "direction_cosine": float(
                            F.cosine_similarity(trained_delta[offset], oracle[offset], dim=0)
                        ),
                        "forced_trained_flip": int(forced_top1[offset]) == teacher,
                        "forced_oracle_flip": (
                            int(forced_oracle_logits[offset].argmax()) == teacher
                        ),
                        "deployed_trained_flip": int(deployed_top1[offset]) == teacher,
                        "deployed_oracle_flip": (
                            int(deployed_oracle_logits[offset].argmax()) == teacher
                        ),
                    }
                )
            output.append(row)
        if start == 0 or stop == len(records) or stop % (AUDIT_BATCH_SIZE * 64) == 0:
            print(
                f"p33_verification_progress seed={seed} population={population} "
                f"rows={stop}/{len(records)}",
                flush=True,
            )
    return output


@torch.inference_mode()
def evaluate_seed(args: argparse.Namespace, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sources = {
        "old": (args.old_summary, args.old_private),
        "new": (args.new_summary, args.new_private),
    }
    lm_head, lm_head_receipt = _lm_head(sources)
    direction_index, directions, direction_receipt = _direction_lookup(args.direction_cache)
    material = load_audit_material(
        positive_path=args.positive_audit,
        negative_path=args.negative_audit,
        retention_path=args.retention_panel,
        sources=sources,
    )
    module, checkpoint = _load_final_module(
        checkpoint=args.checkpoint[seed],
        embedding_weight=lm_head,
        seed=seed,
        device=args.device,
    )
    rows = []
    for population in ("positive", "negative", "retention"):
        rows.extend(
            _row_metrics(
                records=material[population]["records"],
                hidden_all=material[population]["hidden4"],
                module=module,
                directions=directions,
                direction_index=direction_index,
                population=population,
                seed=seed,
                device=args.device,
            )
        )
    positive = [row for row in rows if row["population"] == "positive"]
    negative = [row for row in rows if row["population"] == "negative"]
    retention = [row for row in rows if row["population"] == "retention"]
    if not all(bool(row["base_reader_matches_cached_student"]) for row in rows):
        raise RuntimeError("canonical BF16 reader does not reproduce cached student top1")
    verification = verification_verdict(
        negative_rows=negative,
        retention_rows=retention,
        positive_deployed_flips=sum(bool(row["deployed_trained_flip"]) for row in positive),
    )
    summary = {
        "kind": RUN_KIND,
        "seed": seed,
        "status": "passed" if verification["all_passed"] else "failed_positive_control",
        "reader": {
            "name": P33_CANONICAL_READER_NAME,
            "hidden_dtype": "torch.bfloat16",
            "embedding_dtype": "torch.bfloat16",
            "matmul_dtype": "torch.bfloat16",
            "base_rows_matching_cache": len(rows),
            "base_rows_total": len(rows),
        },
        "pi_dir": _ratio_bootstrap(
            positive,
            numerator="forced_trained_flip",
            denominator="forced_oracle_flip",
            seed=20260812 + seed,
        ),
        "pi_dep": _ratio_bootstrap(
            positive,
            numerator="deployed_trained_flip",
            denominator="deployed_oracle_flip",
            seed=20270812 + seed,
        ),
        "collateral": {
            "changes": sum(bool(row["collateral_change"]) for row in negative),
            "rows": len(negative),
        },
        "retention": {
            "retained": sum(not bool(row["collateral_change"]) for row in retention),
            "rows": len(retention),
        },
        "verification": verification,
        "same_deployed_path_contract": {
            "evaluation_path_id": EVALUATION_PATH_ID,
            "positive_and_negative_rows_share_function": True,
            "positive_deployed_flips": sum(
                bool(row["deployed_trained_flip"]) for row in positive
            ),
            "negative_deployed_flips": sum(bool(row["collateral_change"]) for row in negative),
        },
        "checkpoint": checkpoint,
        "lm_head": lm_head_receipt,
        "direction_cache": direction_receipt,
        "audit_inputs": {
            "positive_sha256": sha256_file(args.positive_audit),
            "negative_sha256": sha256_file(args.negative_audit),
            "retention_sha256": sha256_file(args.retention_panel),
        },
        "parameter_updates": 0,
    }
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old_summary", type=Path, required=True)
    parser.add_argument("--old_private", type=Path, required=True)
    parser.add_argument("--new_summary", type=Path, required=True)
    parser.add_argument("--new_private", type=Path, required=True)
    parser.add_argument("--positive_audit", type=Path, required=True)
    parser.add_argument("--negative_audit", type=Path, required=True)
    parser.add_argument("--retention_panel", type=Path, required=True)
    parser.add_argument("--direction_cache", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.checkpoint) != 2:
        parser.error("exactly two --checkpoint arguments are required in seed order")
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_rows = []
    for seed in (0, 1):
        summary, rows = evaluate_seed(args, seed)
        summaries.append(summary)
        all_rows.extend(rows)
        write_json(args.output_dir / f"seed_{seed}_summary.json", summary)
        write_jsonl(args.output_dir / f"seed_{seed}_rows.jsonl", rows)
    positive = [row for row in all_rows if row["population"] == "positive"]
    combined = {
        "kind": "paper2_phase3_p33_zero_verification_combined_v1",
        "status": (
            "passed" if all(summary["status"] == "passed" for summary in summaries)
            else "failed_positive_control"
        ),
        "reader": P33_CANONICAL_READER_NAME,
        "pi_dir": _ratio_bootstrap(
            positive,
            numerator="forced_trained_flip",
            denominator="forced_oracle_flip",
            seed=20260812,
        ),
        "pi_dep": _ratio_bootstrap(
            positive,
            numerator="deployed_trained_flip",
            denominator="deployed_oracle_flip",
            seed=20270812,
        ),
        "per_seed": summaries,
        "historical_fp32_receipt_retained": True,
        "parameter_updates": 0,
        "p34_authorized": False,
    }
    write_json(args.output_dir / "summary.json", combined)
    print(json.dumps(combined, indent=2, sort_keys=True), flush=True)
    return 0 if combined["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
