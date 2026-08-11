"""Score the exact P3.3 e2 token-retention panel before any optimizer exists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from eval.cache_paper2_phase3_agreement_oracle import (
    _lm_head,
    _load_phase3_module,
    load_selected_anchor_hidden,
    write_json,
)
from training.paper2_phase3_p33 import activate_operating_clamp
from training.paper2_phase3_p33_prep import P33_GATE_CEILING, sha256_file


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def position_buckets(positions: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(positions)
    output[(positions >= 1) & (positions <= 3)] = 1
    output[(positions >= 4) & (positions <= 31)] = 2
    output[(positions >= 32) & (positions <= 127)] = 3
    output[positions >= 128] = 4
    return output


@torch.inference_mode()
def bridge_hidden(
    module: torch.nn.Module,
    hidden4: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    dummy = torch.zeros_like(hidden4[:, :1])
    hidden = torch.cat([dummy, hidden4], dim=1)
    attention = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
    attention[:, 0] = False
    batch = hidden.shape[0]
    candidate_ids = torch.zeros((batch, 4, 1), dtype=torch.long, device=hidden.device)
    previous_logits = torch.zeros((batch, 4, 1), dtype=hidden.dtype, device=hidden.device)
    output = module(
        hidden=hidden,
        previous_logits=previous_logits,
        steps=1,
        attention_mask=attention,
        position_bucket=position_buckets(positions),
        candidate_ids=candidate_ids,
    )
    return output.hidden[:, 1:]


def score(
    *,
    retention_panel: Path,
    sources: Mapping[str, tuple[Path, Path]],
    migrated_checkpoints: Iterable[Path],
    output_rows: Path,
    output_summary: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    records = read_jsonl(retention_panel)
    if len(records) != 1024:
        raise RuntimeError(f"P3.3 retention panel changed: {len(records)}")
    if {int(row["horizon"]) for row in records} != {1, 2, 3, 4}:
        raise RuntimeError("P3.3 retention panel lost a horizon")
    lm_head, lm_head_receipt = _lm_head(sources)
    hidden, record_anchor, _lookup, hidden_receipt = load_selected_anchor_hidden(
        records=records,
        sources=sources,
    )
    expanded_hidden = hidden.index_select(0, record_anchor)
    horizons = torch.tensor([int(row["horizon"]) - 1 for row in records], dtype=torch.long)
    positions = torch.tensor([int(row["prediction_position"]) for row in records], dtype=torch.long)
    expected_base = torch.tensor([int(row["student_top1"]) for row in records], dtype=torch.long)
    all_rows: list[dict[str, Any]] = []
    checkpoint_receipts = []
    clamp_receipts = []
    for checkpoint in migrated_checkpoints:
        module, checkpoint_receipt = _load_phase3_module(
            checkpoint=checkpoint,
            embedding_weight=lm_head,
            device=device,
        )
        checkpoint_receipts.append(checkpoint_receipt)
        clamp = activate_operating_clamp(module)
        clamp_receipts.append({"seed": checkpoint_receipt["source_seed"], **clamp})
        if float(module.bridge.gate_ceiling) != P33_GATE_CEILING:
            raise RuntimeError("P3.3 operating clamp failed to activate")
        embedding = module.draft.tied_embedding.weight
        seed = int(checkpoint_receipt["source_seed"])
        seed_rows = []
        base_mismatches = 0
        for start in range(0, len(records), batch_size):
            stop = min(len(records), start + batch_size)
            hidden4 = expanded_hidden[start:stop].to(device=device, dtype=torch.float32)
            augmented4 = bridge_hidden(
                module,
                hidden4,
                positions[start:stop].to(device),
            )
            local_horizon = horizons[start:stop].to(device)
            batch_index = torch.arange(stop - start, device=device)
            base_state = hidden4[batch_index, local_horizon]
            augmented_state = augmented4[batch_index, local_horizon]
            base_token = (base_state @ embedding.T).argmax(dim=-1).cpu()
            augmented_token = (augmented_state @ embedding.T).argmax(dim=-1).cpu()
            base_mismatches += int((base_token != expected_base[start:stop]).sum())
            for offset in range(stop - start):
                source = records[start + offset]
                seed_rows.append(
                    {
                        "seed": seed,
                        "record_id": source["record_id"],
                        "horizon": int(source["horizon"]),
                        "prediction_position": int(source["prediction_position"]),
                        "base_top1": int(base_token[offset]),
                        "augmented_top1": int(augmented_token[offset]),
                        "retained": bool(base_token[offset] == augmented_token[offset]),
                    }
                )
            if start == 0 or stop == len(records) or stop % (batch_size * 8) == 0:
                print(
                    f"p33_retention_step0 seed={seed} positions={stop}/{len(records)}",
                    flush=True,
                )
        if base_mismatches:
            raise RuntimeError(
                f"P3.3 retention base-reader mismatch seed={seed}: {base_mismatches}/1024"
            )
        all_rows.extend(seed_rows)
        del module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(output_rows, all_rows)
    by_seed = {
        str(seed): {
            "positions": sum(int(row["seed"]) == seed for row in all_rows),
            "retained": sum(
                int(row["seed"]) == seed and bool(row["retained"]) for row in all_rows
            ),
        }
        for seed in (0, 1)
    }
    for values in by_seed.values():
        values["retention"] = values["retained"] / values["positions"]
    summary = {
        "kind": "paper2_phase3_p33_retention_step0_v1",
        "status": "complete_no_update_exact_estimator",
        "retention_panel": {
            "path": str(retention_panel),
            "sha256": sha256_file(retention_panel),
            "positions": len(records),
        },
        "step0_rows": {
            "path": str(output_rows),
            "sha256": sha256_file(output_rows),
            "rows": len(all_rows),
        },
        "by_seed": by_seed,
        "checkpoints": checkpoint_receipts,
        "operating_clamp": clamp_receipts,
        "lm_head": lm_head_receipt,
        "hidden_sources": hidden_receipt,
        "estimand": "fraction of positions where augmented top1 matches frozen base top1",
        "task_level_capability_scoring": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "assertions": {
            "panel_1024": len(records) == 1024,
            "both_seeds": {row["source_seed"] for row in checkpoint_receipts} == {0, 1},
            "rows_2048": len(all_rows) == 2048,
            "base_reader_exact": True,
            "operating_clamp_0p02": all(
                float(row["ceiling"]) == P33_GATE_CEILING for row in clamp_receipts
            ),
            "task_scoring_absent": True,
            "optimizer_absent": True,
            "training_steps_zero": True,
        },
    }
    failed = [name for name, passed in summary["assertions"].items() if not passed]
    if failed:
        raise RuntimeError(f"P3.3 retention step-zero assertions failed: {failed}")
    write_json(output_summary, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retention_panel", type=Path, required=True)
    parser.add_argument("--old_summary", type=Path, required=True)
    parser.add_argument("--old_private", type=Path, required=True)
    parser.add_argument("--new_summary", type=Path, required=True)
    parser.add_argument("--new_private", type=Path, required=True)
    parser.add_argument("--migrated_checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output_rows", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    result = score(
        retention_panel=args.retention_panel,
        sources={
            "old": (args.old_summary, args.old_private),
            "new": (args.new_summary, args.new_private),
        },
        migrated_checkpoints=args.migrated_checkpoint,
        output_rows=args.output_rows,
        output_summary=args.output_summary,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
