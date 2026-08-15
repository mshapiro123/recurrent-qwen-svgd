"""Refresh P3.4 endpoint oracle reads and quantify source-token staleness."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from eval.cache_paper2_phase3_agreement_oracle import (
    analytic_oracle_directions,
    load_selected_anchor_hidden,
)
from eval.eval_paper2_phase3_p31_references import MODEL_SPECS
from eval.eval_paper2_phase3_p34_task_trajectory import load_condition, sha256_file, write_json
from training.run_paper2_phase3_p33 import (
    _direction_lookup,
    audit_model,
    read_jsonl,
    write_jsonl,
)


def load_oracle_audit_material(
    *, positive_path: Path, negative_path: Path, sources: Mapping[str, tuple[Path, Path]]
) -> dict[str, Any]:
    populations = {
        "positive": read_jsonl(positive_path),
        "negative": read_jsonl(negative_path),
    }
    if (len(populations["positive"]), len(populations["negative"])) != (4096, 12288):
        raise RuntimeError("P3.4 oracle refresh audit population counts changed")
    material = {}
    for name, records in populations.items():
        hidden, record_anchor, _lookup, receipt = load_selected_anchor_hidden(
            records=records, sources=sources
        )
        material[name] = {
            "records": records,
            "hidden4": hidden.index_select(0, record_anchor),
            "hidden_receipt": receipt,
        }
    return material


def direction_refresh_read(
    rows: Sequence[Mapping[str, Any]], lm_head_weight: torch.Tensor
) -> dict[str, Any]:
    positive = [row for row in rows if row["population"] == "positive"]
    base_matches_cache = sum(
        int(row["base_top1"] == row["cached_student_top1"]) for row in positive
    )
    source_changed = [row for row in positive if row["deployed_top1"] != row["base_top1"]]
    target_reached = [row for row in positive if row["deployed_top1"] == row["teacher_top1"]]
    refreshable = [row for row in positive if row["deployed_top1"] != row["teacher_top1"]]
    if refreshable:
        base_source = torch.tensor([row["base_top1"] for row in refreshable])
        deployed_source = torch.tensor([row["deployed_top1"] for row in refreshable])
        target = torch.tensor([row["teacher_top1"] for row in refreshable])
        old = analytic_oracle_directions(
            lm_head_weight=lm_head_weight,
            source_tokens=base_source,
            target_tokens=target,
        )
        refreshed = analytic_oracle_directions(
            lm_head_weight=lm_head_weight,
            source_tokens=deployed_source,
            target_tokens=target,
        )
        cosine = F.cosine_similarity(old, refreshed, dim=-1)
        cosine_read = {
            "rows": len(refreshable),
            "mean": float(cosine.mean()),
            "minimum": float(cosine.min()),
            "fraction_below_0p99": float((cosine < 0.99).float().mean()),
        }
    else:
        cosine_read = {"rows": 0, "mean": None, "minimum": None, "fraction_below_0p99": None}
    return {
        "positive_rows": len(positive),
        "base_reader_matches_cached_source": base_matches_cache,
        "base_reader_match_rate": base_matches_cache / len(positive),
        "deployed_source_changed_rows": len(source_changed),
        "deployed_source_changed_fraction": len(source_changed) / len(positive),
        "deployed_target_already_reached_rows": len(target_reached),
        "deployed_target_already_reached_fraction": len(target_reached) / len(positive),
        "old_vs_deployed_anchored_direction_cosine": cosine_read,
        "registered_estimator_direction_stale": base_matches_cache != len(positive),
        "persistent_serving_reanchor_needed": bool(source_changed),
    }


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old_summary", type=Path, required=True)
    parser.add_argument("--old_private", type=Path, required=True)
    parser.add_argument("--new_summary", type=Path, required=True)
    parser.add_argument("--new_private", type=Path, required=True)
    parser.add_argument("--positive_audit", type=Path, required=True)
    parser.add_argument("--negative_audit", type=Path, required=True)
    parser.add_argument("--direction_cache", type=Path, required=True)
    parser.add_argument("--migrated", type=Path, nargs=2, required=True)
    parser.add_argument("--migrated_sha256", nargs=2, required=True)
    parser.add_argument("--p33", type=Path, nargs=2, required=True)
    parser.add_argument("--p33_sha256", nargs=2, required=True)
    parser.add_argument("--i1", type=Path, nargs=2, required=True)
    parser.add_argument("--i1_sha256", nargs=2, required=True)
    parser.add_argument("--p34", type=Path, nargs=2, required=True)
    parser.add_argument("--p34_sha256", nargs=2, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sources = {
        "old": (args.old_summary, args.old_private),
        "new": (args.new_summary, args.new_private),
    }
    material = load_oracle_audit_material(
        positive_path=args.positive_audit,
        negative_path=args.negative_audit,
        sources=sources,
    )
    direction_index, directions, direction_receipt = _direction_lookup(args.direction_cache)
    spec = MODEL_SPECS["base"]
    base_model = AutoModelForCausalLM.from_pretrained(
        spec["model"],
        revision=spec["revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device).eval()
    lm_head = base_model.get_output_embeddings().weight.detach()
    output = {
        "kind": "paper2_phase3_p34_oracle_direction_refresh_v1",
        "status": "complete_dev_audit_only",
        "direction_cache": direction_receipt,
        "seeds": {},
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in (0, 1):
        module, chain = load_condition(
            embedding_weight=lm_head.cpu(),
            migrated=args.migrated[seed],
            migrated_sha256=args.migrated_sha256[seed],
            p33=args.p33[seed],
            p33_sha256=args.p33_sha256[seed],
            i1=args.i1[seed],
            i1_sha256=args.i1_sha256[seed],
            p34=args.p34[seed],
            p34_sha256=args.p34_sha256[seed],
        )
        campaign = next(item for item in chain if item["label"] == "p34")
        module.bridge.set_gate_ceiling((0.02, 0.08, 0.20, 0.50)[int(campaign["controller_rung"])])
        summary, rows = audit_model(
            module=module,
            material=material,
            direction_index=direction_index,
            directions=directions,
            seed=seed,
            step=4000,
            device=args.device,
        )
        refresh = direction_refresh_read(rows, lm_head)
        row_path = args.output_dir / f"seed_{seed}_audit_rows.jsonl"
        write_jsonl(row_path, rows)
        output["seeds"][str(seed)] = {
            "checkpoint_chain": chain,
            "registered_endpoint_gate_ceiling": float(module.bridge.gate_ceiling),
            "audit": summary,
            "direction_refresh": refresh,
            "rows_sha256": sha256_file(row_path),
        }
        del module
        gc.collect()
        torch.cuda.empty_cache()
    output["interpretation"] = {
        "registered_pi_dir": (
            "the frozen-base source token remains the registered oracle anchor; any mismatch is a fault"
        ),
        "persistent_serving": (
            "when a deployed write changes the source token, a later persistent step should re-anchor its oracle direction"
        ),
    }
    write_json(args.output_dir / "summary.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
