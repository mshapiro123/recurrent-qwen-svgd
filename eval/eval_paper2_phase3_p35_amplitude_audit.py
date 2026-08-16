"""Read the registered causal and collateral audit at one fixed P3.5 ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from eval.eval_paper2_phase3_p31_references import MODEL_SPECS
from eval.eval_paper2_phase3_p34_task_trajectory import load_condition, sha256_file
from training.paper2_phase3_p35 import load_p35_direction_lookup
from training.run_paper2_phase3_p33 import audit_model, load_audit_material


AUTHORIZED = (0.02, 0.05, 0.08, 0.11)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--ceiling", type=float, required=True)
    parser.add_argument("--old_summary", type=Path, required=True)
    parser.add_argument("--old_private", type=Path, required=True)
    parser.add_argument("--new_summary", type=Path, required=True)
    parser.add_argument("--new_private", type=Path, required=True)
    parser.add_argument("--positive_audit", type=Path, required=True)
    parser.add_argument("--negative_audit", type=Path, required=True)
    parser.add_argument("--retention_panel", type=Path, required=True)
    parser.add_argument("--direction_cache", type=Path, required=True)
    parser.add_argument("--direction_cache_sha256", required=True)
    parser.add_argument("--migrated", type=Path, required=True)
    parser.add_argument("--migrated_sha256", required=True)
    parser.add_argument("--p33", type=Path, required=True)
    parser.add_argument("--p33_sha256", required=True)
    parser.add_argument("--i1", type=Path, required=True)
    parser.add_argument("--i1_sha256", required=True)
    parser.add_argument("--p34", type=Path, required=True)
    parser.add_argument("--p34_sha256", required=True)
    parser.add_argument("--p35", type=Path, required=True)
    parser.add_argument("--p35_sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if float(args.ceiling) not in AUTHORIZED:
        raise ValueError(f"unauthorized amplitude ceiling: {args.ceiling}")
    for path in vars(args).values():
        if isinstance(path, Path) and any(term in str(path).casefold() for term in ("confirm", "eval_e")):
            raise RuntimeError("sealed partition contact")
    spec = MODEL_SPECS["base"]
    base = AutoModelForCausalLM.from_pretrained(
        spec["model"], revision=spec["revision"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    module, chain = load_condition(
        embedding_weight=base.get_output_embeddings().weight.detach().cpu(),
        migrated=args.migrated, migrated_sha256=args.migrated_sha256,
        p33=args.p33, p33_sha256=args.p33_sha256,
        i1=args.i1, i1_sha256=args.i1_sha256,
        p34=args.p34, p34_sha256=args.p34_sha256,
        p35=args.p35, p35_sha256=args.p35_sha256,
    )
    module.bridge.set_gate_ceiling(float(args.ceiling))
    material = load_audit_material(
        positive_path=args.positive_audit,
        negative_path=args.negative_audit,
        retention_path=args.retention_panel,
        sources={
            "old": (args.old_summary, args.old_private),
            "new": (args.new_summary, args.new_private),
        },
    )
    if sha256_file(args.direction_cache) != args.direction_cache_sha256:
        raise RuntimeError("registered serving-reader direction cache SHA mismatch")
    direction_payload = torch.load(args.direction_cache, map_location="cpu", weights_only=False)
    direction_index, directions = load_p35_direction_lookup(direction_payload)
    audit, _rows = audit_model(
        module=module,
        material=material,
        direction_index=direction_index,
        directions=directions,
        seed=args.seed,
        step=4400,
        device="cuda",
    )
    output = {
        "kind": "paper2_phase3_p35_amplitude_audit_v1",
        "status": "complete_dev_audit_only",
        "seed": args.seed,
        "ceiling": args.ceiling,
        "checkpoint_sha256": args.p35_sha256,
        "checkpoint_chain": chain,
        "audit": audit,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
