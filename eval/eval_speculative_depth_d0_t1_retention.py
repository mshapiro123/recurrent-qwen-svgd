"""Run the locked T1-family mechanism-retention guardrail for D0."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_speculative_depth_d0_teachers import load_drafter, read_jsonl
from eval.eval_internal_think_token_t1_lite import evaluate_rows, prepare_control_rows
from training.speculative_depth_d0_corpus import sha256_file


REFERENCE_CORRECT = 971
REFERENCE_TOTAL = 1024
MAXIMUM_ABSOLUTE_DROP = 0.03


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint_sha256", required=True)
    parser.add_argument("--frozen_eval_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("D0 T1 retention checkpoint SHA-256 mismatch")
    source = [row for row in read_jsonl(args.frozen_eval_jsonl) if 1 <= int(row["depth"]) <= 8]
    if len(source) != REFERENCE_TOTAL:
        raise RuntimeError(f"D0 T1 retention expected {REFERENCE_TOTAL} frozen rows, found {len(source)}")
    output_dir = Path(args.output_dir)
    data_path = prepare_control_rows(source, output_dir / "private" / "t1_retention_control.jsonl")
    tokenizer, wrapper, resize, _original_vocab = load_drafter(
        checkpoint=Path(args.checkpoint),
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    continue_id, stop_id, readout_id = (int(value) for value in resize.control_token_ids)
    evaluation = evaluate_rows(
        wrapper,
        tokenizer,
        data_path,
        device=args.device,
        max_loops=12,
        batch_size=args.batch_size,
        continue_id=continue_id,
        stop_id=stop_id,
        readout_id=readout_id,
        include_features=False,
    )
    reference_accuracy = REFERENCE_CORRECT / REFERENCE_TOTAL
    observed_accuracy = evaluation["forced_correct"] / evaluation["total"]
    passed = reference_accuracy - observed_accuracy <= MAXIMUM_ABSOLUTE_DROP
    write_json(output_dir / "private" / "rows.json", evaluation.pop("rows"))
    summary = {
        "kind": "paper2_d0_t1_mechanism_retention",
        "status": "complete" if passed else "blocked_guardrail",
        "checkpoint_sha256": args.checkpoint_sha256,
        "reference": {
            "correct": REFERENCE_CORRECT,
            "total": REFERENCE_TOTAL,
            "accuracy": reference_accuracy,
            "source": "stage5_paper2_t1_lite_r_20260725/eval/raw_primary/summary.json",
        },
        "observed": evaluation,
        "absolute_drop": reference_accuracy - observed_accuracy,
        "maximum_absolute_drop": MAXIMUM_ABSOLUTE_DROP,
        "passed": passed,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
