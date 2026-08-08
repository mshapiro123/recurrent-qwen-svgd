"""Materialize the absent EVAL-D rows under the original frozen data recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.cache_paper2_phase2_stage0a import read_jsonl, write_json  # noqa: E402
from eval.prepare_paper2_dc0_eval_b import prior_document_ids  # noqa: E402
from eval.prepare_paper2_dc1_dev_c import read_prior_eval_ids  # noqa: E402
from eval.prepare_paper2_phase2_eval_de import (  # noqa: E402
    PARTITION_SEEDS,
    _prepare_partition,
)
from training.speculative_depth_d0_corpus import sha256_file  # noqa: E402
from training.speculative_depth_d0_spec import (  # noqa: E402
    DRAFTER_MODEL,
    DRAFTER_MODEL_REVISION,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_manifest", type=Path, required=True)
    parser.add_argument("--prior_partition_jsonl", action="append", default=[])
    parser.add_argument("--output_data", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    prior_paths = [Path(value) for value in args.prior_partition_jsonl]
    prior_ids = prior_document_ids(source) | read_prior_eval_ids(
        [str(path) for path in prior_paths]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        DRAFTER_MODEL, revision=DRAFTER_MODEL_REVISION
    )
    rows, receipt, _ids = _prepare_partition(
        partition="eval_d",
        output_path=args.output_data,
        tokenizer=tokenizer,
        excluded_ids=prior_ids,
    )
    overlap = sorted(
        {str(row["document_id"]) for row in rows} & prior_ids
    )
    if overlap:
        raise RuntimeError("materialized EVAL-D overlaps quarantined documents")
    public = {
        "kind": "paper2_phase2_e1_eval_d_data_freeze_v1",
        "status": "complete_frozen_unscored",
        "partition": "eval_d",
        "partition_recipe": {
            "source": "original_prewindow_eval_d_recipe",
            "selection_seed": PARTITION_SEEDS["eval_d"],
            "token_budget": 200_000,
            "strata": {"general": 100_000, "code": 100_000},
            "dataset_revisions": source["dataset_revisions"],
            "tokenizer": DRAFTER_MODEL,
            "tokenizer_revision": DRAFTER_MODEL_REVISION,
        },
        "data": receipt["data"],
        "private_manifest_sha256": receipt["private_manifest_sha256"],
        "disjointness": receipt["disjointness"],
        "quarantine": {
            "prior_document_count": len(prior_ids),
            "overlap": [],
            "sources": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in prior_paths
            ],
            "d0_data_manifest_sha256": sha256_file(args.data_manifest),
        },
        "scores_exposed": False,
        "read_once_scoring_spent": False,
        "models_loaded": False,
        "teacher_cache_generated": False,
        "training_started": False,
        "optimizer_steps": 0,
        "eval_e_touched": False,
    }
    write_json(args.output_summary, public)
    print(json.dumps(public, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
