"""Offline parent-replay child for A3/V4 WEFT-1 P-A materialization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.weft1_corpus_materialize_a3 import (  # noqa: E402
    run_production_materialization_worker_v4,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enumeration-receipt", required=True, type=Path)
    parser.add_argument("--cache-download-receipt", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--fasttext-model", required=True, type=Path)
    parser.add_argument("--breakdown-root", required=True, type=Path)
    parser.add_argument("--execution-provenance", required=True, type=Path)
    parser.add_argument("--runtime-build-receipt", required=True, type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    run_production_materialization_worker_v4(
        enumeration_receipt_path=arguments.enumeration_receipt,
        cache_download_receipt_path=arguments.cache_download_receipt,
        source_manifest_path=arguments.source_manifest,
        cache_root=arguments.cache_root,
        fasttext_model_path=arguments.fasttext_model,
        breakdown_root=arguments.breakdown_root,
        execution_provenance_path=arguments.execution_provenance,
        runtime_build_receipt_path=arguments.runtime_build_receipt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
