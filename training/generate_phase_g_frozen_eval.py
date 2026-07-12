"""Generate the locked N=24 Phase G-alpha calibration and test sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.abductive_injective_task import (
    PhaseGFrozenEvalConfig,
    build_phase_g_frozen_rows,
    validate_phase_g_frozen_rows,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="data/phase_g_alpha")
    parser.add_argument("--seed", type=int, default=7_194_203)
    parser.add_argument("--rows_per_stratum", type=int, default=128)
    parser.add_argument("--depths", default="1,2,3,4")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = PhaseGFrozenEvalConfig(
        n_symbols=24,
        depths=tuple(int(item) for item in args.depths.split(",") if item.strip()),
        rows_per_stratum=int(args.rows_per_stratum),
        seed=int(args.seed),
    )
    manifests = {}
    row_ids: dict[str, set[str]] = {}
    for split in ("calibration", "test"):
        rows = build_phase_g_frozen_rows(config, split=split)
        validation = validate_phase_g_frozen_rows(
            rows,
            rows_per_stratum=config.rows_per_stratum,
        )
        if validation["status"] != "passed":
            raise RuntimeError(f"{split} frozen rows failed validation: {validation['errors'][:10]}")
        path = output_dir / f"{split}_n24.jsonl"
        write_jsonl(path, rows)
        validation["path"] = path.relative_to(ROOT).as_posix()
        manifests[split] = validation
        row_ids[split] = {str(row["id"]) for row in rows}

    overlap = sorted(row_ids["calibration"].intersection(row_ids["test"]))
    if overlap:
        raise RuntimeError(f"Calibration/test row overlap: {overlap[:5]}")
    payload = {
        "kind": "phase_g_alpha_frozen_eval_manifest",
        "status": "passed",
        "config": {
            "n_symbols": config.n_symbols,
            "depths": list(config.depths),
            "rows_per_stratum": config.rows_per_stratum,
            "seed": config.seed,
            "preimage_strata": {
                "unique": "exactly 1",
                "small": "2-4",
                "large": ">=5",
            },
            "mapping_distribution": "iid_uniform_arbitrary_table_conditioned_on_reachable_target_and_stratum",
            "posterior_chain_sampling": "uniform_over_exact_valid_preimages",
        },
        "splits": manifests,
        "calibration_test_id_overlap": 0,
        "oracle_denominator": "independent_forward_orbit_enumeration_over_all_24_starts",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
