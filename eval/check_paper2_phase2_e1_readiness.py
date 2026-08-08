"""Emit a score-blind readiness receipt for the E1 confirmation lock."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.paper2_phase2_e1_confirmation import (  # noqa: E402
    DRAFT_REGISTRATION,
    LEGACY_EVAL_DE_SUMMARY,
    OPTION_B_SUMMARY,
    RULE_INVENTORY,
    assess_readiness,
    sha256_file,
)


def load_optional(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=DRAFT_REGISTRATION)
    parser.add_argument("--rule_inventory", type=Path, default=RULE_INVENTORY)
    parser.add_argument("--option_b_summary", type=Path, default=OPTION_B_SUMMARY)
    parser.add_argument("--eval_d_freeze", type=Path, default=LEGACY_EVAL_DE_SUMMARY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    inventory = json.loads(args.rule_inventory.read_text(encoding="utf-8"))
    receipt = assess_readiness(
        registration=registration,
        rule_inventory=inventory,
        option_b_summary=load_optional(args.option_b_summary),
        eval_d_freeze=load_optional(args.eval_d_freeze),
    )
    receipt["inputs"] = {
        "registration": str(args.registration),
        "registration_sha256": sha256_file(args.registration),
        "rule_inventory": str(args.rule_inventory),
        "rule_inventory_sha256": sha256_file(args.rule_inventory),
        "option_b_summary": str(args.option_b_summary),
        "option_b_summary_sha256": (
            sha256_file(args.option_b_summary) if args.option_b_summary.is_file() else None
        ),
        "eval_d_freeze": str(args.eval_d_freeze),
        "eval_d_freeze_sha256": (
            sha256_file(args.eval_d_freeze) if args.eval_d_freeze.is_file() else None
        ),
    }
    write_json(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["ready_to_lock"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
