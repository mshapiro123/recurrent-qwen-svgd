"""Write the Stage 2B fixed-prompt A100/L4 comparison receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.eval_paper2_stage2b_riders import fixed_prompt_comparison_receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a100-logits", type=Path, required=True)
    parser.add_argument("--l4-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = fixed_prompt_comparison_receipt(args.a100_logits, args.l4_logits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "runtime_aligned" else 2


if __name__ == "__main__":
    raise SystemExit(main())
