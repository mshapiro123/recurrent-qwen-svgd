"""Freeze prompt-only input and registered parameters before the hermetic screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from training.paper2_tm0 import atomic_json, load_lock, read_jsonl, sha256_file, write_jsonl


def prompt_text(row: Mapping[str, Any]) -> str:
    prompt = row.get("prompt", "")
    if isinstance(prompt, str):
        return prompt
    question = str(prompt.get("question") or "")
    labels = list(prompt.get("choice_labels") or [])
    choices = list(prompt.get("choice_text") or [])
    rendered = [question]
    rendered.extend(f"{label}. {choice}" for label, choice in zip(labels, choices))
    return "\n".join(rendered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    lock = load_lock()
    rows = read_jsonl(args.panel)
    prompts = [
        {"item_id": str(row["item_id"]), "prompt_text": prompt_text(row)} for row in rows
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = args.output_dir / "tm0_prompt_only_screen_manifest.jsonl"
    prompt_sha = write_jsonl(prompt_path, prompts)
    receipt = {
        "kind": "paper2_tm0_eval_e_pre_screen_receipt_v1",
        "authority": lock["authority"]["preflight_rulings"],
        "parameters": lock["panels"]["eval_e_hermetic_screen"],
        "panel_sha256": sha256_file(args.panel),
        "prompt_manifest_rows": len(prompts),
        "prompt_manifest_sha256": prompt_sha,
        "contains_answers": False,
        "contains_labels": False,
        "created_before_screen": True,
    }
    receipt_path = args.output_dir / "tm0_eval_e_pre_screen_receipt.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
