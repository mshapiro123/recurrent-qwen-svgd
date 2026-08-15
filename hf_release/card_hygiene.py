"""Validate bounded claims and Hugging Face model-card metadata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from huggingface_hub import ModelCard


ROOT = Path(__file__).resolve().parents[1]
ARXIV_ID = "arXiv:2608.11233"


def validate_card(path: Path, repo_name: str, prohibited: list[str]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    parsed = ModelCard(text)
    metadata = parsed.data.to_dict()
    errors: list[str] = []
    if metadata.get("license") != "apache-2.0":
        errors.append("license must be apache-2.0")
    if metadata.get("base_model") != "Qwen/Qwen2.5-0.5B-Instruct":
        errors.append("base_model must auto-link to Qwen/Qwen2.5-0.5B-Instruct")
    if metadata.get("library_name") != "transformers":
        errors.append("selected in-repo custom loader requires library_name: transformers")
    tags = set(metadata.get("tags") or [])
    for tag in {"recurrent-depth", "latent-reasoning", "qwen2.5", "research"}:
        if tag not in tags:
            errors.append(f"missing tag: {tag}")
    for phrase in prohibited:
        if phrase.lower() in text.lower():
            errors.append(f"prohibited phrase: {phrase}")
    if "[insert from lineage receipt]" in text:
        errors.append("lineage SHA placeholder remains")
    if "arXiv:XXXX.XXXXX" in text:
        errors.append("reserved arXiv placeholder remains; the paper is published")
    if ARXIV_ID not in text:
        errors.append(f"card must cite the published paper as {ARXIV_ID}")
    stale_loader_phrases = {
        "recurrent-qwen2.5-0.5b-full-block": "or through the loader in the companion repository",
        "recurrent-qwen2.5-0.5b-natural-keeper": "`trust_remote_code=True` or the repository loader",
        "recurrent-qwen2.5-0.5b-r16-adapter": "Load through the wrapper in the companion repository",
    }
    if stale_loader_phrases[repo_name] in text:
        errors.append("loading prose does not match the selected in-repo custom loader")
    for paragraph in re.split(r"\n\s*\n", text):
        if "arm t" in paragraph.lower() and re.search(r"step[- ]?4,?000", paragraph, re.I):
            errors.append("card cites the prohibited step-4,000 Arm T peak")
    return {
        "repo_name": repo_name,
        "card": str(path),
        "metadata": metadata,
        "errors": errors,
        "status": "green" if not errors else "red",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-root", type=Path, default=ROOT / "docs/hf_cards")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    ledger = json.loads((ROOT / "docs/part1_claim_evidence_ledger.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "hf_release/release_manifest.json").read_text(encoding="utf-8"))
    results = []
    for repo_name, spec in manifest["repos"].items():
        card_name = Path(spec["card"]).name
        results.append(validate_card(args.cards_root / card_name, repo_name, ledger["global_prohibited_phrases"]))
    receipt = {
        "schema_version": 1,
        "kind": "paper_one_hf_card_hygiene",
        "status": "green" if all(row["status"] == "green" for row in results) else "red",
        "cards": results,
    }
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "green" else 1


if __name__ == "__main__":
    raise SystemExit(main())
