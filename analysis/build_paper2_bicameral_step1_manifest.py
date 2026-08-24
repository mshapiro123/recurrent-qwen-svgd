"""Build the locked 256-row Bicameral Step-1 cache manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoTokenizer

from eval.eval_paper2_phase3_p31_references import (
    MODEL_SPECS,
    _chat_prompt,
    _generation_prompt,
    _mcq,
    _mcq_prompt,
)
from models.bicameral import SEQUENTIAL_EXECUTION_SCHEDULE


KIND = "paper2_bicameral_step1_manifest_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def native_reader_lengths(tokenizer: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    battery = str(row["battery"])
    if battery in {"arc_easy", "arc_challenge", "mmlu"}:
        question, choices, _answer = _mcq(row)
        labels = [label for label, _text in choices]
        candidate_lengths: list[int] = []
        prompt_lengths: list[int] = []
        for shift in range(len(choices)):
            permuted = []
            for new_index, new_label in enumerate(labels):
                _original_label, original_text = choices[(new_index - shift) % len(choices)]
                permuted.append((new_label, original_text))
            prompt = _mcq_prompt(question, permuted)
            prompt_lengths.append(
                len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            )
            for label in labels:
                candidate_lengths.append(
                    len(tokenizer(prompt + f" {label}", add_special_tokens=True)["input_ids"])
                )
        return {
            "reader_schedule": "cyclic_label_aggregated_permutation_mean_v1",
            "native_input_token_length": max(candidate_lengths),
            "native_prompt_token_length": max(prompt_lengths),
            "reader_forward_count": len(candidate_lengths),
        }
    content, generation_cap = _generation_prompt(row)
    prompt = _chat_prompt(tokenizer, content)
    prompt_length = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    return {
        "reader_schedule": "greedy_generation_v1",
        "native_input_token_length": prompt_length,
        "native_prompt_token_length": prompt_length,
        "reader_forward_count": 1,
        "generation_cap": int(generation_cap),
    }


def representative_reader_input(tokenizer: Any, row: Mapping[str, Any]) -> list[int]:
    battery = str(row["battery"])
    if battery in {"arc_easy", "arc_challenge", "mmlu"}:
        question, choices, _answer = _mcq(row)
        prompt = _mcq_prompt(question, choices)
        first_label = choices[0][0]
        return [
            int(value)
            for value in tokenizer(
                prompt + f" {first_label}", add_special_tokens=True
            )["input_ids"]
        ]
    content, _generation_cap = _generation_prompt(row)
    prompt = _chat_prompt(tokenizer, content)
    return [
        int(value)
        for value in tokenizer(prompt, add_special_tokens=True)["input_ids"]
    ]


def load_preserved_ids(paths: list[Path]) -> tuple[list[str], list[str], dict[str, str]]:
    identities: list[tuple[list[str], list[str]]] = []
    hashes: dict[str, str] = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        item_ids = [str(value) for value in payload["item_ids"]]
        batteries = [str(value) for value in payload["batteries"]]
        if len(item_ids) != 256 or len(item_ids) != len(set(item_ids)):
            raise RuntimeError(f"invalid preserved arm-6 population: {path}")
        identities.append((item_ids, batteries))
        hashes[path.name + ":" + path.parent.name] = sha256_file(path)
    if any(value != identities[0] for value in identities[1:]):
        raise RuntimeError("seed-specific preserved arm-6 populations differ")
    return identities[0][0], identities[0][1], hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--correction-artifact", type=Path, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--probe-batch-output", type=Path)
    args = parser.parse_args()

    model = MODEL_SPECS["base"]
    tokenizer = AutoTokenizer.from_pretrained(model["model"], revision=model["revision"])
    item_ids, batteries, artifact_hashes = load_preserved_ids(args.correction_artifact)
    panel_rows = read_jsonl(args.panel)
    panel = {str(row["item_id"]): row for row in panel_rows}
    if len(panel) != len(panel_rows):
        raise RuntimeError("registered panel contains duplicate item ids")

    records: list[dict[str, Any]] = []
    for index, (item_id, battery) in enumerate(zip(item_ids, batteries)):
        if item_id not in panel:
            raise RuntimeError(f"preserved arm-6 row missing from registered panel: {item_id}")
        source = panel[item_id]
        if str(source["battery"]) != battery:
            raise RuntimeError(f"battery mismatch for preserved row: {item_id}")
        lengths = native_reader_lengths(tokenizer, source)
        row = {
            "manifest_index": index,
            "item_id": item_id,
            "battery": battery,
            "native_split": source["native_split"],
            "partition": source["partition"],
            "reader": source["reader"],
            "source_content_sha256": source["content_sha256"],
            "source_row_sha256": canonical_sha256(source),
            **lengths,
        }
        row["manifest_row_sha256"] = canonical_sha256(row)
        records.append(row)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in records),
        encoding="utf-8",
        newline="\n",
    )
    lengths = [int(row["native_input_token_length"]) for row in records]
    summary = {
        "kind": KIND,
        "status": "LOCKED_BEFORE_REVISED_COST_PROBE",
        "training_performed": False,
        "optimizer_constructed": False,
        "sealed_partitions_touched": False,
        "row_count": len(records),
        "battery_counts": dict(sorted(Counter(row["battery"] for row in records).items())),
        "native_input_token_lengths": {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "sorted_counts": dict(sorted(Counter(lengths).items())),
        },
        "execution_schedule": SEQUENTIAL_EXECUTION_SCHEDULE,
        "model": model,
        "source_panel": {"path": str(args.panel), "sha256": sha256_file(args.panel)},
        "source_correction_artifacts": artifact_hashes,
        "manifest": {"path": str(args.output_jsonl), "sha256": sha256_file(args.output_jsonl)},
    }
    if args.probe_batch_output is not None:
        selected: list[dict[str, Any]] = []
        selected_counts: Counter[str] = Counter()
        for record in records:
            battery = str(record["battery"])
            if selected_counts[battery] >= 2:
                continue
            source = panel[str(record["item_id"])]
            selected.append(
                {
                    "item_id": record["item_id"],
                    "battery": battery,
                    "input_ids": representative_reader_input(tokenizer, source),
                }
            )
            selected_counts[battery] += 1
        args.probe_batch_output.parent.mkdir(parents=True, exist_ok=True)
        args.probe_batch_output.write_text(
            json.dumps(selected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        summary["probe_batch"] = {
            "path": str(args.probe_batch_output),
            "sha256": sha256_file(args.probe_batch_output),
            "rows": len(selected),
            "battery_counts": dict(sorted(selected_counts.items())),
            "selection": "first_two_manifest_rows_per_battery",
        }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
