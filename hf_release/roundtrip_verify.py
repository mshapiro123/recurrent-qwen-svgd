"""Download one private HF release through the API and run its frozen gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


LETTER_SYMBOLS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_for_row(row: dict[str, Any]) -> str:
    return f"{str(row.get('question', row.get('prompt'))).rstrip()}\nAnswer:"


def symbol_names(row: dict[str, Any]) -> list[str]:
    if isinstance(row.get("symbol_names"), list):
        return [str(value) for value in row["symbol_names"]]
    return list(LETTER_SYMBOLS[: int(row["n_symbols"])])


def candidate_token_ids(tokenizer: Any, prompt: str, candidates: list[str]) -> dict[str, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    result: dict[str, int] = {}
    for candidate in candidates:
        full_ids = tokenizer(prompt + f" {candidate}", add_special_tokens=True)["input_ids"]
        suffix = full_ids[len(prompt_ids) :]
        if len(suffix) != 1:
            raise RuntimeError(f"Verification candidate is not one token: {candidate!r} -> {suffix}")
        result[candidate] = int(suffix[0])
    return result


def evaluate(model: Any, tokenizer: Any, rows: list[dict[str, Any]], device: str) -> dict[str, Any]:
    correct = Counter()
    totals = Counter()
    predictions: list[dict[str, Any]] = []
    for row in rows:
        prompt = prompt_for_row(row)
        candidates = symbol_names(row)
        ids = candidate_token_ids(tokenizer, prompt, candidates)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        depth = int(row["depth"])
        with torch.no_grad():
            output = model(
                **encoded,
                max_loops=depth,
                loop_selection=depth,
                use_cache=False,
                return_dict=True,
            )
        next_logits = output.logits[0, -1]
        prediction = max(ids, key=lambda name: float(next_logits[ids[name]].item()))
        target = str(row["target"])
        hit = prediction == target
        totals[str(depth)] += 1
        correct[str(depth)] += int(hit)
        predictions.append({"id": row["id"], "depth": depth, "prediction": prediction, "target": target, "hit": hit})
    return {
        "correct_by_depth": dict(sorted(correct.items())),
        "total_by_depth": dict(sorted(totals.items())),
        "predictions": predictions,
    }


def one_loop_identity(model: Any, tokenizer: Any, row: dict[str, Any], device: str) -> dict[str, Any]:
    encoded = tokenizer(prompt_for_row(row), return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        recurrent = model(**encoded, max_loops=1, use_cache=False, return_dict=True).logits
        dense = model.backbone(**encoded, use_cache=False, return_dict=True).logits
    difference = float((recurrent - dense).abs().max().item())
    return {"max_abs_logit_difference": difference, "exact_tensor_equality": bool(torch.equal(recurrent, dense))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    api = HfApi()
    info = api.model_info(args.repo_id, files_metadata=True)
    if info.private is not True:
        raise RuntimeError(f"Release gate requires a private repo, got private={info.private}")
    snapshot = Path(snapshot_download(args.repo_id, repo_type="model", revision=info.sha))
    spec = json.loads((snapshot / "verification_spec.json").read_text(encoding="utf-8"))
    data_path = snapshot / "verification_subset.jsonl"
    if sha256_file(data_path) != spec["verification_data_sha256"]:
        raise RuntimeError("Downloaded verification subset hash does not match its receipt")
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", revision="main")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        trust_remote_code=True,
        dtype=dtype,
    ).to(args.device).eval()
    evaluation = evaluate(model, tokenizer, rows, args.device)
    expected = {str(key): int(value) for key, value in spec["expected_correct_by_depth"].items()}
    counts_match = evaluation["correct_by_depth"] == expected
    identity = one_loop_identity(model, tokenizer, rows[0], args.device) if spec["identity_check"] else None
    identity_green = identity is None or (
        identity["exact_tensor_equality"] and identity["max_abs_logit_difference"] == 0.0
    )
    file_hashes = {
        path.name: sha256_file(path)
        for path in snapshot.iterdir()
        if path.is_file() and path.name in {"config.json", "recurrent_delta.safetensors", "conversion_receipt.json"}
    }
    receipt = {
        "schema_version": 1,
        "kind": "paper_one_hf_private_roundtrip",
        "status": "green" if counts_match and identity_green else "red",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo_id": args.repo_id,
        "repo_private": True,
        "repo_commit": info.sha,
        "download_method": "huggingface_hub.snapshot_download pinned to repo_commit",
        "snapshot_file_sha256": file_hashes,
        "model_load": getattr(model, "_release_load_receipt", None),
        "forced_depth_interface": "max_loops plus external loop_selection; no halting",
        "verification_spec": spec,
        "observed_correct_by_depth": evaluation["correct_by_depth"],
        "observed_total_by_depth": evaluation["total_by_depth"],
        "counts_exact_match": counts_match,
        "loop1_identity": identity,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "accelerator": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            "dtype": str(dtype).removeprefix("torch."),
        },
        "predictions": evaluation["predictions"],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "predictions"}, indent=2))
    return 0 if receipt["status"] == "green" else 1


if __name__ == "__main__":
    raise SystemExit(main())

