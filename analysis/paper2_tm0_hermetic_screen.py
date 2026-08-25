"""Hermetically reconstruct EVAL-E and emit only anonymous fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.prepare_paper2_dc1_dev_c import read_prior_eval_ids
from training.speculative_depth_d0_corpus import (
    SourceDocument,
    chunk_document,
    iter_fineweb_documents,
    iter_stack_documents,
)
from training.speculative_depth_d0_spec import DRAFTER_MODEL, DRAFTER_MODEL_REVISION
from training.paper2_tm0 import atomic_json, load_lock, read_jsonl, sha256_file


SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def character_shingles(text: str, width: int) -> set[bytes]:
    normalized = normalize_text(text)
    if len(normalized) <= width:
        return {normalized.encode("utf-8")}
    return {
        normalized[index : index + width].encode("utf-8")
        for index in range(len(normalized) - width + 1)
    }


def minhash_signature(
    text: str, *, width: int, components: int, seed: int
) -> np.ndarray:
    shingles = character_shingles(text, width)
    values = np.fromiter(
        (
            int.from_bytes(hashlib.sha256(value).digest()[:8], "little")
            for value in shingles
        ),
        dtype=np.uint64,
    )
    rng = np.random.default_rng(seed)
    multipliers = rng.integers(0, np.iinfo(np.uint64).max, size=components, dtype=np.uint64) | np.uint64(1)
    offsets = rng.integers(0, np.iinfo(np.uint64).max, size=components, dtype=np.uint64)
    signature = np.full(components, np.iinfo(np.uint64).max, dtype=np.uint64)
    for start in range(0, len(values), 4096):
        block = values[start : start + 4096]
        hashed = multipliers[:, None] * block[None, :] + offsets[:, None]
        signature = np.minimum(signature, hashed.min(axis=1))
    return signature


def salted_exact(text: str, salt_hex: str) -> str:
    return hashlib.sha256(bytes.fromhex(salt_hex) + normalize_text(text).encode("utf-8")).hexdigest()


def prior_d0_document_ids(manifest: dict[str, Any], data_root: Path) -> set[str]:
    result: set[str] = set()
    checked = 0
    for artifact in manifest["artifacts"].values():
        original = Path(str(artifact.get("drive_path") or ""))
        if original.suffix != ".jsonl":
            continue
        local = data_root / original.name
        if not local.is_file() or sha256_file(local) != artifact["sha256"]:
            raise RuntimeError(f"pinned D0 artifact missing or mismatched: {original.name}")
        result.update(str(row["document_id"]) for row in read_jsonl(local))
        checked += 1
    if checked < 5:
        raise RuntimeError("hermetic screen could not verify the prior D0 document universe")
    return result


def _collect_documents(
    documents: Iterable[SourceDocument],
    tokenizer: Any,
    *,
    stratum: str,
    token_budget: int,
    excluded: set[str],
    retain_text: bool,
) -> tuple[set[str], list[str], int]:
    selected_ids: set[str] = set()
    selected_texts: list[str] = []
    observed = 0
    for document in documents:
        if document.document_id in excluded:
            continue
        chunks = list(chunk_document(document, tokenizer, stratum=stratum))
        if not chunks:
            continue
        used = False
        for row in chunks:
            remaining = token_budget - observed
            if remaining <= 0:
                break
            observed += min(int(row["token_count"]), remaining)
            used = True
        if used:
            selected_ids.add(document.document_id)
            if retain_text:
                selected_texts.append(document.text)
        if observed >= token_budget:
            break
    if observed != token_budget:
        raise RuntimeError(f"{stratum} reconstructed {observed} of {token_budget} tokens")
    return selected_ids, selected_texts, observed


def reconstruct_partition(
    tokenizer: Any, *, excluded: set[str], retain_text: bool
) -> tuple[set[str], list[str], dict[str, int]]:
    general_ids, general_texts, general_tokens = _collect_documents(
        iter_fineweb_documents(),
        tokenizer,
        stratum="general",
        token_budget=100_000,
        excluded=excluded,
        retain_text=retain_text,
    )
    code_ids, code_texts, code_tokens = _collect_documents(
        iter_stack_documents(),
        tokenizer,
        stratum="code",
        token_budget=100_000,
        excluded=excluded | general_ids,
        retain_text=retain_text,
    )
    return general_ids | code_ids, general_texts + code_texts, {
        "general": general_tokens,
        "code": code_tokens,
    }


def _band_key(signature: np.ndarray, band: int, rows: int) -> bytes:
    start = band * rows
    return signature[start : start + rows].tobytes()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_manifest", type=Path, required=True)
    parser.add_argument("--d0_data_root", type=Path, required=True)
    parser.add_argument("--prior_partition_jsonl", type=Path, action="append", default=[])
    parser.add_argument("--prompt_manifest", type=Path, required=True)
    parser.add_argument("--pre_screen_receipt", type=Path, required=True)
    parser.add_argument("--sealed_index", type=Path, required=True)
    parser.add_argument("--screen_result", type=Path, required=True)
    args = parser.parse_args()

    lock = load_lock()
    params = lock["panels"]["eval_e_hermetic_screen"]
    receipt = json.loads(args.pre_screen_receipt.read_text(encoding="utf-8"))
    if receipt["parameters"] != params:
        raise RuntimeError("hermetic screen parameters differ from pre-screen receipt")
    if sha256_file(args.prompt_manifest) != receipt["prompt_manifest_sha256"]:
        raise RuntimeError("prompt-only screen manifest hash mismatch")
    source_manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    excluded = prior_d0_document_ids(source_manifest, args.d0_data_root) | read_prior_eval_ids(
        [str(path) for path in args.prior_partition_jsonl]
    )
    print(f"hermetic_screen_prior_documents={len(excluded)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        DRAFTER_MODEL, revision=DRAFTER_MODEL_REVISION
    )
    eval_d_ids, _unused, eval_d_tokens = reconstruct_partition(
        tokenizer, excluded=excluded, retain_text=False
    )
    print(f"hermetic_screen_eval_d_documents={len(eval_d_ids)}", flush=True)
    eval_e_ids, eval_e_texts, eval_e_tokens = reconstruct_partition(
        tokenizer, excluded=excluded | eval_d_ids, retain_text=True
    )
    print(f"hermetic_screen_eval_e_documents={len(eval_e_ids)}", flush=True)
    if len(eval_e_ids) != len(eval_e_texts):
        raise RuntimeError("EVAL-E document reconstruction lost text identity")

    width = int(params["character_shingle_size"])
    components = int(params["minhash_components"])
    seed = int(params["minhash_seed"])
    bands = int(params["lsh_bands"])
    rows_per_band = int(params["lsh_rows_per_band"])
    if bands * rows_per_band != components:
        raise RuntimeError("invalid registered LSH shape")
    exact_hashes = sorted(salted_exact(text, params["salt_hex"]) for text in eval_e_texts)
    signatures = [
        minhash_signature(text, width=width, components=components, seed=seed)
        for text in eval_e_texts
    ]
    print(f"hermetic_screen_signatures={len(signatures)}", flush=True)
    ordered_signatures = sorted(signature.astype(str).tolist() for signature in signatures)
    index_payload = {
        "kind": "paper2_tm0_eval_e_anonymous_screening_index_v1",
        "parameters": params,
        "document_count": len(signatures),
        "salted_exact_hashes": exact_hashes,
        "minhash_signatures_uint64_decimal": ordered_signatures,
        "plaintext_persisted": False,
        "document_ids_persisted": False,
        "metadata_persisted": False,
    }
    index_sha = atomic_json(args.sealed_index, index_payload)

    lsh: list[dict[bytes, list[int]]] = [dict() for _ in range(bands)]
    for index, signature in enumerate(signatures):
        for band in range(bands):
            lsh[band].setdefault(_band_key(signature, band, rows_per_band), []).append(index)
    exact_set = set(exact_hashes)
    threshold = float(params["estimated_jaccard_threshold"])
    dropped: list[str] = []
    maximum_similarity = 0.0
    prompt_rows = read_jsonl(args.prompt_manifest)
    for row in prompt_rows:
        text = str(row["prompt_text"])
        exact = salted_exact(text, params["salt_hex"]) in exact_set
        signature = minhash_signature(text, width=width, components=components, seed=seed)
        candidate_indices: set[int] = set()
        for band in range(bands):
            candidate_indices.update(lsh[band].get(_band_key(signature, band, rows_per_band), ()))
        similarities = [float(np.mean(signature == signatures[index])) for index in candidate_indices]
        similarity = max(similarities, default=0.0)
        maximum_similarity = max(maximum_similarity, similarity)
        if exact or similarity >= threshold:
            dropped.append(str(row["item_id"]))
    result = {
        "kind": "paper2_tm0_eval_e_hermetic_screen_result_v1",
        "status": "PASS",
        "sealed_index_sha256": index_sha,
        "sealed_document_count": len(signatures),
        "eval_d_reconstructed_document_count": len(eval_d_ids),
        "eval_e_reconstructed_document_count": len(eval_e_ids),
        "eval_d_token_counts": eval_d_tokens,
        "eval_e_token_counts": eval_e_tokens,
        "panel_rows_screened": len(prompt_rows),
        "dropped_row_count": len(dropped),
        "dropped_panel_row_ids": sorted(dropped),
        "maximum_estimated_jaccard": maximum_similarity,
        "eval_e_scored": False,
        "models_loaded": False,
        "labels_loaded": False,
        "plaintext_persisted": False,
        "membership_materialized": "hermetic-screen-only per R-TM0-P1",
    }
    atomic_json(args.screen_result, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
