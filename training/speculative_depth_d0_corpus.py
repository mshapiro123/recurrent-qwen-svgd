"""Deterministic corpus freezing for the registered Paper Two D0 pilot."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from training.speculative_depth_d0_spec import (
    FINEWEB_DATASET,
    FINEWEB_DUMP,
    FINEWEB_IN_ERA_DUMP,
    FINEWEB_REVISION,
    PARTITION_SEED,
    PILOT_SEED,
    STACK_DATASET,
    STACK_LANGUAGE_DIRECTORIES,
    STACK_REVISION,
)


PARTITION_FRACTIONS = {"label_train": 0.8, "calibration": 0.1, "evaluation": 0.1}


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    text: str
    metadata: dict[str, Any]


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fraction(value: str, *, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def partition_for_document(document_id: str, *, seed: int = PARTITION_SEED) -> str:
    value = stable_fraction(document_id, seed=seed)
    if value < PARTITION_FRACTIONS["label_train"]:
        return "label_train"
    if value < PARTITION_FRACTIONS["label_train"] + PARTITION_FRACTIONS["calibration"]:
        return "calibration"
    return "evaluation"


def choose_stratum_mix(densities: dict[str, float]) -> dict[str, Any]:
    """Apply the pre-observation density/domain-coverage rule."""

    if set(densities) != {"general", "code"}:
        raise ValueError("D0 density receipt must contain general and code")
    if min(densities.values()) < 0:
        raise ValueError("D0 rejection densities cannot be negative")
    high = max(densities, key=densities.get)
    low = "code" if high == "general" else "general"
    maximum = float(densities[high])
    ratio = 1.0 if maximum == 0 else float(densities[low]) / maximum
    if ratio >= 0.5:
        mix = {"general": 0.5, "code": 0.5}
        rule = "balanced_density_and_domain_coverage"
    else:
        mix = {high: 0.6, low: 0.4}
        rule = "density_skew_with_forty_percent_domain_floor"
    return {
        "densities_per_1000_tokens": {key: float(value) for key, value in densities.items()},
        "minimum_to_maximum_density_ratio": ratio,
        "rule": rule,
        "mix": mix,
    }


def token_quotas(total_tokens: int, mix: dict[str, float]) -> dict[str, dict[str, int]]:
    if total_tokens <= 0 or abs(sum(mix.values()) - 1.0) > 1e-9:
        raise ValueError("D0 token budget and stratum mix are invalid")
    result: dict[str, dict[str, int]] = {}
    remaining_total = int(total_tokens)
    strata = list(sorted(mix))
    for stratum_index, stratum in enumerate(strata):
        if stratum_index == len(strata) - 1:
            stratum_total = remaining_total
        else:
            stratum_total = int(round(total_tokens * float(mix[stratum])))
            remaining_total -= stratum_total
        train = int(round(stratum_total * PARTITION_FRACTIONS["label_train"]))
        calibration = int(round(stratum_total * PARTITION_FRACTIONS["calibration"]))
        evaluation = stratum_total - train - calibration
        result[stratum] = {
            "label_train": train,
            "calibration": calibration,
            "evaluation": evaluation,
        }
    return result


def chunk_document(
    document: SourceDocument,
    tokenizer: Any,
    *,
    stratum: str,
    max_length: int = 512,
    minimum_length: int = 32,
) -> Iterator[dict[str, Any]]:
    token_ids = tokenizer(document.text, add_special_tokens=False)["input_ids"]
    for chunk_index, start in enumerate(range(0, len(token_ids), max_length)):
        chunk = [int(value) for value in token_ids[start : start + max_length]]
        if len(chunk) < minimum_length:
            continue
        yield {
            "row_id": f"{stratum}_{document.document_id}_{chunk_index:05d}",
            "document_id": document.document_id,
            "chunk_index": chunk_index,
            "stratum": stratum,
            "input_ids": chunk,
            "token_count": len(chunk),
            "source": document.metadata,
        }


def collect_probe_rows(
    documents: Iterable[SourceDocument],
    tokenizer: Any,
    *,
    stratum: str,
    token_budget: int,
    excluded_document_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    excluded = set(excluded_document_ids or set())
    selected: list[dict[str, Any]] = []
    selected_documents: set[str] = set()
    observed = 0
    for document in documents:
        if document.document_id in excluded:
            continue
        chunks = list(chunk_document(document, tokenizer, stratum=stratum))
        if not chunks:
            continue
        for row in chunks:
            remaining = token_budget - observed
            if remaining <= 0:
                break
            if int(row["token_count"]) > remaining:
                row = dict(row)
                row["input_ids"] = row["input_ids"][:remaining]
                row["token_count"] = remaining
            selected.append(row)
            observed += int(row["token_count"])
            selected_documents.add(document.document_id)
        if observed >= token_budget:
            break
    if observed != int(token_budget):
        raise RuntimeError(f"{stratum} probe produced {observed} of {token_budget} tokens")
    return selected, selected_documents


def collect_partition_rows(
    documents: Iterable[SourceDocument],
    tokenizer: Any,
    *,
    stratum: str,
    quotas: dict[str, int],
    excluded_document_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    selected = {partition: [] for partition in PARTITION_FRACTIONS}
    counts = {partition: 0 for partition in PARTITION_FRACTIONS}
    for document in documents:
        if document.document_id in excluded_document_ids:
            continue
        partition = partition_for_document(document.document_id)
        if counts[partition] >= int(quotas[partition]):
            continue
        chunks = list(chunk_document(document, tokenizer, stratum=stratum))
        for row in chunks:
            remaining = int(quotas[partition]) - counts[partition]
            if remaining <= 0:
                break
            if int(row["token_count"]) > remaining:
                row = dict(row)
                row["input_ids"] = row["input_ids"][:remaining]
                row["token_count"] = remaining
            selected[partition].append(row)
            counts[partition] += int(row["token_count"])
        if all(counts[name] >= int(quotas[name]) for name in quotas):
            break
    missing = {name: int(quotas[name]) - counts[name] for name in quotas if counts[name] != int(quotas[name])}
    if missing:
        raise RuntimeError(f"{stratum} corpus did not fill partition quotas: {missing}")
    return selected


def select_pilot_rows(rows: list[dict[str, Any]], *, count: int = 256) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"D0 pilot requires {count} rows, observed {len(rows)}")
    ordered = sorted(rows, key=lambda row: stable_fraction(str(row["row_id"]), seed=PILOT_SEED))
    return [dict(row) for row in ordered[:count]]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    token_count = 0
    document_ids: set[str] = set()
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            row_count += 1
            token_count += int(row.get("token_count", 0))
            document_ids.add(str(row.get("document_id") or ""))
    return {
        "path": destination.as_posix(),
        "sha256": sha256_file(destination),
        "rows": row_count,
        "tokens": token_count,
        "documents": len(document_ids - {""}),
        "document_id_sha256": sha256_bytes(("\n".join(sorted(document_ids)) + "\n").encode("utf-8")),
    }


def assert_document_disjoint(partitions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    document_sets = {
        name: {str(row["document_id"]) for row in rows}
        for name, rows in partitions.items()
    }
    overlaps: dict[str, list[str]] = {}
    names = sorted(document_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = sorted(document_sets[left] & document_sets[right])
            if shared:
                overlaps[f"{left}:{right}"] = shared[:20]
    if overlaps:
        raise RuntimeError(f"D0 document-level split leaked: {overlaps}")
    return {"document_disjoint": True, "overlaps": {}, "document_counts": {k: len(v) for k, v in document_sets.items()}}


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def iter_fineweb_documents(*, dump: str = FINEWEB_DUMP) -> Iterator[SourceDocument]:
    from datasets import load_dataset

    dataset = load_dataset(
        FINEWEB_DATASET,
        dump,
        split="train",
        streaming=True,
        revision=FINEWEB_REVISION,
        token=_hf_token(),
    )
    for row in dataset:
        text = str(row.get("text") or "")
        document_id = str(row.get("id") or "")
        if not document_id or not text.strip():
            continue
        yield SourceDocument(
            document_id=f"fineweb:{dump}:{document_id}",
            text=text,
            metadata={
                "dataset": FINEWEB_DATASET,
                "revision": FINEWEB_REVISION,
                "dump": dump,
                "id": document_id,
                "url": row.get("url"),
                "date": row.get("date"),
                "license": "ODC-By-1.0",
            },
        )


def iter_in_era_documents() -> Iterator[SourceDocument]:
    return iter_fineweb_documents(dump=FINEWEB_IN_ERA_DUMP)


def normalize_stack_licenses(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = [str(item) for item in value]
    else:
        values = []
    return sorted({item.strip() for item in values if item.strip()})


def iter_stack_documents() -> Iterator[SourceDocument]:
    from datasets import load_dataset

    streams: dict[str, tuple[str, Iterator[dict[str, Any]]]] = {}
    for language_directory, language in STACK_LANGUAGE_DIRECTORIES.items():
        dataset = load_dataset(
            STACK_DATASET,
            data_dir=f"data/{language_directory}",
            split="train",
            streaming=True,
            revision=STACK_REVISION,
            token=_hf_token(),
        )
        streams[language_directory] = (language, iter(dataset))

    while streams:
        for language_directory in list(streams):
            language, stream = streams[language_directory]
            try:
                row = next(stream)
            except StopIteration:
                del streams[language_directory]
                continue
            text = str(row.get("content") or "")
            if not text.strip():
                continue
            licenses = normalize_stack_licenses(row.get("licenses"))
            if not licenses:
                continue
            repository_name = str(row.get("repository_name") or "")
            path = str(row.get("path") or "")
            identity = canonical_json(
                {
                    "language_directory": language_directory,
                    "repository_name": repository_name,
                    "path": path,
                    "content_sha256": sha256_bytes(text.encode("utf-8")),
                }
            )
            document_id = sha256_bytes(identity.encode("utf-8"))
            yield SourceDocument(
                document_id=f"stacksmol:{document_id}",
                text=text,
                metadata={
                    "dataset": STACK_DATASET,
                    "revision": STACK_REVISION,
                    "lineage": "Stack_v1",
                    "provenance_period": "in_pretraining_era",
                    "language_directory": language_directory,
                    "language": language,
                    "repository_name": repository_name,
                    "path": path,
                    "licenses": licenses,
                    "size": row.get("size"),
                },
            )
