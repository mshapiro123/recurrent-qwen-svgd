"""Locked utilities for the score-blind TM-0 data and geometry wave."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
LOCK_PATH = Path(__file__).with_name("paper2_tm0_lock.json")


def load_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")
    return sha256_file(path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
    return sha256_file(path)


def row_text(row: Mapping[str, Any]) -> str:
    prompt = row.get("prompt", "")
    if isinstance(prompt, str):
        value = prompt
    else:
        value = json.dumps(prompt, sort_keys=True, ensure_ascii=True)
    answer = row.get("answer")
    return value if answer is None else value + "\n" + str(answer)


def token_ngrams(text: str, *, minimum: int = 3, maximum: int = 5) -> set[str]:
    tokens = [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]
    if not tokens:
        return set()
    values: set[str] = set()
    for width in range(minimum, min(maximum, len(tokens)) + 1):
        values.update("\x1f".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1))
    return values or {"\x1f".join(tokens)}


def near_duplicate_index(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, set[int]], list[set[str]]]:
    features = [token_ngrams(row_text(row)) for row in rows]
    inverted: dict[str, set[int]] = defaultdict(set)
    for index, grams in enumerate(features):
        for gram in grams:
            inverted[gram].add(index)
    return dict(inverted), features


def best_overlap(
    candidate: set[str], inverted: Mapping[str, set[int]], references: Sequence[set[str]]
) -> tuple[float, int | None]:
    counts: Counter[int] = Counter()
    for gram in candidate:
        counts.update(inverted.get(gram, ()))
    best_score = 0.0
    best_index: int | None = None
    for index, shared in counts.items():
        reference = references[index]
        if not candidate or not reference:
            continue
        score = max(
            shared / len(candidate),
            shared / len(reference),
            shared / (len(candidate) + len(reference) - shared),
        )
        if score > best_score:
            best_score, best_index = score, index
    return best_score, best_index


def selection_rank(row: Mapping[str, Any], seed: int) -> str:
    value = f"{seed}:tm0_extension:{row['battery']}:{row['item_id']}:{row['content_sha256']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_tm0_panel(
    source_rows: Sequence[Mapping[str, Any]],
    dev2_manifest: Sequence[Mapping[str, Any]],
    *,
    extension_size: int,
    seed: int,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_id = {str(row["item_id"]): dict(row) for row in source_rows}
    if len(by_id) != len(source_rows):
        raise RuntimeError("TM-0 source rows contain duplicate item ids")
    base: list[dict[str, Any]] = []
    for entry in dev2_manifest:
        item_id = str(entry["item_id"])
        if item_id not in by_id:
            raise RuntimeError(f"DEV-2 row missing from source table: {item_id}")
        row = dict(by_id[item_id])
        row["tm0_panel_role"] = "base_dev2"
        base.append(row)
    if len(base) != 2048 or len({row["item_id"] for row in base}) != 2048:
        raise RuntimeError("TM-0 base panel must be the exact unique 2,048-row DEV-2 set")

    base_ids = {str(row["item_id"]) for row in base}
    base_documents = {str(row["document_id"]) for row in base}
    base_content = {str(row["content_sha256"]) for row in base}

    sealed = [row for row in source_rows if row.get("partition") in {"dev", "confirm"}]
    sealed_ids = {str(row["item_id"]) for row in sealed}
    sealed_documents = {str(row["document_id"]) for row in sealed}
    sealed_content = {str(row["content_sha256"]) for row in sealed}
    inverted, reference_grams = near_duplicate_index(sealed)

    clean: dict[str, list[dict[str, Any]]] = {"arc_challenge": [], "gsm8k": []}
    rejected: list[dict[str, Any]] = []
    maximum_overlap = 0.0
    for source in source_rows:
        battery = str(source.get("battery"))
        if battery not in clean or source.get("partition") != "verified_train" or source.get("native_split") != "train":
            continue
        row = dict(source)
        overlaps_base = (
            str(row["item_id"]) in base_ids
            or str(row["document_id"]) in base_documents
            or str(row["content_sha256"]) in base_content
        )
        exact = (
            str(row["item_id"]) in sealed_ids
            or str(row["document_id"]) in sealed_documents
            or str(row["content_sha256"]) in sealed_content
        )
        overlap, matched = best_overlap(token_ngrams(row_text(row)), inverted, reference_grams)
        maximum_overlap = max(maximum_overlap, overlap)
        if overlaps_base or exact or overlap >= threshold:
            rejected.append(
                {
                    "item_id": row["item_id"],
                    "battery": battery,
                    "base_overlap": overlaps_base,
                    "exact_overlap": exact,
                    "near_duplicate_score": overlap,
                    "matched_sealed_item_id": None if matched is None else sealed[matched]["item_id"],
                }
            )
            continue
        row["tm0_selection_rank"] = selection_rank(row, seed)
        clean[battery].append(row)

    arc = sorted(clean["arc_challenge"], key=lambda row: row["tm0_selection_rank"])
    gsm = sorted(clean["gsm8k"], key=lambda row: row["tm0_selection_rank"])
    if len(arc) > extension_size:
        arc = arc[:extension_size]
    remaining = extension_size - len(arc)
    if remaining < 0 or len(gsm) < remaining:
        raise RuntimeError(
            f"TM-0 clean extension supply is insufficient: arc={len(arc)} gsm={len(gsm)} need={extension_size}"
        )
    extension = arc + gsm[:remaining]
    for row in extension:
        row["tm0_panel_role"] = "extension_verified_train"
    extension.sort(key=lambda row: (str(row["battery"]), str(row["tm0_selection_rank"])))
    panel = base + extension
    if len(panel) != 2048 + extension_size or len({row["item_id"] for row in panel}) != len(panel):
        raise RuntimeError("TM-0 panel cardinality or uniqueness changed")
    receipt = {
        "kind": "paper2_tm0_panel_freeze_v1",
        "selection_seed": seed,
        "base_rows": len(base),
        "extension_rows": len(extension),
        "total_rows": len(panel),
        "base_battery_counts": dict(sorted(Counter(row["battery"] for row in base).items())),
        "extension_battery_counts": dict(sorted(Counter(row["battery"] for row in extension).items())),
        "clean_supply": {key: len(value) for key, value in clean.items()},
        "rejected_rows": len(rejected),
        "rejected": rejected,
        "sealed_reference_rows": len(sealed),
        "near_duplicate_method": "token_ngrams_3_through_5_max_containment_or_jaccard",
        "near_duplicate_threshold": threshold,
        "maximum_candidate_overlap_before_rejection": maximum_overlap,
        "eval_e_screen": "pending_hermetic_screen_R_TM0_P1",
        "training_performed": False,
        "optimizer_constructed": False,
        "confirm_scored": False,
        "eval_e_scored": False,
    }
    return panel, extension, receipt


def deterministic_folds(labels: Sequence[str], *, folds: int, seed: int) -> torch.Tensor:
    values = torch.empty(len(labels), dtype=torch.long)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[str(label)].append(index)
    for label, indices in grouped.items():
        ordered = sorted(
            indices,
            key=lambda index: hashlib.sha256(f"{seed}:{label}:{index}".encode("ascii")).hexdigest(),
        )
        for position, index in enumerate(ordered):
            values[index] = position % folds
    return values


def centered_linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.double() - left.double().mean(dim=0, keepdim=True)
    y = right.double() - right.double().mean(dim=0, keepdim=True)
    cross = x.T @ y
    numerator = cross.square().sum()
    denominator = ((x.T @ x).square().sum() * (y.T @ y).square().sum()).sqrt().clamp_min(1e-30)
    return float(numerator / denominator)


def trace_scaled_ridge(
    x: torch.Tensor, y: torch.Tensor, multiplier: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    x = x.double()
    y = y.double()
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    xc = x - x_mean
    yc = y - y_mean
    gram = xc.T @ xc
    scale = float(torch.trace(gram) / max(gram.shape[0], 1))
    ridge = max(scale * float(multiplier), 1e-12)
    weight = torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype), xc.T @ yc
    )
    return weight.float(), x_mean.float(), y_mean.float(), ridge


def apply_ridge(x: torch.Tensor, weight: torch.Tensor, x_mean: torch.Tensor, y_mean: torch.Tensor) -> torch.Tensor:
    return (x.float() - x_mean.float()) @ weight.float() + y_mean.float()


def reconstruction_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    target = target.float()
    prediction = prediction.float()
    mse = (prediction - target).square().mean()
    mean = target.mean(dim=0, keepdim=True)
    mean_mse = (target - mean).square().mean().clamp_min(1e-30)
    cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1e-12).mean()
    return {
        "mse": float(mse),
        "relative_mse_vs_mean": float(mse / mean_mse),
        "cosine": float(cosine),
    }


def window_boundaries(start_layer: int, final_layer: int, windows: int = 3) -> list[int]:
    if not 0 <= start_layer < final_layer or windows < 1:
        raise ValueError("invalid TM-2g window request")
    span = final_layer - start_layer
    if span < windows:
        raise ValueError("teacher trajectory is too short for registered windows")
    values = [start_layer]
    for index in range(1, windows):
        candidate = start_layer + round(span * index / windows)
        candidate = max(candidate, values[-1] + 1)
        candidate = min(candidate, final_layer - (windows - index))
        values.append(candidate)
    values.append(final_layer)
    return values


def random_orthoproject(rows: int, columns: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn((columns, rows), generator=generator, dtype=torch.float64)
    q = torch.linalg.qr(raw, mode="reduced").Q.T
    return q.float()


def bivector_coordinates(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    matrix = left.unsqueeze(-1) * right.unsqueeze(-2) - right.unsqueeze(-1) * left.unsqueeze(-2)
    indices = torch.triu_indices(matrix.shape[-2], matrix.shape[-1], offset=1)
    return matrix[..., indices[0], indices[1]]


def rademacher_sketch(input_features: int, output_features: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randint(0, 2, (input_features, output_features), generator=generator, dtype=torch.int8)
    return (values.float().mul_(2).sub_(1) / math.sqrt(output_features)).contiguous()


def geometric_features(
    states: torch.Tensor,
    displacements: torch.Tensor,
    q: torch.Tensor,
    sketch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_unit = F.normalize(states.float(), dim=-1, eps=1e-12)
    displacement = displacements.float()
    radial_scalar = (state_unit * displacement).sum(dim=-1, keepdim=True)
    perpendicular = displacement - radial_scalar * state_unit
    radial_fraction = radial_scalar.square().squeeze(-1) / displacement.square().sum(dim=-1).clamp_min(1e-12)
    left = state_unit @ q.T
    right = F.normalize(perpendicular, dim=-1, eps=1e-12) @ q.T
    bivector = bivector_coordinates(left, right)
    compressed = bivector @ sketch
    return radial_fraction, F.normalize(compressed, dim=-1, eps=1e-12), perpendicular
