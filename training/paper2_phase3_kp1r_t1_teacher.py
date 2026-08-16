"""Deterministic contracts for KP-1R and teacher-fingerprint scoring."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from training.paper2_phase3_kp1_t1 import ridge_embedding_probe


KP1R_SPLIT_SEED = 20260816
KP1R_BOOTSTRAP_SEED = 20260817
KP1R_BOOTSTRAP_DRAWS = 10_000
KP1R_PERMUTATION_SEED = 20260818
KP1R_PERMUTATIONS = 10_000
KP1R_PRIMARY_BATTERIES = (
    "arc_challenge",
    "arc_easy",
    "gsm8k",
    "mmlu",
    "tier1",
)
KP1R_MCQ_BATTERIES = ("arc_challenge", "arc_easy", "mmlu")
T1_ALIGNMENT_SEED = 20260819
T1_ALIGNMENT_FRACTION = 0.60
T1_TEACHER_TAPS = (12, 24, 36, 48)
T1_TEACHER_PCA_DIM = 128


def canonical_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def answer_token_ids(row: Mapping[str, Any], tokenizer: Any) -> list[int]:
    """Return answer-bearing target IDs without prompt-separator tokens."""

    answer = str(row["answer"]).strip()
    if not answer:
        raise ValueError(f"empty canonical answer: {row['item_id']}")
    token_ids = [int(value) for value in tokenizer(answer, add_special_tokens=False)["input_ids"]]
    kept = [
        token_id
        for token_id in token_ids
        if str(tokenizer.decode([token_id], skip_special_tokens=True)).strip()
    ]
    if not kept:
        raise ValueError(f"no answer-bearing tokens: {row['item_id']}")
    return kept


def cheap_target_id(row: Mapping[str, Any], tokenizer: Any) -> int:
    return answer_token_ids(row, tokenizer)[0]


def target_entropy_audit(
    rows: Sequence[Mapping[str, Any]],
    target_ids: Sequence[int],
    *,
    enforce_batteries: Sequence[str] = KP1R_PRIMARY_BATTERIES,
) -> dict[str, dict[str, Any]]:
    if len(rows) != len(target_ids):
        raise ValueError("target audit inputs must align")
    grouped: dict[str, list[int]] = defaultdict(list)
    for row, target_id in zip(rows, target_ids):
        grouped[str(row["battery"])].append(int(target_id))
    result: dict[str, dict[str, Any]] = {}
    for battery, values in sorted(grouped.items()):
        counts = Counter(values)
        probabilities = torch.tensor(list(counts.values()), dtype=torch.float64)
        probabilities /= probabilities.sum()
        entropy = float(-(probabilities * probabilities.log()).sum())
        dominant = max(counts.values()) / len(values)
        result[battery] = {
            "rows": len(values),
            "unique_target_tokens": len(counts),
            "dominant_target_share": float(dominant),
            "target_entropy_nats": entropy,
            "target_ids_sha256": canonical_sha256([str(value) for value in values]),
        }
        if (
            battery in set(enforce_batteries)
            and battery not in KP1R_MCQ_BATTERIES
            and len(values) >= 10
        ):
            if len(counts) < 2 or dominant >= 0.90:
                raise RuntimeError(
                    f"KP-1R target degeneracy remains for {battery}: "
                    f"unique={len(counts)} dominant={dominant:.4f}"
                )
    return result


def battery_frequency_predictions(
    train_target_ids: Sequence[int],
    train_batteries: Sequence[str],
    eval_batteries: Sequence[str],
) -> list[int]:
    if len(train_target_ids) != len(train_batteries):
        raise ValueError("frequency-control train inputs must align")
    pooled = Counter(int(value) for value in train_target_ids)
    if not pooled:
        raise ValueError("frequency control needs training targets")

    def mode(counter: Counter[int]) -> int:
        maximum = max(counter.values())
        return min(token_id for token_id, count in counter.items() if count == maximum)

    fallback = mode(pooled)
    grouped: dict[str, Counter[int]] = defaultdict(Counter)
    for target_id, battery in zip(train_target_ids, train_batteries):
        grouped[str(battery)][int(target_id)] += 1
    return [mode(grouped[battery]) if grouped[battery] else fallback for battery in eval_batteries]


def permute_within_battery(
    values: Sequence[int], batteries: Sequence[str], *, seed: int
) -> list[int]:
    if len(values) != len(batteries):
        raise ValueError("permutation inputs must align")
    result = list(int(value) for value in values)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, battery in enumerate(batteries):
        grouped[str(battery)].append(index)
    generator = random.Random(int(seed))
    for indexes in grouped.values():
        shuffled = [result[index] for index in indexes]
        generator.shuffle(shuffled)
        for index, value in zip(indexes, shuffled):
            result[index] = value
    return result


def probe_token_predictions(
    *,
    train_features: torch.Tensor,
    train_target_ids: Sequence[int],
    eval_features: torch.Tensor,
    output_embedding: torch.Tensor,
    ridge: float,
) -> tuple[list[int], list[int]]:
    logits = probe_token_logits(
        train_features=train_features,
        train_target_ids=train_target_ids,
        eval_features=eval_features,
        output_embedding=output_embedding,
        ridge=ridge,
    )
    top1 = logits.argmax(dim=-1)
    return top1.tolist(), logits.topk(k=min(10, logits.shape[1]), dim=-1).indices.tolist()


def probe_token_logits(
    *,
    train_features: torch.Tensor,
    train_target_ids: Sequence[int],
    eval_features: torch.Tensor,
    output_embedding: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    normalized_embedding = F.normalize(output_embedding.float(), dim=-1)
    target = normalized_embedding[torch.tensor(train_target_ids, dtype=torch.long)]
    predicted = ridge_embedding_probe(
        train_features.float(), target, eval_features.float(), ridge=float(ridge)
    )
    return F.normalize(predicted, dim=-1) @ normalized_embedding.T


def row_accuracy(predictions: Sequence[int], targets: Sequence[int]) -> list[float]:
    if len(predictions) != len(targets):
        raise ValueError("accuracy inputs must align")
    return [float(int(prediction) == int(target)) for prediction, target in zip(predictions, targets)]


def knowledge_margin_rows(
    probe_predictions: Sequence[int],
    frequency_predictions: Sequence[int],
    targets: Sequence[int],
) -> list[float]:
    probe = row_accuracy(probe_predictions, targets)
    control = row_accuracy(frequency_predictions, targets)
    return [left - right for left, right in zip(probe, control)]


def summarize_margin(
    margins: Sequence[float],
    batteries: Sequence[str],
    *,
    seed: int = KP1R_BOOTSTRAP_SEED,
    draws: int = KP1R_BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    if len(margins) != len(batteries) or not margins:
        raise ValueError("margin summary inputs must be nonempty and aligned")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, battery in enumerate(batteries):
        grouped[str(battery)].append(index)

    def statistics(indexes_by_battery: Mapping[str, Sequence[int]]) -> tuple[float, float]:
        indexes = [index for values in indexes_by_battery.values() for index in values]
        pooled = sum(margins[index] for index in indexes) / len(indexes)
        by_battery = [
            sum(margins[index] for index in values) / len(values)
            for values in indexes_by_battery.values()
            if values
        ]
        return float(pooled), float(sum(by_battery) / len(by_battery))

    pooled, macro = statistics(grouped)
    generator = random.Random(int(seed))
    sampled_pooled: list[float] = []
    sampled_macro: list[float] = []
    for _ in range(int(draws)):
        sample = {
            battery: [generator.choice(indexes) for _ in indexes]
            for battery, indexes in grouped.items()
        }
        local_pooled, local_macro = statistics(sample)
        sampled_pooled.append(local_pooled)
        sampled_macro.append(local_macro)

    def interval(values: Sequence[float]) -> list[float]:
        ordered = sorted(values)
        return [
            float(ordered[math.floor(0.025 * (len(ordered) - 1))]),
            float(ordered[math.ceil(0.975 * (len(ordered) - 1))]),
        ]

    return {
        "rows": len(margins),
        "pooled_margin": pooled,
        "pooled_bootstrap_95ci": interval(sampled_pooled),
        "battery_macro_margin": macro,
        "battery_macro_bootstrap_95ci": interval(sampled_macro),
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
        "present_but_unread_gate": bool(
            pooled > 0
            and macro > 0
            and interval(sampled_pooled)[0] > 0
            and interval(sampled_macro)[0] > 0
        ),
    }


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    """Linear CKA without requiring the two feature widths to match."""

    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("CKA inputs must be row-aligned matrices")
    left = left.float() - left.float().mean(dim=0, keepdim=True)
    right = right.float() - right.float().mean(dim=0, keepdim=True)
    cross = left.T @ right
    numerator = cross.square().sum()
    denominator = torch.linalg.norm(left.T @ left) * torch.linalg.norm(right.T @ right)
    if not bool(denominator > 0):
        raise ValueError("CKA is undefined for zero-variance features")
    return float(numerator / denominator)


def centered_gram(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("Gram features must be a matrix")
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    return centered @ centered.T


def linear_cka_from_grams(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise ValueError("CKA Gram inputs must be equally sized square matrices")
    numerator = (left.float() * right.float()).sum()
    denominator = torch.linalg.norm(left.float()) * torch.linalg.norm(right.float())
    if not bool(denominator > 0):
        raise ValueError("CKA is undefined for zero-norm Gram matrices")
    return float(numerator / denominator)


def sample_space_basis(features: torch.Tensor, *, rank: int) -> torch.Tensor:
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("sample-space basis requires a matrix with at least two rows")
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    width = min(int(rank), centered.shape[0] - 1, centered.shape[1])
    # The left singular vectors are the eigenvectors of X X^T.  Working in
    # sample space avoids materializing the very wide 14B right-singular basis.
    gram = (centered @ centered.T).double()
    _eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    return eigenvectors[:, -width:].flip(dims=(1,)).contiguous()


def principal_angle_metrics_from_bases(
    left_basis: torch.Tensor, right_basis: torch.Tensor
) -> dict[str, float]:
    if (
        left_basis.ndim != 2
        or right_basis.ndim != 2
        or left_basis.shape[0] != right_basis.shape[0]
    ):
        raise ValueError("principal-angle bases must share the sample dimension")
    singular = torch.linalg.svdvals(left_basis.double().T @ right_basis.double()).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.acos(singular))
    return {
        "subspace_dimensions_compared": int(singular.numel()),
        "mean_angle_degrees": float(angles.mean()),
        "median_angle_degrees": float(angles.median()),
        "minimum_angle_degrees": float(angles.min()),
        "maximum_angle_degrees": float(angles.max()),
        "mean_cosine": float(singular.mean()),
    }


def principal_angle_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    """Compare feature column spaces in their shared row/sample space."""

    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("principal-angle inputs must be row-aligned matrices")
    return principal_angle_metrics_from_bases(
        sample_space_basis(left, rank=min(left.shape)),
        sample_space_basis(right, rank=min(right.shape)),
    )


def stratified_alignment_split(
    item_ids: Sequence[str],
    batteries: Sequence[str],
    *,
    seed: int = T1_ALIGNMENT_SEED,
    fit_fraction: float = T1_ALIGNMENT_FRACTION,
) -> list[str]:
    if len(item_ids) != len(batteries):
        raise ValueError("alignment split inputs must align")
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for index, (item_id, battery) in enumerate(zip(item_ids, batteries)):
        grouped[str(battery)].append((str(item_id), index))
    assignment = [""] * len(item_ids)
    for battery, values in sorted(grouped.items()):
        ranked = sorted(
            values,
            key=lambda pair: hashlib.sha256(
                f"{seed}:{battery}:{pair[0]}".encode("utf-8")
            ).digest(),
        )
        fit_rows = max(1, min(len(ranked) - 1, round(len(ranked) * float(fit_fraction))))
        for local_index, (_item_id, original_index) in enumerate(ranked):
            assignment[original_index] = "alignment_fit" if local_index < fit_rows else "alignment_eval"
    return assignment


def fit_teacher_pca(teacher_fit: torch.Tensor, *, output_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if teacher_fit.ndim != 2 or teacher_fit.shape[0] < 2:
        raise ValueError("teacher PCA requires a matrix with at least two rows")
    mean = teacher_fit.float().mean(dim=0, keepdim=True)
    centered = teacher_fit.float() - mean
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    width = min(int(output_dim), vh.shape[0])
    return mean, vh[:width].T.contiguous()


def fit_orthogonal_procrustes(
    student_fit: torch.Tensor, teacher_fit_projected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if student_fit.shape != teacher_fit_projected.shape or student_fit.ndim != 2:
        raise ValueError("Procrustes requires equal row and feature dimensions")
    student_mean = student_fit.double().mean(dim=0, keepdim=True)
    teacher_mean = teacher_fit_projected.double().mean(dim=0, keepdim=True)
    left = student_fit.double() - student_mean
    right = teacher_fit_projected.double() - teacher_mean
    u, _s, vh = torch.linalg.svd(left.T @ right, full_matrices=False)
    rotation = u @ vh
    return rotation, student_mean, teacher_mean


def transport_retrieval_metrics(
    *,
    student_fit: torch.Tensor,
    student_eval: torch.Tensor,
    teacher_fit: torch.Tensor,
    teacher_eval: torch.Tensor,
    teacher_pca_dim: int = T1_TEACHER_PCA_DIM,
) -> dict[str, Any]:
    pca_mean, pca_basis = fit_teacher_pca(teacher_fit, output_dim=teacher_pca_dim)
    teacher_fit_projected = (teacher_fit.float() - pca_mean) @ pca_basis
    teacher_eval_projected = (teacher_eval.float() - pca_mean) @ pca_basis
    if student_fit.shape[1] != teacher_fit_projected.shape[1]:
        raise ValueError("student width must match the frozen teacher PCA width")
    rotation, student_mean, teacher_mean = fit_orthogonal_procrustes(
        student_fit, teacher_fit_projected
    )
    transported = (student_eval.double() - student_mean) @ rotation + teacher_mean
    normalized_student = F.normalize(transported.float(), dim=-1)
    normalized_teacher = F.normalize(teacher_eval_projected.float(), dim=-1)
    similarity = normalized_student @ normalized_teacher.T
    order = similarity.argsort(dim=-1, descending=True)
    target = torch.arange(similarity.shape[0])[:, None]
    ranks = (order == target).nonzero(as_tuple=False)[:, 1] + 1
    relative_error = torch.linalg.norm(transported - teacher_eval_projected) / torch.linalg.norm(
        teacher_eval_projected
    ).clamp_min(1e-12)
    return {
        "fit_rows": int(student_fit.shape[0]),
        "eval_rows": int(student_eval.shape[0]),
        "teacher_pca_dim": int(pca_basis.shape[1]),
        "top1_retrieval_accuracy": float(ranks.eq(1).double().mean()),
        "top10_retrieval_accuracy": float(ranks.le(10).double().mean()),
        "mean_reciprocal_rank": float((1.0 / ranks.double()).mean()),
        "median_rank": float(ranks.double().median()),
        "relative_transport_error": float(relative_error),
    }
