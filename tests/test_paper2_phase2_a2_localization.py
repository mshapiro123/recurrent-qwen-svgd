from __future__ import annotations

import torch

from eval.eval_paper2_phase2_a2_localization import localize
from training.paper2_phase2_matched_alpha import stable_fraction


def _rows(accepted: list[float], correct: list[list[bool]], gate: float) -> dict:
    base = torch.ones((len(accepted), 2), dtype=torch.bool)
    return {
        "accepted_length": torch.tensor(accepted),
        "base_accepted_length": torch.ones(len(accepted)),
        "acceptance_delta": torch.tensor(accepted) - 1,
        "base_correct_by_horizon": base,
        "bridge_correct_by_horizon": torch.tensor(correct),
        "quality_loss": (base & ~torch.tensor(correct)).any(dim=1),
        "probe_kl": torch.arange(len(accepted), dtype=torch.float32),
        "probe_top1": torch.linspace(0, 1, len(accepted)),
        "draft_gate": torch.full((len(accepted), 2), gate),
    }


def test_localization_finds_replicated_structural_harm_pocket() -> None:
    count = 400
    metadata = {
        "documents": [f"doc-{index}" for index in range(count)],
        "strata": ["code" if index < 200 else "general" for index in range(count)],
        "positions": [8 if index < 200 else 64 for index in range(count)],
        "position_buckets": ["token_4_31" if index < 200 else "token_32_127" for index in range(count)],
    }
    full_accepted = [0.5] * 200 + [1.2] * 200
    control_accepted = [1.0] * count
    full_correct = [[False, True]] * 200 + [[True, True]] * 200
    control_correct = [[True, True]] * count
    rows = {}
    for seed in (0, 1):
        rows[(seed, "full_a2")] = _rows(full_accepted, full_correct, 0.9)
        rows[(seed, "draft_only_control")] = _rows(control_accepted, control_correct, 0.0)

    # Keep every synthetic document on the evaluation side while exercising the
    # production document-isolation function.
    evaluation_documents = []
    candidate = 0
    while len(evaluation_documents) < count:
        document = f"eval-doc-{candidate}"
        if stable_fraction(document, seed=20260804) < 0.2:
            evaluation_documents.append(document)
        candidate += 1
    metadata["documents"] = evaluation_documents
    result = localize(
        a2_summary={"train_anchors": 400, "evaluation_anchors": 400},
        metadata=metadata,
        rows=rows,
    )
    assert result["recommended_single_mask"] is not None
    assert "code" in result["recommended_single_mask"]["label"]
    assert result["recommended_single_mask"]["qualifies_as_structural_mask"]
    assert result["model_inference_runs"] == 0
    assert result["optimizer_steps"] == 0


def test_localization_source_prohibits_training_and_model_compute() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "eval/eval_paper2_phase2_a2_localization.py"
    ).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "transformers" not in source
    assert '"model_inference_runs": 0' in source
    assert '"optimizer_steps": 0' in source
