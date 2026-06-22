from __future__ import annotations

import json

from colab import run_stage5_mcq_debias_diagnostic as diagnostic
from eval.mcq_debias import aggregate_permutation_scores, cyclic_permutation_rows, edge_minus_middle, rotate_mcq_row


def test_rotate_mcq_row_tracks_original_content_labels() -> None:
    row = {
        "id": "q1",
        "question": "Pick beta.",
        "choices": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        "answer": "B",
    }

    rotated = rotate_mcq_row(row, 1)

    assert rotated["id"] == "q1::perm1"
    assert rotated["choices"] == {"A": "delta", "B": "alpha", "C": "beta", "D": "gamma"}
    assert rotated["label_map"] == {"A": "D", "B": "A", "C": "B", "D": "C"}
    assert rotated["answer"] == "C"
    assert rotated["original_answer"] == "B"


def test_aggregate_permutation_scores_maps_scores_back_to_original_labels() -> None:
    row = {
        "id": "q1",
        "question": "Pick beta.",
        "choices": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        "answer": "B",
    }
    permuted = cyclic_permutation_rows([row])
    scored = []
    for perm_row in permuted:
        # Make the original B content highest regardless of its current label.
        scores = {label: -5.0 for label in perm_row["choices"]}
        current_correct_label = next(label for label, orig in perm_row["label_map"].items() if orig == "B")
        scores[current_correct_label] = 2.0
        scored.append(
            {
                "id": perm_row["id"],
                "prediction": current_correct_label,
                "answer": perm_row["answer"],
                "hit": True,
                "scores": scores,
            }
        )

    aggregated = aggregate_permutation_scores(scored, permuted)

    assert len(aggregated) == 1
    assert aggregated[0]["prediction"] == "B"
    assert aggregated[0]["answer"] == "B"
    assert aggregated[0]["hit"] is True
    assert aggregated[0]["num_permutations"] == 4
    assert aggregated[0]["permutation_prediction_counts"] == {"B": 4}


def test_edge_minus_middle_exposes_first_last_bias() -> None:
    assert edge_minus_middle({"A": 58, "B": 15, "C": 22, "D": 33}) == 54


def test_debias_diagnostic_source_payloads_follow_prior_debias_nested_summary(tmp_path, monkeypatch) -> None:
    nested = tmp_path / "arc_mix" / "summary.json"
    nested.parent.mkdir()
    nested.write_text(json.dumps({"run_id": "arc_mix", "resume_checkpoint": "checkpoint.pt"}), encoding="utf-8")
    source = tmp_path / "debias" / "summary.json"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
            {
                "kind": "stage5_mcq_debias_diagnostic",
                "status": "selection_bias_likely",
                "nested_source_summary": str(nested),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(diagnostic, "SOURCE_SUMMARY", str(source))

    source_path, source_payload, nested_path, nested_payload = diagnostic.source_payloads()

    assert source_path == source
    assert source_payload["kind"] == "stage5_mcq_debias_diagnostic"
    assert nested_path == nested
    assert nested_payload["resume_checkpoint"] == "checkpoint.pt"
