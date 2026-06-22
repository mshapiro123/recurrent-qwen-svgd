from __future__ import annotations

import json

from colab import run_stage5_mcq_debias_diagnostic as diagnostic
from colab import apply_stage5_mcq_scoring_policy as scoring_policy
from colab.assess_stage5_mcq_debias_pair import assess_pair
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


def mcq_debias_summary(*, arc_config: str, status: str, cyclic_delta: int, best_delta: int) -> dict:
    return {
        "kind": "stage5_mcq_debias_diagnostic",
        "run_id": f"{arc_config}_run",
        "arc_config": arc_config,
        "arc_limit": 128,
        "status": status,
        "passed": status == "selection_bias_likely",
        "decision": {
            "status": status,
            "label_delta": -5,
            "content_delta": -8,
            "cyclic_delta": cyclic_delta,
            "best_debiased_delta": best_delta,
            "closure_vs_label": 6,
        },
    }


def test_mcq_debias_pair_confirms_selection_bias_across_easy_and_challenge(tmp_path) -> None:
    payload = assess_pair(
        arc_easy_path=tmp_path / "easy.json",
        arc_easy_payload=mcq_debias_summary(
            arc_config="ARC-Easy",
            status="selection_bias_likely",
            cyclic_delta=1,
            best_delta=1,
        ),
        arc_challenge_path=tmp_path / "challenge.json",
        arc_challenge_payload=mcq_debias_summary(
            arc_config="ARC-Challenge",
            status="selection_bias_likely",
            cyclic_delta=0,
            best_delta=0,
        ),
        max_debiased_gap=2,
        min_closure=3,
    )

    assert payload["kind"] == "stage5_mcq_debias_pair_assessment"
    assert payload["status"] == "mcq_selection_bias_confirmed"
    assert payload["passed"] is True
    assert payload["blocking_summary"] is None


def test_mcq_debias_pair_routes_residual_content_gap_to_blocking_summary(tmp_path) -> None:
    challenge_path = tmp_path / "challenge.json"
    payload = assess_pair(
        arc_easy_path=tmp_path / "easy.json",
        arc_easy_payload=mcq_debias_summary(
            arc_config="ARC-Easy",
            status="selection_bias_likely",
            cyclic_delta=1,
            best_delta=1,
        ),
        arc_challenge_path=challenge_path,
        arc_challenge_payload=mcq_debias_summary(
            arc_config="ARC-Challenge",
            status="content_degradation_persists",
            cyclic_delta=-4,
            best_delta=-4,
        ),
        max_debiased_gap=2,
        min_closure=3,
    )

    assert payload["status"] == "mcq_content_gap_persists"
    assert payload["passed"] is False
    assert payload["blocking_summary"] == str(challenge_path).replace("\\", "/")


def test_mcq_scoring_policy_activates_from_confirmed_pair_and_flags_stale_label_artifacts(
    tmp_path, monkeypatch
) -> None:
    outputs = tmp_path / "outputs" / "stage5" / "old_balanced"
    outputs.mkdir(parents=True)
    (outputs / "summary.json").write_text(
        json.dumps(
            {
                "kind": "stage5_balanced_mcq_checkpoint_assessment",
                "score_target": "label",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(scoring_policy, "ROOT", tmp_path)

    payload = scoring_policy.build_policy(
        source_summary=tmp_path / "pair" / "summary.json",
        source_payload={
            "kind": "stage5_mcq_debias_pair_assessment",
            "status": "mcq_selection_bias_confirmed",
        },
    )

    assert payload["kind"] == "stage5_mcq_scoring_policy"
    assert payload["status"] == "debiased_mcq_policy_active"
    assert payload["passed"] is True
    assert payload["primary_metrics"] == ["cyclic_label_aggregated", "content_question_only"]
    assert payload["diagnostic_only_metrics"] == ["label"]
    assert payload["stale_label_only_artifacts"][0]["summary"] == "outputs/stage5/old_balanced/summary.json"
