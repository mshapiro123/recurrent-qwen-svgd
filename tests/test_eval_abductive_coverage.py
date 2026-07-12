from __future__ import annotations

import torch

from eval.eval_abductive_coverage import (
    parse_sample_counts,
    read_target_entropies,
    sample_names,
    score_sample_prefix,
    summarize_rows,
)


def test_sample_count_parser_sorts_and_deduplicates() -> None:
    assert parse_sample_counts("8,1,4,4,2") == [1, 2, 4, 8]


def test_sample_names_is_seed_reproducible() -> None:
    scores = {"A": 1.0, "B": 0.5, "C": -1.0}
    first = sample_names(
        scores,
        count=20,
        temperature=0.7,
        generator=torch.Generator().manual_seed(19),
    )
    second = sample_names(
        scores,
        count=20,
        temperature=0.7,
        generator=torch.Generator().manual_seed(19),
    )

    assert first == second


def test_prefix_score_counts_only_unique_valid_solutions() -> None:
    scored = score_sample_prefix(["A", "A", "B", "X"], {"A", "B", "C"})

    assert scored["valid_samples"] == 3
    assert scored["unique_valid"] == ["A", "B"]
    assert scored["coverage"] == 2 / 3
    assert scored["full_coverage"] is False


def test_summary_reports_validity_coverage_and_duplicates() -> None:
    rows = [
        {
            "depth": 2,
            "coverage_denominator": 2,
            "greedy_valid": True,
            "sampling": {
                "2": {
                    "valid_samples": 2,
                    "unique_samples": 2,
                    "unique_valid_count": 2,
                    "coverage": 1.0,
                    "full_coverage": True,
                }
            },
        },
        {
            "depth": 2,
            "coverage_denominator": 2,
            "greedy_valid": False,
            "sampling": {
                "2": {
                    "valid_samples": 1,
                    "unique_samples": 1,
                    "unique_valid_count": 1,
                    "coverage": 0.5,
                    "full_coverage": False,
                }
            },
        },
    ]

    summary = summarize_rows(rows, [2])

    assert summary["overall"]["greedy_valid_rate"] == 0.5
    assert summary["overall"]["sampling"]["2"]["valid_sample_rate"] == 0.75
    assert summary["overall"]["sampling"]["2"]["mean_coverage"] == 0.75
    assert summary["overall"]["sampling"]["2"]["duplicate_rate"] == 0.25


def test_target_entropy_reader_uses_explicit_candidate_entropy(tmp_path) -> None:
    path = tmp_path / "entropies.jsonl"
    path.write_text(
        '{"id":"row-1","latent_candidate_entropy":1.25}\n',
        encoding="utf-8",
    )

    assert read_target_entropies(path.as_posix(), field="latent_candidate_entropy") == {"row-1": 1.25}
