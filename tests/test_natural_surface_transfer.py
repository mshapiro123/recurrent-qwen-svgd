from __future__ import annotations

import json
from pathlib import Path

from training.natural_surface_transfer import (
    HELDOUT_NAME_SYMBOLS,
    NAME_SYMBOLS,
    NaturalSurfaceConfig,
    assert_verbal_row_invariants,
    build_paired_verbal_rows,
    build_verbal_rows,
    manifest_for_rows,
    verify_single_token_names,
    write_natural_surface_dataset,
)


def test_relay_rows_pin_depth_with_day_counter_and_step_targets() -> None:
    rows = build_verbal_rows(
        family="relay",
        split="test",
        n_symbols=20,
        max_depth=4,
        rows_per_depth=2,
        seed=123,
        max_target_loops=4,
    )

    row = next(item for item in rows if item["depth"] == 4)
    assert "Each day it is passed once" in row["question"]
    assert "After 4 days" in row["question"]
    assert len(row["step_sentences"]) == 4
    assert row["latent_targets"] == row["orbit"][1:5]
    assert row["completion"] == f" {row['orbit'][4]}"
    assert row["target_loop_count"] == 4
    assert row["score_target"] == "full_symbols"
    assert row["prompt_style"] == "question_only"
    assert row["verbal_surface_family"] == "relay"
    assert_verbal_row_invariants(row)


def test_pointer_rows_are_heldout_template_with_hop_counter() -> None:
    rows = build_verbal_rows(
        family="pointer",
        split="test",
        n_symbols=20,
        max_depth=3,
        rows_per_depth=1,
        seed=456,
        max_target_loops=3,
    )

    row = rows[-1]
    assert "follow the notes exactly 3 times" in row["question"]
    assert "After hop 3" in row["step_sentences"][-1]
    assert row["latent_targets"] == row["orbit"][1:4]
    assert row["verbal_surface_family"] == "pointer"
    assert_verbal_row_invariants(row)


def test_baton_third_family_uses_handoff_surface() -> None:
    rows = build_verbal_rows(
        family="baton",
        split="test",
        n_symbols=20,
        max_depth=2,
        rows_per_depth=1,
        seed=457,
        max_target_loops=2,
    )

    row = rows[-1]
    assert "baton" in row["question"]
    assert "handoff" in row["question"] or "handed" in row["question"]
    assert row["verbal_surface_family"] == "baton"
    assert_verbal_row_invariants(row)


def test_unseen_name_rows_carry_row_level_symbol_names() -> None:
    rows = build_verbal_rows(
        family="relay",
        split="test",
        n_symbols=20,
        max_depth=2,
        rows_per_depth=1,
        seed=458,
        max_target_loops=2,
        symbol_names=HELDOUT_NAME_SYMBOLS,
    )

    row = rows[-1]
    assert row["symbol_names"] == list(HELDOUT_NAME_SYMBOLS)
    assert set(row["orbit"]).issubset(set(HELDOUT_NAME_SYMBOLS))
    assert not set(row["orbit"]) & set(NAME_SYMBOLS)
    assert_verbal_row_invariants(row)


def test_paired_rows_share_instance_and_answer_across_surfaces() -> None:
    paired = build_paired_verbal_rows(
        families=("relay", "pointer"),
        split="test",
        n_symbols=20,
        max_depth=3,
        rows_per_depth=2,
        seed=459,
        max_target_loops=3,
    )

    relay = paired["relay"][-1]
    pointer = paired["pointer"][-1]
    assert relay["paired_instance_id"] == pointer["paired_instance_id"]
    assert relay["orbit"] == pointer["orbit"]
    assert relay["target"] == pointer["target"]
    assert relay["question"] != pointer["question"]
    assert relay["verbal_surface_family"] == "relay"
    assert pointer["verbal_surface_family"] == "pointer"


def test_natural_surface_generation_is_deterministic_and_manifested(tmp_path: Path) -> None:
    config = NaturalSurfaceConfig(
        n_symbols=20,
        train_max_depth=3,
        eval_max_depth=4,
        train_rows_per_depth=3,
        val_rows_per_depth=2,
        eval_rows_per_depth=2,
        seed=789,
        max_target_loops=4,
    )

    first = write_natural_surface_dataset(output_dir=tmp_path / "first", config=config)
    second = write_natural_surface_dataset(output_dir=tmp_path / "second", config=config)

    assert first["kind"] == "stage5_natural_surface_transfer_dataset"
    assert first["status"] == "finished"
    assert first["manifests"] == second["manifests"]
    assert first["manifests"]["train_relay_chain_symbol_sft"]["depth_counts"] == {"1": 3, "2": 3, "3": 3}
    assert first["manifests"]["relay_test_chain_mcq"]["depth_counts"] == {"1": 2, "2": 2, "3": 2, "4": 2}
    assert first["manifests"]["pointer_test_chain_mcq"]["depth_counts"] == {"1": 2, "2": 2, "3": 2, "4": 2}
    assert first["manifests"]["synthetic_rehearsal_chain_symbol_sft"]["rows"] == 9
    assert first["manifests"]["rung0_train_mix_chain_symbol_sft"]["rows"] == 18

    relay_row = json.loads((tmp_path / "first" / "relay_test_chain_mcq.jsonl").read_text().splitlines()[0])
    pointer_row = json.loads((tmp_path / "first" / "pointer_test_chain_mcq.jsonl").read_text().splitlines()[0])
    assert relay_row["verbal_surface_family"] == "relay"
    assert pointer_row["verbal_surface_family"] == "pointer"
    assert first["manifests"]["relay_test_chain_mcq"] == manifest_for_rows(
        [json.loads(line) for line in (tmp_path / "first" / "relay_test_chain_mcq.jsonl").read_text().splitlines()]
    )


def test_single_token_name_verifier_accepts_fake_qwen_like_tokenizer() -> None:
    class FakeTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool = False):
            token = text.strip()
            assert token in NAME_SYMBOLS
            return {"input_ids": [NAME_SYMBOLS.index(token) + 100]}

    verdict = verify_single_token_names(FakeTokenizer(), n_symbols=20)

    assert verdict["all_single_token"] is True
    assert len(verdict["rows"]) == 20
    assert all(row["bare_token_count"] == 1 for row in verdict["rows"])
    assert all(row["space_prefixed_token_count"] == 1 for row in verdict["rows"])
