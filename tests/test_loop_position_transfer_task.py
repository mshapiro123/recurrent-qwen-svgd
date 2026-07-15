from training.loop_position_transfer_task import (
    LoopPositionConfig,
    build_eval_rows,
    build_training_rows,
    validate_loop_position_rows,
)


def test_mixed_direction_row_moves_forward_then_once_backward() -> None:
    rows = build_eval_rows(
        LoopPositionConfig(rows_per_position=2, seed=17),
        prefix_lengths=(0, 1, 2, 3),
        split="test",
    )
    assert validate_loop_position_rows(rows)["status"] == "passed"
    for row in rows:
        prefix = int(row["forward_prefix_length"])
        chain = row["chain_values"]
        assert len(chain) == prefix + 2
        mapping = {int(k): int(v) for k, v in row["mapping_values"].items()}
        for index in range(prefix):
            assert mapping[chain[index]] == chain[index + 1]
        inverse = {value: key for key, value in mapping.items()}
        assert inverse[chain[prefix]] == chain[prefix + 1]
        assert row["target_loop_count"] == prefix + 1
        assert len(row["loop_completions"]) == prefix + 1


def test_training_mix_is_exactly_30_percent_forward_rehearsal() -> None:
    rows = build_training_rows(LoopPositionConfig(train_rows=100, seed=23))
    assert len(rows) == 100
    roles = [row["curriculum_role"] for row in rows]
    assert roles.count("pure_forward_rehearsal") == 30
    assert roles.count("mixed_forward_then_inverse") == 70
    mixed_prefixes = {
        int(row["forward_prefix_length"])
        for row in rows
        if row["curriculum_role"] == "mixed_forward_then_inverse"
    }
    assert mixed_prefixes == {0, 1}
    assert validate_loop_position_rows(rows)["status"] == "passed"


def test_loop_position_splits_have_stable_disjoint_ids() -> None:
    cfg = LoopPositionConfig(rows_per_position=4, seed=31)
    first = build_eval_rows(cfg, prefix_lengths=(2, 3), split="test")
    second = build_eval_rows(cfg, prefix_lengths=(2, 3), split="test")
    train = build_eval_rows(cfg, prefix_lengths=(0, 1), split="train_eval")
    assert first == second
    assert {row["id"] for row in first}.isdisjoint({row["id"] for row in train})
