from eval.eval_jsonl import parse_group_by_fields


def test_parse_group_by_fields_accepts_repeated_and_csv_values() -> None:
    assert parse_group_by_fields(["curriculum_mode,target_loop_count", "routing_type"]) == [
        "curriculum_mode",
        "target_loop_count",
        "routing_type",
    ]


def test_parse_group_by_fields_ignores_empty_values() -> None:
    assert parse_group_by_fields(["curriculum_mode,, ", ""]) == ["curriculum_mode"]
