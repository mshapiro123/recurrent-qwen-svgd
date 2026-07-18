from eval.analyze_phase_g_multimodal_supervision import analyze


def row(
    row_id: str,
    *,
    question: str,
    target: str,
    reachable: list[str],
) -> dict:
    return {
        "id": row_id,
        "depth": 1,
        "question": question,
        "start": "S",
        "successors": {"S": reachable},
        "target": target,
        "reachable_symbols": reachable,
        "reachable_set_stratum": "2",
    }


def cache(row_id: str, prior: list[str], teacher: list[str]) -> dict:
    def arm(predictions: list[str]) -> dict:
        return {
            "1": {"predictions": predictions[:1]},
            "2": {"predictions": predictions[:2]},
        }

    return {
        "id": row_id,
        "arms": {
            "prior": arm(prior),
            "posterior_teacher": arm(teacher),
        },
    }


def test_audit_detects_repeated_targets_and_teacher_target_signal() -> None:
    train = [
        row("a1", question="same", target="A", reachable=["A", "B"]),
        row("a2", question="same", target="B", reachable=["A", "B"]),
        row("b1", question="unique", target="A", reachable=["A", "B"]),
    ]
    test = [
        row("t1", question="test 1", target="A", reachable=["A", "B"]),
        row("t2", question="test 2", target="B", reachable=["A", "B"]),
    ]
    cached = [
        cache("t1", prior=["B", "B"], teacher=["A", "A"]),
        cache("t2", prior=["A", "A"], teacher=["B", "A"]),
    ]

    summary = analyze(train, test, cached)

    exposure = summary["curriculum_exposure"]
    assert exposure["problem_groups"] == 2
    assert exposure["groups_with_multiple_targets"] == 1
    assert exposure["max_distinct_targets_per_problem"] == 2
    fidelity = summary["posterior_target_fidelity"]["by_k"]["2"]
    assert fidelity["prior"]["target_in_k_rate"] == 0.0
    assert fidelity["posterior_teacher"]["target_in_k_rate"] == 1.0
    assert fidelity["paired_target_in_k"] == {"helped": 2, "hurt": 0, "tied": 0}
    assert summary["interpretation"] == "posterior_target_signal_present"


def test_audit_names_single_target_per_problem_curriculum() -> None:
    train = [
        row("a1", question="one", target="A", reachable=["A", "B"]),
        row("b1", question="two", target="B", reachable=["A", "B"]),
    ]
    test = [row("t1", question="test", target="A", reachable=["A", "B"])]
    cached = [cache("t1", prior=["A", "A"], teacher=["A", "B"])]

    summary = analyze(train, test, cached)

    assert summary["curriculum_exposure"]["groups_with_repeated_prompt"] == 0
    assert summary["curriculum_exposure"]["groups_with_multiple_targets"] == 0
    assert summary["interpretation"] == "single_target_per_problem_curriculum"


def test_audit_measures_posterior_control_across_same_prompt_target_variants() -> None:
    train = [
        row("a1", question="same", target="A", reachable=["A", "B"]),
        row("a2", question="same", target="B", reachable=["A", "B"]),
    ]
    test = [
        row("t1", question="shared", target="A", reachable=["A", "B"]),
        row("t2", question="shared", target="B", reachable=["A", "B"]),
    ]
    cached = [
        cache("t1", prior=["A", "A"], teacher=["A", "A"]),
        cache("t2", prior=["A", "A"], teacher=["B", "B"]),
    ]

    summary = analyze(train, test, cached)

    control = summary["posterior_target_conditioning"]
    assert control["multi_target_groups"] == 1
    assert control["prior"]["mean_distinct_first_predictions"] == 1.0
    assert control["posterior_teacher"]["mean_distinct_first_predictions"] == 2.0
    assert control["prior"]["all_variants_match_target_rate"] == 0.0
    assert control["posterior_teacher"]["all_variants_match_target_rate"] == 1.0
    assert control["prior"]["mean_group_selected_target_rate"] == 0.5
    assert control["posterior_teacher"]["mean_group_selected_target_rate"] == 1.0
    assert control["paired_group_selected_target_rate"] == {
        "helped": 1,
        "hurt": 0,
        "tied": 0,
        "posterior_minus_prior_mean": 0.5,
    }
