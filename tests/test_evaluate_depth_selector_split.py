from eval.evaluate_depth_selector_split import evaluate_candidate, split_examples, summarize_many_splits


def example(row_id, answer="A", loop1_hit=False, loop2_hit=True):
    return {
        "id": row_id,
        "answer": answer,
        "base_hit": False,
        "loop_hits": {1: loop1_hit, 2: loop2_hit},
        "loop_scores": {
            1: {"A": 0.1, "B": 1.0},
            2: {"A": 3.0, "B": 0.0},
        },
        "loop_predicted_margins": {1: 0.9, 2: 3.0},
        "loop_answer_margins": {1: -0.9, 2: 3.0},
        "base_predicted_margin": 0.5,
        "base_answer_margin": -0.5,
    }


def test_weighted_score_candidate_keeps_raw_method_for_reevaluation():
    examples = [example("a")]
    candidate = {
        "family": "score_selector",
        "subset": [1, 2],
        "method": "loop1_plus_weighted_deeper",
        "weight": 0.5,
    }

    first = evaluate_candidate(examples, candidate)

    assert first["method"] == "loop1_plus_weighted_deeper"
    assert first["display_method"] == "loop1_plus_weighted_deeper:0.5"
    assert first["correct"] == 1

    # This is the train/test path: a candidate selected on train is evaluated
    # again on test. The raw method must still be callable.
    second = evaluate_candidate(examples, first)

    assert second["method"] == "loop1_plus_weighted_deeper"
    assert second["display_method"] == "loop1_plus_weighted_deeper:0.5"
    assert second["correct"] == 1

    wrong_examples = [example("b", answer="B", loop1_hit=True, loop2_hit=False)]
    reevaluated = evaluate_candidate(wrong_examples, first)

    assert reevaluated["correct"] == 0
    assert reevaluated["delta_vs_loop1"] == -1


def test_stable_split_is_deterministic_and_exhaustive():
    examples = [example(str(i)) for i in range(10)]

    train1, test1 = split_examples(examples, benchmark="toy", seed=7, train_fraction=0.6)
    train2, test2 = split_examples(examples, benchmark="toy", seed=7, train_fraction=0.6)

    assert [row["id"] for row in train1] == [row["id"] for row in train2]
    assert [row["id"] for row in test1] == [row["id"] for row in test2]
    assert len(train1) == 6
    assert len(test1) == 4
    assert {row["id"] for row in train1}.isdisjoint({row["id"] for row in test1})
    assert {row["id"] for row in train1 + test1} == {str(i) for i in range(10)}


def split_payload(seed, delta, selector):
    return {
        "source_sweep_summary": "outputs/stage5/sweep/summary.json",
        "source_sweep_run_id": "sweep",
        "seed": seed,
        "train_fraction": 0.5,
        "loops": [1, 2],
        "benchmarks": {
            "toy": {
                "train": {
                    "best_selector": selector,
                },
                "test": {
                    "baseline": {
                        "total": 4,
                        "base_correct": 2,
                        "loop1_correct": 2,
                    },
                    "oracle": {
                        "any_depth_gain_vs_loop1": 2,
                    },
                    "selected_selector": {
                        "correct": 2 + delta,
                        "delta_vs_loop1": delta,
                        "wins_vs_loop1": max(delta, 0),
                        "losses_vs_loop1": max(-delta, 0),
                    },
                },
            }
        },
    }


def test_summarize_many_splits_tracks_stability_and_selector_choices():
    selector = {
        "family": "score_selector",
        "subset": [1, 2],
        "method": "max",
        "display_method": "max",
    }
    payload = summarize_many_splits(
        [
            split_payload(0, 1, selector),
            split_payload(1, 0, selector),
            split_payload(2, -1, selector),
        ]
    )

    toy = payload["benchmarks"]["toy"]

    assert toy["num_splits"] == 3
    assert toy["mean_test_delta_vs_loop1"] == 0
    assert toy["positive_delta_splits"] == 1
    assert toy["zero_delta_splits"] == 1
    assert toy["negative_delta_splits"] == 1
    assert toy["selector_signatures"] == {"score:max[1,2]": 3}
