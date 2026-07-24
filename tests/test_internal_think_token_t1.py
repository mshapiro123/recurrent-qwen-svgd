from __future__ import annotations

import pytest

from training.internal_think_token_t1 import (
    augment_control_row,
    build_candidate_trie_contract,
    build_pilot_mixture_rows,
    candidate_trie_edges,
    class_weights_from_ratio,
    control_targets_for_depth,
    gate3_verdict,
    gather_control_examples,
    locate_readout_positions,
    pilot_grid,
    score_control_predictions,
    select_pilot_cell,
)


class _TwoTokenSymbolTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = True) -> dict[str, list[int]]:
        del add_special_tokens
        prompt = "Question\nAnswer:"
        if text == prompt:
            return {"input_ids": [1, 2, 3]}
        if text.startswith(prompt + " "):
            value = int(text[len(prompt) + 1 :])
            suffix = (
                [220, 1000 + value]
                if value < 10
                else [220, 1010, 1000 + value - 10]
            )
            return {"input_ids": [1, 2, 3, *suffix]}
        raise AssertionError(text)


def test_candidate_trie_contract_supports_variable_length_sequences() -> None:
    contract = build_candidate_trie_contract(
        _TwoTokenSymbolTokenizer(),
        prompt="Question\nAnswer:",
        candidate_values=range(16),
    )
    assert contract.prompt_token_count == 3
    assert contract.candidate_token_ids[0] == (220, 1000)
    assert contract.candidate_token_ids[10] == (220, 1010, 1000)
    assert contract.scoring_prefixes == ((), (220,), (220, 1010))
    edges = candidate_trie_edges(contract)
    assert edges[()] == tuple((index, 220) for index in range(16))
    assert (10, 1010) in edges[(220,)]
    assert edges[(220, 1010)] == tuple((index, 990 + index) for index in range(10, 16))


def test_candidate_trie_batch_scores_exact_normalized_sequences() -> None:
    torch = pytest.importorskip("torch")
    from training.internal_think_token_t1 import CandidateTrieContract
    from training.run_internal_think_token_p0_cell import score_candidate_trie_batch

    contract = CandidateTrieContract(
        prompt_token_count=2,
        candidate_values=("A", "B"),
        candidate_token_ids=((10, 20), (10, 30, 40)),
        scoring_prefixes=((), (10,), (10, 30)),
    )

    class _Output:
        def __init__(self, loop_logits):
            self.loop_logits = loop_logits

    class _Wrapper:
        def __call__(self, *, input_ids, max_loops, **kwargs):
            del kwargs
            batch_size, sequence_length = input_ids.shape
            logits = torch.full(
                (batch_size, 1, max_loops, sequence_length, 64),
                -20.0,
            )
            for row in range(batch_size):
                tokens = input_ids[row].tolist()
                if tokens[-2:] == [10, 30]:
                    logits[row, 0, :, sequence_length - 1, 40] = 0.0
                elif tokens[-1:] == [10]:
                    logits[row, 0, :, sequence_length - 1, 20] = -6.0
                    logits[row, 0, :, sequence_length - 1, 30] = 0.0
                else:
                    raise AssertionError(tokens)
            return _Output(logits)

    batch = {
        "input_ids": torch.tensor([[1, 2, 10, 30, 40]]),
        "required_depth": torch.tensor([1]),
    }
    root_logits = torch.full((1, 64), -20.0)
    root_logits[0, 10] = 0.0
    scores = score_candidate_trie_batch(
        _Wrapper(),
        batch,
        root_next_logits=root_logits,
        answer_starts=torch.tensor([2]),
        candidate_contract=contract,
        pad_token_id=0,
        device="cpu",
        max_loops=1,
    )

    assert scores.shape == (1, 2)
    assert int(scores.argmax(dim=-1).item()) == 1
    assert float(scores[0, 1]) > float(scores[0, 0])


def test_p0_loop_target_alignment_preflight_rejects_length_drift() -> None:
    pytest.importorskip("torch")
    from training.run_internal_think_token_p0_cell import assert_loop_completion_alignment

    class _Tokenizer:
        def __call__(self, text: str, **kwargs):
            del kwargs
            extra = 1 if text.endswith(" C") else 0
            return {"input_ids": list(range(len(text.split()) + extra))}

    aligned = [{
        "id": "aligned",
        "prompt": "Question\nAnswer:",
        "completion": " B",
        "loop_completions": [" A", " B"],
    }]
    receipt = assert_loop_completion_alignment(aligned, _Tokenizer(), max_length=512)
    assert receipt == {"rows": 1, "loop_targets": 2, "all_position_aligned": True}

    misaligned = [{**aligned[0], "loop_completions": [" C", " B"]}]
    with pytest.raises(AssertionError, match="not position-aligned"):
        assert_loop_completion_alignment(misaligned, _Tokenizer(), max_length=512)


def test_pilot_grid_is_locked_nine_cells_plus_reference() -> None:
    cells = pilot_grid()

    assert len(cells) == 10
    assert cells[0].cell_id == "lambda0_reference"
    assert cells[0].control_loss_lambda == 0.0
    assert {
        (cell.control_loss_lambda, cell.stop_to_continue_ratio)
        for cell in cells[1:]
    } == {
        (0.5, 1.0),
        (0.5, 3.5),
        (0.5, 7.0),
        (1.0, 1.0),
        (1.0, 3.5),
        (1.0, 7.0),
        (2.0, 1.0),
        (2.0, 3.5),
        (2.0, 7.0),
    }


def test_control_targets_stop_exactly_at_required_depth() -> None:
    assert control_targets_for_depth(1, max_loops=8) == [1]
    assert control_targets_for_depth(4, max_loops=8) == [0, 0, 0, 1]
    with pytest.raises(ValueError, match="within"):
        control_targets_for_depth(9, max_loops=8)


def test_class_weights_preserve_ratio_and_realized_mean_one() -> None:
    weights = class_weights_from_ratio(
        stop_to_continue_ratio=3.5,
        continue_count=28,
        stop_count=8,
    )

    assert weights[1] / weights[0] == pytest.approx(3.5)
    assert (28 * weights[0] + 8 * weights[1]) / 36 == pytest.approx(1.0)


def test_control_prompt_inserts_readout_before_answer_without_changing_targets() -> None:
    row = {
        "instance_id": "train_d03_00000",
        "prompt": "Question text\nAnswer:",
        "completion": " 7",
        "loop_completions": [" 2", " 5", " 7"],
        "depth": 3,
        "target_loop_count": 3,
    }

    augmented = augment_control_row(row)

    assert "<|recur_readout|>" in augmented["prompt"]
    assert augmented["prompt"].endswith("Answer:")
    assert augmented["completion"] == row["completion"]
    assert augmented["loop_completions"] == row["loop_completions"]
    assert augmented["control_targets"] == [0, 0, 1]
    assert augmented["control_active"] is True


def test_pilot_mixture_is_exactly_balanced_and_seventy_thirty() -> None:
    source = []
    for depth in range(1, 9):
        for index in range(200):
            source.append(
                {
                    "instance_id": f"d{depth}_{index}",
                    "prompt": "Q\nAnswer:",
                    "completion": " 1",
                    "loop_completions": [" 1"] * depth,
                    "depth": depth,
                    "target_loop_count": depth,
                }
            )

    rows, manifest = build_pilot_mixture_rows(source, seed=9999)

    assert len(rows) == 2000
    assert manifest["control_rows"] == 1400
    assert manifest["rehearsal_rows"] == 600
    assert manifest["control_fraction"] == pytest.approx(0.70)
    assert all(cell == {"control": 175, "rehearsal": 75} for cell in manifest["by_depth"].values())


def test_gate3_requires_every_depth_and_pooled_count() -> None:
    passing = {str(depth): {"correct": 116, "total": 128} for depth in range(1, 9)}
    passing["1"]["correct"] = 115
    verdict = gate3_verdict(passing)
    assert verdict["passed"] is True
    assert verdict["pooled_correct"] == 927

    depth_fail = {key: dict(value) for key, value in passing.items()}
    depth_fail["4"]["correct"] = 114
    assert gate3_verdict(depth_fail)["passed"] is False

    pooled_fail = {str(depth): {"correct": 115, "total": 128} for depth in range(1, 9)}
    assert gate3_verdict(pooled_fail)["pooled_correct"] == 920
    assert gate3_verdict(pooled_fail)["passed"] is False


def test_pilot_selection_filters_recalls_then_minimizes_answer_drop() -> None:
    results = [
        {
            "cell_id": "lambda0_reference",
            "control_loss_lambda": 0.0,
            "stop_to_continue_ratio": 1.0,
            "step_1500": {"stop_recall": 0.1, "continue_recall": 0.9, "answer_accuracy": 0.90},
        },
        {
            "cell_id": "lambda1_ratio3p5",
            "control_loss_lambda": 1.0,
            "stop_to_continue_ratio": 3.5,
            "step_1500": {"stop_recall": 0.8, "continue_recall": 0.8, "answer_accuracy": 0.88},
        },
        {
            "cell_id": "lambda2_ratio7",
            "control_loss_lambda": 2.0,
            "stop_to_continue_ratio": 7.0,
            "step_1500": {"stop_recall": 0.9, "continue_recall": 0.7, "answer_accuracy": 0.90},
        },
    ]

    selected = select_pilot_cell(results)

    assert selected["status"] == "selected"
    assert selected["selected_cell_id"] == "lambda2_ratio7"
    assert selected["reference_answer_accuracy"] == pytest.approx(0.90)


def test_pilot_selection_refuses_silent_extension_when_no_cell_qualifies() -> None:
    results = [
        {
            "cell_id": "lambda0_reference",
            "control_loss_lambda": 0.0,
            "stop_to_continue_ratio": 1.0,
            "step_1500": {"stop_recall": 0.0, "continue_recall": 1.0, "answer_accuracy": 0.9},
        },
        {
            "cell_id": "weak",
            "control_loss_lambda": 1.0,
            "stop_to_continue_ratio": 3.5,
            "step_1500": {"stop_recall": 0.59, "continue_recall": 0.99, "answer_accuracy": 0.9},
        },
    ]

    assert select_pilot_cell(results)["status"] == "no_qualifying_cell_reassess_before_lock"


def test_readout_positions_require_exactly_one_token_on_active_rows() -> None:
    torch = pytest.importorskip("torch")
    input_ids = torch.tensor([[3, 9, 4], [9, 2, 0], [1, 2, 3]])
    active = torch.tensor([True, True, False])

    assert locate_readout_positions(input_ids, readout_token_id=9, control_active=active).tolist() == [1, 0, -1]

    with pytest.raises(AssertionError, match="exactly one"):
        locate_readout_positions(
            torch.tensor([[9, 9, 2]]),
            readout_token_id=9,
            control_active=torch.tensor([True]),
        )


def test_control_gather_uses_only_active_transitions_through_stop() -> None:
    torch = pytest.importorskip("torch")
    # [batch, trajectory, loop, sequence, vocab]
    loop_logits = torch.arange(2 * 1 * 4 * 3 * 7, dtype=torch.float32).reshape(2, 1, 4, 3, 7)
    gathered, targets, rows, loops = gather_control_examples(
        loop_logits,
        readout_positions=torch.tensor([1, -1]),
        required_depths=torch.tensor([3, 4]),
        control_active=torch.tensor([True, False]),
        continue_token_id=5,
        stop_token_id=6,
    )

    assert gathered.shape == (3, 2)
    assert targets.tolist() == [0, 0, 1]
    assert rows.tolist() == [0, 0, 0]
    assert loops.tolist() == [1, 2, 3]
    assert gathered[0].tolist() == loop_logits[0, 0, 0, 1, [5, 6]].tolist()


def test_control_scoring_reports_recall_and_exact_selected_depth() -> None:
    rows = [
        {"row_id": "a", "depth": 2, "predictions": [0, 1]},
        {"row_id": "b", "depth": 3, "predictions": [0, 0, 1]},
        {"row_id": "c", "depth": 3, "predictions": [0, 1, 0]},
        {"row_id": "d", "depth": 2, "predictions": [0, 0]},
    ]

    result = score_control_predictions(rows, max_loops=3)

    assert result["stop_recall"] == pytest.approx(0.5)
    assert result["continue_recall"] == pytest.approx(5 / 6)
    assert result["exact_selected_depth_accuracy"] == pytest.approx(0.5)
    assert result["exhausted_without_stop"] == 1
    assert result["by_depth"]["2"]["correct"] == 1
    assert result["by_depth"]["3"]["correct"] == 1
