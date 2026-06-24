import json
import math

import pytest
import torch

from eval.eval_effective_pathways import aggregate, project_states, q_values, read_prompts


def test_read_prompts_accepts_prompt_question_and_text(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "prompt field"}),
                json.dumps({"question": "question field"}),
                json.dumps({"text": "text field"}),
            ]
        ),
        encoding="utf-8",
    )

    assert read_prompts(str(path)) == ["prompt field", "question field", "text field"]
    assert read_prompts(str(path), limit=2) == ["prompt field", "question field"]


def test_read_prompts_rejects_rows_without_prompt(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"answer": "42"}), encoding="utf-8")

    with pytest.raises(ValueError, match="no prompt/question/text"):
        read_prompts(str(path))


def test_q_values_parses_infinity():
    values = q_values("0,1,2,inf,infinity")

    assert values[:3] == [0.0, 1.0, 2.0]
    assert math.isinf(values[3])
    assert math.isinf(values[4])


def test_aggregate_preserves_effective_pathway_orders():
    records = [
        {
            "initial_pairwise_distance": 2.0,
            "final_pairwise_distance": 1.0,
            "spread_ratio_final_over_initial": 0.5,
            "lyapunov_proxy_per_loop": -0.1,
            "unique_next_token_argmax": 1,
            "effective_pathways": {"0": 2.0, "1": 1.5, "2": 1.25, "inf": 1.0},
        },
        {
            "initial_pairwise_distance": 4.0,
            "final_pairwise_distance": 8.0,
            "spread_ratio_final_over_initial": 2.0,
            "lyapunov_proxy_per_loop": 0.2,
            "unique_next_token_argmax": 3,
            "effective_pathways": {"0": 4.0, "1": 3.0, "2": 2.0, "inf": 1.5},
        },
    ]

    out = aggregate(records, [0.0, 1.0, 2.0, math.inf])

    assert out["prompts"] == 2
    assert out["mean_initial_pairwise_distance"] == pytest.approx(3.0)
    assert out["mean_final_pairwise_distance"] == pytest.approx(4.5)
    assert out["mean_unique_next_token_argmax"] == pytest.approx(2.0)
    assert out["mean_effective_pathways"] == pytest.approx(
        {"0": 3.0, "1": 2.25, "2": 1.625, "inf": 1.25}
    )


def test_project_states_uses_saved_projection(tmp_path):
    projection_path = tmp_path / "projection.pt"
    projection = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    torch.save({"projection": projection}, projection_path)
    states = torch.tensor([[2.0, 3.0, 4.0, 5.0]])

    out = project_states(states, str(projection_path), 2)

    assert out.shape == (1, 2)
    assert out.tolist() == [[7.0, 8.0]]

