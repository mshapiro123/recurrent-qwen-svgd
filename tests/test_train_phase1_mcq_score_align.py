import torch
import torch.nn.functional as F

from training.train_phase1_mcq_score_align import (
    encode_options,
    halting_target_nll,
    option_distribution_kl,
    option_scores_from_logits,
)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, **_kwargs):
        return {"input_ids": [101] + [(ord(char) % 50) + 2 for char in text]}


class FakeOutput:
    def __init__(self, halting_weights):
        self.halting_weights = halting_weights


def test_encode_options_masks_prompt_and_keeps_target_index() -> None:
    row = {
        "id": "q1",
        "question": "Pick one.",
        "choices": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        "answer": "C",
        "prompt_style": "question_only",
        "score_target": "option_text",
        "target_loop_count": 9,
        "routing_type": "score_content_align",
    }

    encoded = encode_options(
        row,
        FakeTokenizer(),
        max_length=256,
        pad_token_id=0,
        default_prompt_style="with_options",
        default_score_target="label",
        max_train_loops=4,
    )

    assert encoded.input_ids.shape[0] == 4
    assert encoded.attention_mask.shape == encoded.input_ids.shape
    assert encoded.labels.shape == encoded.input_ids.shape
    assert encoded.target_index == 2
    assert encoded.target_loop_count == 4
    assert encoded.routing_type == "score_content_align"
    assert encoded.labels[2].ne(-100).any()
    assert encoded.labels[2, 0].item() == -100


def test_option_scores_from_logits_uses_only_completion_labels() -> None:
    labels = torch.tensor(
        [
            [-100, 2, 3],
            [-100, 2, 4],
        ]
    )
    logits = torch.full((2, 3, 6), -5.0)
    # sequence_logprobs scores labels shifted one token left, so labels[:, 1]
    # is scored from logits[:, 0] and labels[:, 2] from logits[:, 1].
    logits[0, 0, 2] = 5.0
    logits[0, 1, 3] = 5.0
    logits[1, 0, 2] = 5.0
    logits[1, 1, 4] = 1.0

    scores = option_scores_from_logits(logits, labels, normalize=True)

    assert scores.shape == (2,)
    assert scores[0] > scores[1]


def test_halting_target_nll_clamps_target_to_loop_range() -> None:
    output = FakeOutput(torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]))

    loss = halting_target_nll(output, target_loop_count=8, max_loops=3)

    assert loss is not None
    assert torch.isfinite(loss)


def test_option_distribution_kl_treats_options_as_one_distribution() -> None:
    student = torch.tensor([0.0, 1.0, 2.0, 3.0])
    teacher = torch.tensor([3.0, 2.0, 1.0, 0.0])

    actual = option_distribution_kl(student, teacher, temperature=2.0)
    expected = F.kl_div(
        F.log_softmax((student / 2.0).unsqueeze(0), dim=-1),
        F.softmax((teacher / 2.0).unsqueeze(0), dim=-1),
        reduction="batchmean",
    ) * 4.0
    wrong_options_as_batch = F.kl_div(
        F.log_softmax(student / 2.0, dim=-1),
        F.softmax(teacher / 2.0, dim=-1),
        reduction="batchmean",
    ) * 4.0

    assert torch.allclose(actual, expected)
    assert actual > wrong_options_as_batch
