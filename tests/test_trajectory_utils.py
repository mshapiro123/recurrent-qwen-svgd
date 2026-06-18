import torch

from models.trajectory_utils import average_pairwise_cosine_distance, repeat_for_trajectories, unflatten_trajectories


def test_repeat_and_unflatten_trajectories():
    x = torch.arange(6).view(2, 3)
    repeated = repeat_for_trajectories(x, 2)
    assert repeated.tolist() == [[0, 1, 2], [0, 1, 2], [3, 4, 5], [3, 4, 5]]
    restored = unflatten_trajectories(repeated, batch_size=2, num_trajectories=2)
    assert restored.shape == (2, 2, 3)


def test_average_pairwise_cosine_distance_zero_for_one_trajectory():
    pooled = torch.randn(2, 1, 4)
    assert average_pairwise_cosine_distance(pooled).item() == 0.0


def test_average_pairwise_cosine_distance_uses_fp32_for_small_bf16_deltas():
    pooled = torch.ones(1, 2, 256, dtype=torch.bfloat16)
    pooled[0, 1, 0] = 1.0625
    distance = average_pairwise_cosine_distance(pooled)
    assert distance.dtype == torch.float32
    assert distance.item() > 0.0
