import torch

from analysis.analyze_paper2_bicameral_w3 import (
    dm5_power,
    permutation_eta,
    projected_ridge,
)


def test_projected_ridge_is_deterministic() -> None:
    generator = torch.Generator().manual_seed(11)
    x_train = torch.randn(32, 12, generator=generator)
    weight = torch.randn(12, 7, generator=generator)
    y_train = x_train @ weight
    x_eval = torch.randn(8, 12, generator=generator)

    first = projected_ridge(x_train, y_train, x_eval, rank=7, ridge=1e-6)
    second = projected_ridge(x_train, y_train, x_eval, rank=7, ridge=1e-6)

    torch.testing.assert_close(first, second)
    assert first.shape == (8, 7)


def test_dm5_power_is_deterministic_and_cross_fitted() -> None:
    generator = torch.Generator().manual_seed(17)
    loop1 = torch.randn(256, 10, generator=generator)
    direction = torch.randn(256, 10, generator=generator)
    correction = torch.cat((loop1[:, :5], direction[:, :5]), dim=-1)

    first = dm5_power({"loop1": loop1, "direction": direction}, correction, seed=23)
    second = dm5_power({"loop1": loop1, "direction": direction}, correction, seed=23)

    assert first == second
    assert all(cell["fit_rows_per_split"] == 192 for cell in first["cells"])
    assert all(cell["eval_rows_per_split"] == 64 for cell in first["cells"])


def test_permutation_eta_detects_strong_group_association() -> None:
    values = torch.cat((torch.full((20,), -2.0), torch.full((20,), 2.0)))
    labels = ["left"] * 20 + ["right"] * 20
    receipt = permutation_eta(values, labels, draws=199, seed=29)

    assert receipt["eta_squared"] > 0.99
    assert receipt["empirical_p"] <= 0.01
