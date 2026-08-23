import numpy as np
import torch

from analysis.analyze_paper2_stage2bs_desk_math import (
    fit_common_c,
    infer_theta,
    mp_spectrum,
    recurrence_cv,
    required_n,
)


def test_common_c_recovers_shared_affine_drift() -> None:
    rng = np.random.default_rng(20260823)
    first = rng.normal(2.5, 0.5, size=64)
    rates = rng.uniform(0.35, 0.55, size=64)
    margins = [first]
    for _ in range(3):
        margins.append(rates * margins[-1] - 0.2)
    matrix = np.stack(margins, axis=1)
    recovered_r, recovered_c = fit_common_c(matrix, (0, 1, 2))
    assert np.max(np.abs(recovered_r - rates)) < 1e-8
    assert abs(recovered_c + 0.2) < 1e-8
    comparison = recurrence_cv(matrix)
    assert comparison["common_c"]["rmse"] < 1e-8


def test_mp_spectrum_detects_rank_one_spike() -> None:
    generator = torch.Generator().manual_seed(20260823)
    noise = torch.randn(128, 896, generator=generator)
    left = torch.randn(128, generator=generator)
    right = torch.randn(896, generator=generator)
    spike = 80.0 * torch.outer(left / left.norm(), right / right.norm())
    receipt = mp_spectrum(noise + spike)
    assert receipt["shape"] == [128, 896]
    assert receipt["outlier_spikes"] >= 1


def test_bbp_inversion_and_target_rows_are_monotone() -> None:
    theta = infer_theta(aspect=0.6, cosine=0.15)
    assert theta is not None
    threshold = int(896 / theta**4) + 1
    rows_25 = required_n(896, theta, 0.25)
    rows_50 = required_n(896, theta, 0.5)
    rows_75 = required_n(896, theta, 0.75)
    assert threshold <= rows_25 < rows_50 < rows_75
