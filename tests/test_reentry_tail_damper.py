from __future__ import annotations

import torch

from models.reentry_tail_damper import apply_tail_damper, strength_scaled_damper


def test_strength_scaled_damper_interpolates_in_log_space() -> None:
    scale = torch.tensor([0.25, 1.0])

    assert torch.allclose(strength_scaled_damper(scale, 0.0), torch.ones_like(scale))
    assert torch.allclose(strength_scaled_damper(scale, 1.0), scale)
    assert torch.allclose(strength_scaled_damper(scale, 0.5), torch.tensor([0.5, 1.0]))


def test_apply_tail_damper_only_changes_tail_subspace() -> None:
    hidden = torch.tensor([[[10.0, 4.0, 3.0]]])
    mean = torch.zeros(3)
    basis = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    scale = torch.tensor([0.5, 1.0])

    out = apply_tail_damper(hidden, mean=mean, basis=basis, damper_scale=scale, strength=1.0)

    assert torch.allclose(out, torch.tensor([[[10.0, 2.0, 3.0]]]))

