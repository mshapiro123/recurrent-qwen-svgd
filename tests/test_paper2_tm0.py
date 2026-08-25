import json
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.build_paper2_tm0_manifest import _cka_calibration_rows
from analysis.analyze_paper2_tm0_tm1_cka import debiased_linear_cka
from analysis.analyze_paper2_tm0_tm1_stitch import mp_median, rmt_whitener
from analysis.analyze_paper2_tm0_tm2 import (
    bootstrap_mean_difference,
    classify_direction_cell,
    remove_common_mode,
    subspace_overlap,
    two_half_discriminative_read,
)
from analysis.paper2_tm0_hermetic_screen import (
    character_shingles,
    minhash_signature,
    normalize_text,
)
from analysis.prepare_paper2_tm0_hermetic_screen import prompt_text
from eval.cache_paper2_tm0 import active_pools
from training.paper2_tm0 import (
    bivector_coordinates,
    deterministic_folds,
    load_lock,
    random_orthoproject,
    window_boundaries,
)


def test_tm0_lock_is_no_training_and_authorities_match() -> None:
    lock = load_lock()
    assert lock["status"] == "RATIFIED_EXECUTABLE_NO_TRAINING"
    assert lock["scope"]["training_authorized"] is False
    assert lock["scope"]["optimizer_construction_allowed"] is False
    assert lock["scope"]["confirm_scored"] is False
    assert lock["scope"]["eval_e_scored"] is False
    root = Path(__file__).resolve().parents[1]
    for authority in lock["authority"].values():
        payload = (root / "docs" / authority["filename"]).read_bytes()
        assert len(payload) == authority["bytes"]
        import hashlib

        assert hashlib.sha256(payload).hexdigest() == authority["sha256"]


def test_active_pools_use_only_active_tokens() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    pools = active_pools(hidden, mask)
    assert torch.equal(pools[:, 0], torch.tensor([[3.0, 4.0], [9.0, 10.0]]))
    assert torch.equal(pools[:, 1], torch.tensor([[2.0, 3.0], [7.0, 8.0]]))


def test_cka_subset_is_deterministic_and_includes_each_battery() -> None:
    panel = []
    for battery, count in (("a", 900), ("b", 90), ("c", 10)):
        for index in range(count):
            panel.append({"battery": battery, "item_id": f"{battery}-{index}"})
    first = _cka_calibration_rows(panel, rows=100, seed=7)
    second = _cka_calibration_rows(panel, rows=100, seed=7)
    assert first == second
    assert len(first) == 100
    assert {row["battery"] for row in first} == {"a", "b", "c"}


def test_hermetic_text_contract_is_prompt_only_and_deterministic() -> None:
    row = {
        "answer": "B",
        "prompt": {
            "question": "  Which\u00a0one? ",
            "choice_labels": ["A", "B"],
            "choice_text": ["Alpha", "Beta"],
        },
    }
    rendered = prompt_text(row)
    assert "Which" in rendered and "Beta" in rendered
    assert "answer" not in rendered.casefold()
    assert normalize_text("  A\u00a0B\nC  ") == "a b c"
    assert character_shingles("a\u00e9b", 2) == {
        "a\u00e9".encode("utf-8"),
        "\u00e9b".encode("utf-8"),
    }
    first = minhash_signature(rendered, width=3, components=16, seed=4)
    second = minhash_signature(rendered, width=3, components=16, seed=4)
    assert (first == second).all()


def test_registered_geometry_constructions_are_deterministic() -> None:
    assert window_boundaries(5, 26, 3) == [5, 12, 19, 26]
    assert torch.equal(random_orthoproject(8, 4, seed=3), random_orthoproject(8, 4, seed=3))
    coordinates = bivector_coordinates(
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
    )
    assert torch.equal(coordinates, torch.tensor([[1.0, 0.0, 0.0]]))
    folds = deterministic_folds(["x"] * 8 + ["y"] * 8, folds=4, seed=11)
    assert torch.bincount(folds, minlength=4).tolist() == [4, 4, 4, 4]


def test_debiased_linear_cka_identifies_shared_not_permuted_geometry() -> None:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(64, 12, generator=generator)
    mixed = values @ torch.randn(12, 9, generator=generator)
    shared = debiased_linear_cka(values, mixed)
    permuted = debiased_linear_cka(values, mixed[torch.randperm(64, generator=generator)])
    assert shared > 0.6
    assert shared > permuted + 0.5


def test_rmt_whitener_is_finite_and_bulk_shrinking() -> None:
    assert 0.0 < mp_median(0.25) < 2.25
    samples = torch.randn(256, 16, generator=torch.Generator().manual_seed(23))
    result = rmt_whitener(samples)
    assert result["bulk_eigenvalues"] > 0
    assert torch.isfinite(result["transform"]).all()
    assert result["transform"].shape == (16, 16)


def test_tm2_common_mode_removal_and_subspace_overlap() -> None:
    generator = torch.Generator().manual_seed(41)
    shared = torch.randn(16, generator=generator)
    values = torch.randn(128, 16, generator=generator) + 4.0 * shared
    residual, receipt = remove_common_mode(values)
    assert receipt["unit_projection_energy_fraction"] > 0.5
    direction = torch.nn.functional.normalize(
        torch.nn.functional.normalize(values, dim=-1).mean(dim=0), dim=0
    )
    assert torch.max(torch.abs(residual @ direction)) < 1e-4
    basis = torch.linalg.qr(torch.randn(16, 4, generator=generator)).Q.T
    overlap = subspace_overlap(basis, basis)
    assert overlap["mean_cosine"] > 0.999


def test_tm2_bootstrap_difference_tracks_separated_groups() -> None:
    positive = np.linspace(1.0, 2.0, 64)
    negative = np.linspace(-1.0, 0.0, 64)
    receipt = bootstrap_mean_difference(positive, negative, draws=200, seed=7)
    assert receipt["estimate"] == pytest.approx(2.0)
    assert receipt["ci95_low"] > 1.5


def test_tm2_two_half_discriminator_requires_both_directions() -> None:
    generator = torch.Generator().manual_seed(53)
    positive = torch.randn(80, 8, generator=generator) + 2.0
    negative = torch.randn(80, 8, generator=generator) - 2.0
    halves = torch.arange(80).remainder(2)
    receipt = two_half_discriminative_read(
        positive, halves, negative, halves, seed=11, draws=100
    )
    assert receipt["both_halves_above_chance"]


def test_tm2_direction_cell_requires_structure_not_only_low_rank() -> None:
    names = ("D_7>0.5", "D_14>0.5", "D_14>7")
    cell = {
        name: {
            "under_minimum_rows": False,
            "variance_explained": {"32": 0.4},
        }
        for name in (*names, "D_none")
    }
    cell["discriminative"] = {
        name: {"both_halves_above_chance": True} for name in names
    }
    cell["principal_angles"] = {}
    for left_index, left in enumerate(names):
        cell["principal_angles"][f"{left}__vs__D_none"] = {
            "mean_squared_cosine": 0.1
        }
        for right in names[left_index + 1 :]:
            cell["principal_angles"][f"{left}__vs__{right}"] = {
                "mean_squared_cosine": 0.3
            }
    assert classify_direction_cell(cell, 0.3) == "STRUCTURED"
    cell["discriminative"]["D_14>7"]["both_halves_above_chance"] = False
    assert classify_direction_cell(cell, 0.3) == "GENERIC"
