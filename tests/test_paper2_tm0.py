import json
from pathlib import Path

import torch

from analysis.build_paper2_tm0_manifest import _cka_calibration_rows
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
