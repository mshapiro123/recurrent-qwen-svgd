from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_copy_ready_figure5_options_exist() -> None:
    for stem in ("figure5_option_a_three_panel", "figure5_option_b_two_panel"):
        for suffix in ("svg", "pdf", "png"):
            assert (ROOT / f"docs/figures/{stem}.{suffix}").exists()


def test_three_panel_option_names_all_relationships() -> None:
    svg = (ROOT / "docs/figures/figure5_option_a_three_panel.svg").read_text(encoding="utf-8")

    assert "Accuracy by depth" in svg
    assert "Latency by depth" in svg
    assert "Accuracy by latency" in svg
    assert "scratchpad" in svg
