from __future__ import annotations

from eval.eval_paper2_phase3_p32_coverage import coverage_surface


def _row(
    index: int,
    *,
    flip: bool,
    covered: bool,
    concurrent: bool,
    teachability: float,
    margin: float,
) -> dict[str, object]:
    return {
        "record_id": str(index),
        "source": "old" if index < 3 else "new",
        "anchor_index": index,
        "horizon": 1,
        "stratum": "general" if index % 2 == 0 else "code",
        "flip_candidate_14b": flip,
        "cascade_covered": covered,
        "cross_scale_consistent": concurrent,
        "teachability": teachability,
        "confident_agreement_margin": margin,
    }


def test_coverage_surface_separates_strict_writes_and_extension() -> None:
    records = [
        _row(0, flip=True, covered=True, concurrent=True, teachability=0.9, margin=0.0),
        _row(1, flip=True, covered=False, concurrent=False, teachability=0.9, margin=0.0),
        _row(2, flip=True, covered=True, concurrent=False, teachability=0.9, margin=0.0),
        _row(3, flip=False, covered=False, concurrent=False, teachability=0.1, margin=2.5),
    ]
    result = coverage_surface(
        records,
        teachability_thresholds=[0.8],
        margin_thresholds=[2.0],
    )
    write = result["strict_write_surface"][0]
    assert write["14b_flip_candidates"] == 3
    assert write["strict_concurrent_write_candidates"] == 1
    assert write["cross_scale_conflicts"] == 1
    assert write["targeted_32b_extension_candidates"] == 1
    negative = result["permissive_negative_surface"][0]
    assert negative["confident_agreement_negatives"] == 1
    assert negative["14b_only_admissible"] == 1
    assert result["thresholds_selected_for_p33"] is False
