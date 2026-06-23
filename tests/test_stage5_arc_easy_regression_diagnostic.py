from colab.run_stage5_arc_easy_regression_diagnostic import choose_status


def test_choose_status_prefers_conditional_invariance_for_order_sensitivity() -> None:
    status, action, next_step = choose_status(
        {"recommendation": "prioritize_conditional_invariance_repair"},
        {"recommendation": "prioritize_direct_distillation_or_data_repair"},
    )

    assert status == "order_sensitivity_likely"
    assert action == "conditional_invariance_repair"
    assert "conditional-invariance" in next_step


def test_choose_status_uses_surface_alignment_for_stable_cyclic_rescue() -> None:
    status, action, next_step = choose_status(
        {"recommendation": "diagnose_content_route_scoring_or_prompt_alignment_before_more_distillation"},
        {"recommendation": "prioritize_content_cyclic_surface_alignment"},
    )

    assert status == "surface_mismatch_likely"
    assert action == "content_cyclic_surface_alignment"
    assert "surface-alignment" in next_step


def test_choose_status_allows_direct_repair_only_after_surface_explanations_fail() -> None:
    status, action, next_step = choose_status(
        {"recommendation": "prioritize_direct_distillation_or_data_repair"},
        {"recommendation": "prioritize_direct_distillation_or_data_repair"},
    )

    assert status == "content_erosion_likely"
    assert action == "direct_preservation_repair"
    assert "direct-preservation" in next_step
