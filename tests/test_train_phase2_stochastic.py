from __future__ import annotations

import inspect


def test_phase2_training_passes_svgd_projection_knobs() -> None:
    import training.train_phase2_stochastic as module

    source = inspect.getsource(module.main)

    assert 'svgd_kernel_projection_dim=cfg_optional_int(cfg, "svgd_kernel_projection_dim")' in source
    assert 'svgd_kernel_projection_path=cfg.get("svgd_kernel_projection_path")' in source
    assert 'svgd_kernel_geometry=cfg.get("svgd_kernel_geometry", "euclidean")' in source
    assert 'svgd_projection_seed=cfg_int(cfg, "svgd_projection_seed", 0)' in source


def test_phase2_training_supports_trajectory_distillation() -> None:
    import training.train_phase2_stochastic as module

    source = inspect.getsource(module.main)

    assert 'distill_cfg = cfg.get("distillation", {})' in source
    assert 'distill_cfg.get("target", "mean") == "trajectories"' in source
    assert "teacher_out.logits.repeat_interleave(num_trajectories, dim=0)" in source
    assert 'output.metrics["base_distill_kl"]' in source
