from __future__ import annotations

import inspect


def test_phase2_training_passes_svgd_projection_knobs() -> None:
    import training.train_phase2_stochastic as module

    source = inspect.getsource(module.main)

    assert 'svgd_kernel_projection_dim=cfg.get("svgd_kernel_projection_dim")' in source
    assert 'svgd_kernel_projection_path=cfg.get("svgd_kernel_projection_path")' in source
    assert 'svgd_kernel_geometry=cfg.get("svgd_kernel_geometry", "euclidean")' in source
    assert 'svgd_projection_seed=cfg.get("svgd_projection_seed", 0)' in source
