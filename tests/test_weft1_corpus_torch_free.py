from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_production_pa_import_surface_is_torch_free() -> None:
    script = """
import builtins
import sys

real_import = builtins.__import__

def reject_torch(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("torch intentionally unavailable in corpus P-A")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_torch

import training.weft1_corpus_pa
import training.weft1_corpus_materialize_a2
import scripts.run_weft1_corpus_pa
import scripts.run_weft1_corpus_materialize_a2
from training.weft1_seed import derive_module_seed

assert derive_module_seed(20260826, "model.core.0.attention.dropout", 1) == 6870707043680624902
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
