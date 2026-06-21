"""Shared Stage 5 model-size metadata helpers."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from typing import Any


PARAMS_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*[bB](?=$|[^A-Za-z0-9])")


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def infer_params_b(model_name: str) -> float | None:
    matches = PARAMS_RE.findall(model_name)
    if not matches:
        return None
    return finite_float(matches[-1])


def configured_params_b(
    *,
    model_name: str,
    environ: Mapping[str, str] | None = None,
) -> float | None:
    env = environ or os.environ
    for key in ("STAGE5_MODEL_PARAMS_B", "MODEL_PARAMS_B"):
        value = finite_float(env.get(key))
        if value is not None:
            return value
    return infer_params_b(model_name)


def model_metadata(model_name: str, *, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "params_b": configured_params_b(model_name=model_name, environ=environ),
    }
