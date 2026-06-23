"""Colab credential helpers for subprocess-heavy runners."""

from __future__ import annotations

import os


def _colab_secret(name: str) -> str | None:
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return None
    try:
        value = userdata.get(name)
    except Exception:
        return None
    return str(value).strip() if value else None


def ensure_hf_token_from_colab(*, verbose: bool = True) -> bool:
    """Populate Hugging Face token env vars from Colab Secrets when possible.

    Notebook cells often set ``HF_TOKEN`` directly, but our Stage 5 launchers
    run model-loading scripts through nested subprocesses. This helper makes
    the runner entrypoints self-contained when invoked after a fresh clone.
    """

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    source = "environment"
    if not token:
        token = _colab_secret("HF_TOKEN") or _colab_secret("HUGGINGFACE_HUB_TOKEN")
        source = "Colab Secrets"
    if not token:
        if verbose:
            print("HF auth: no HF_TOKEN found; Hub access will be anonymous.", flush=True)
        return False

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    if verbose:
        print(f"HF auth: token configured from {source}.", flush=True)
    return True


def ensure_gh_token_from_colab(*, verbose: bool = True) -> bool:
    """Populate GitHub token env vars from Colab Secrets when possible."""

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        token = _colab_secret("GH_TOKEN") or _colab_secret("GITHUB_TOKEN")
    if not token:
        if verbose:
            print("GitHub auth: no GH_TOKEN/GITHUB_TOKEN found.", flush=True)
        return False

    os.environ["GH_TOKEN"] = token
    os.environ["GITHUB_TOKEN"] = token
    if verbose:
        print("GitHub auth: token configured.", flush=True)
    return True
