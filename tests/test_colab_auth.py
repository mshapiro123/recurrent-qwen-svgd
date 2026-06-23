from __future__ import annotations

import sys
import types
import os

from colab.colab_auth import ensure_gh_token_from_colab, ensure_hf_token_from_colab


class FakeUserdata:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def get(self, name: str) -> str | None:
        return self.values.get(name)


def install_fake_colab(monkeypatch, values: dict[str, str]) -> None:
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.userdata = FakeUserdata(values)  # type: ignore[attr-defined]
    google.colab = colab  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


def clear_auth_env(monkeypatch) -> None:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(key, raising=False)


def test_hf_token_uses_existing_environment(monkeypatch) -> None:
    clear_auth_env(monkeypatch)
    install_fake_colab(monkeypatch, {"HF_TOKEN": "secret-token"})
    monkeypatch.setenv("HF_TOKEN", "env-token")

    assert ensure_hf_token_from_colab(verbose=False)
    assert sys.modules["google.colab"].userdata.values["HF_TOKEN"] == "secret-token"  # type: ignore[attr-defined]
    assert os.environ["HF_TOKEN"] == "env-token"
    assert os.environ["HUGGINGFACE_HUB_TOKEN"] == "env-token"


def test_hf_token_can_be_loaded_from_colab_userdata(monkeypatch) -> None:
    clear_auth_env(monkeypatch)
    install_fake_colab(monkeypatch, {"HF_TOKEN": "secret-token"})

    assert ensure_hf_token_from_colab(verbose=False)
    assert os.environ["HF_TOKEN"] == "secret-token"
    assert os.environ["HUGGINGFACE_HUB_TOKEN"] == "secret-token"


def test_hf_token_missing_is_nonfatal(monkeypatch) -> None:
    clear_auth_env(monkeypatch)
    monkeypatch.delitem(sys.modules, "google", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)

    assert not ensure_hf_token_from_colab(verbose=False)


def test_gh_token_can_be_loaded_from_colab_userdata(monkeypatch) -> None:
    clear_auth_env(monkeypatch)
    install_fake_colab(monkeypatch, {"GH_TOKEN": "gh-secret"})

    assert ensure_gh_token_from_colab(verbose=False)
    assert os.environ["GH_TOKEN"] == "gh-secret"
    assert os.environ["GITHUB_TOKEN"] == "gh-secret"
