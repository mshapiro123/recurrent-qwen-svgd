"""Clean-tree, behavior-bearing source closure for G-TOK base + confirmation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess

from training.weft1_gtok_contract import canonical_sha256


_EXPLICIT_FILES = (
    ".gitattributes",
    "docs/PAPER2_PHASE3_P31_P32_AUTHORIZED_BUILD_HANDOFF_20260810.md",
    "docs/STRATEGY_CORPUS_GTOK_AMENDMENT_A3_20260829.md",
    "docs/STRATEGY_RELEASE_POSTURE_AND_LICENSE_CLOSE_20260830.md",
    "scripts/bind_weft1_gtok_runtime_v2.py",
    "scripts/build_weft1_gtok_runtime_v2.py",
    "scripts/materialize_weft1_gtok_wheelhouse_v2.py",
    "scripts/launch_weft1_gtok_offline_v2.py",
    "scripts/precompute_weft1_gtok_cpu_v2.py",
    "scripts/request_weft1_gtok_runtime_binding_v2.py",
    "scripts/run_weft1_gtok_campaign_v2.py",
    "scripts/run_weft1_gtok_full_campaign_v2.py",
    "scripts/run_weft1_gtok_v2.py",
    "training/weft1_corpus_a2.py",
    "training/paper2_phase3_p31.py",
    "training/weft1_corpus_gtok_a2_bindings_20260828.json",
    "training/weft1_corpus_materialize_a3.py",
    "training/weft1_corpus_replay_a2.py",
    "training/weft1_gtok_campaign_v2.py",
    "training/weft1_gtok_code_closure_v2.py",
    "training/weft1_gtok_confirmation_v2.py",
    "training/weft1_gtok_contract.py",
    "training/weft1_gtok_determinism_v2.py",
    "training/weft1_gtok_pb_adapter_v2.py",
    "training/weft1_gtok_offline_v2.py",
    "training/weft1_gtok_runtime_v2.py",
    "training/weft1_gtok_tokenizer_a2.py",
    "training/weft1_gtok_tokenizer_v2.py",
    "training/weft1_gtok_training_requirements_v2.lock",
    "training/weft1_gtok_training_requirements_v2.txt",
    "training/weft1_gtok_training_v2.py",
    "training/weft1_gtok_v2_contract.py",
    "training/weft1_seed.py",
    "training/weft1_strict_io.py",
)


class GTokCodeClosureV2Error(RuntimeError):
    """The behavior-bearing checkout is dirty or changed."""


@dataclass(frozen=True)
class CodeArtifactV2:
    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class GTokCodeClosureReceiptV2:
    git_commit: str
    artifacts: tuple[CodeArtifactV2, ...]
    status: str = "CLEAN_EXACT_CODE_CLOSURE"
    schema: str = "weft1_gtok_code_closure_v2"

    def __post_init__(self) -> None:
        if len(self.git_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.git_commit
        ):
            raise ValueError("code closure requires one exact git commit")
        paths = tuple(row.relative_path for row in self.artifacts)
        if paths != tuple(sorted(set(paths))) or not paths:
            raise ValueError("code closure artifacts require unique canonical order")
        if any(
            row.bytes < 1
            or len(row.sha256) != 64
            or any(character not in "0123456789abcdef" for character in row.sha256)
            for row in self.artifacts
        ):
            raise ValueError("code closure artifact identity is invalid")

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self)


def _behavior_paths(root: Path) -> tuple[str, ...]:
    training_files = tuple(
        path.relative_to(root).as_posix()
        for path in sorted((root / "training").glob("weft1_*.py"))
    )
    script_files = tuple(
        path.relative_to(root).as_posix()
        for path in sorted((root / "scripts").glob("*weft1*.py"))
    )
    binding_files = tuple(
        path.relative_to(root).as_posix()
        for pattern in ("weft1_*.json", "weft1_*.lock", "weft1_*.txt")
        for path in sorted((root / "training").glob(pattern))
    )
    model_files = tuple(
        path.relative_to(root).as_posix()
        for path in sorted((root / "models" / "ablation_lm").rglob("*.py"))
    )
    sidecar = root / "models" / "sidecar_v2.py"
    sidecar_files = (sidecar.relative_to(root).as_posix(),) if sidecar.is_file() else ()
    return tuple(
        sorted(
            set(_EXPLICIT_FILES)
            | set(training_files)
            | set(script_files)
            | set(binding_files)
            | set(model_files)
            | set(sidecar_files)
        )
    )


def capture_gtok_code_closure_v2(repository_root: Path) -> GTokCodeClosureReceiptV2:
    root = repository_root.resolve(strict=True)
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise GTokCodeClosureV2Error("G-TOK production requires a completely clean checkout")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    artifacts: list[CodeArtifactV2] = []
    for relative in _behavior_paths(root):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise GTokCodeClosureV2Error(f"behavior-bearing source is absent/symlinked: {relative}")
        raw = path.read_bytes()
        artifacts.append(
            CodeArtifactV2(
                relative_path=relative,
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    return GTokCodeClosureReceiptV2(git_commit=commit, artifacts=tuple(artifacts))


def validate_gtok_code_closure_v2(
    receipt: GTokCodeClosureReceiptV2,
    *,
    repository_root: Path,
) -> None:
    if capture_gtok_code_closure_v2(repository_root) != receipt:
        raise GTokCodeClosureV2Error("behavior-bearing source closure changed during campaign")


__all__ = [
    "CodeArtifactV2",
    "GTokCodeClosureReceiptV2",
    "GTokCodeClosureV2Error",
    "capture_gtok_code_closure_v2",
    "validate_gtok_code_closure_v2",
]
