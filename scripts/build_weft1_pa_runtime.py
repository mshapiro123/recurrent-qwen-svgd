#!/usr/bin/env python3
"""Build the exact WEFT-1 P-A Python runtime from hash-pinned sources.

This builder is intended for a fresh Linux/Colab VM.  It builds CPython 3.11.9
against SQLite 3.45.1, installs the existing hash-locked Python environment via
an offline wheelhouse, runs the repository's RuntimeExpectationV3 attestation,
and emits a canonical provenance receipt.  It never overwrites a build root,
installation prefix, download, or receipt.

``--dry-run`` is intentionally host-independent and performs no filesystem,
network, package, or subprocess mutation.  It exists so CI can audit the exact
recipe without compiling the runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from typing import Mapping, Sequence


SCHEMA = "weft1_pa_runtime_builder_v1"
RECIPE_SCHEMA = "weft1_pa_runtime_recipe_v1"
RECEIPT_SCHEMA = "weft1_pa_runtime_build_receipt_v1"
INSTALLED_DISTRIBUTION_INVENTORY_SCHEMA = (
    "weft1_installed_distribution_inventory_v3"
)
TRUSTED_INSTALLER_CHAIN_SCHEMA = "weft1_trusted_installer_chain_v1"
TRUSTED_INSTALLER_THREAT_MODEL = (
    "pinned CPython ensurepip pip is trusted to install lock-authorized wheels; "
    "coherent malicious file-and-RECORD rewriting before the initial receipt is "
    "outside scope"
)
PYTHON_VERSION = "3.11.9"
UNICODE_VERSION = "14.0.0"
SQLITE_VERSION = "3.45.1"
SQLITE_SOURCE_ID = (
    "2024-01-30 16:01:20 "
    "e876e51a0ed5c5b3126f52e532044363a014bc594cfefa87ffb5b82257cc467a"
)
SQLITE3_C_SHA3_256 = (
    "0474604df9e1b69a5544295dd046aad954749279780d557da80f44b958100295"
)
ZSTANDARD_PACKAGE_VERSION = "0.25.0"
LIBZSTD_VERSION = "1.5.7"
LOCK_SHA256 = "bccb8e5b58b5e8fa9eee367fe9c26f59053fff5b7fadf81f23f96b83d1531860"
SOURCE_DATE_EPOCH = "1712016000"
CPYTHON_SITE_PACKAGES_README_RELATIVE_PATH = (
    "lib/python3.11/site-packages/README.txt"
)
CPYTHON_SITE_PACKAGES_README_BYTES = 119
CPYTHON_SITE_PACKAGES_README_SHA256 = (
    "cba8fece8f62c36306ba27a128f124a257710e41fc619301ee97be93586917cb"
)
ALLOWED_MACHINES = frozenset({"amd64", "x86_64"})
LOCK_RELATIVE_PATH = Path("training/weft1_corpus_gtok_a2_requirements.lock")
RUNTIME_CONTRACT_RELATIVE_PATH = Path("training/weft1_corpus_pa.py")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

BUILD_DEPENDENCY_PACKAGES = (
    "build-essential",
    "ca-certificates",
    "libbz2-dev",
    "libdb-dev",
    "libffi-dev",
    "libgdbm-dev",
    "liblzma-dev",
    "libncursesw5-dev",
    "libnsl-dev",
    "libreadline-dev",
    "libssl-dev",
    "pkg-config",
    "tk-dev",
    "uuid-dev",
    "xz-utils",
    "zlib1g-dev",
)
REQUIRED_TOOLS = (
    "ar",
    "cc",
    "dpkg-query",
    "ldd",
    "make",
    "pkg-config",
    "readelf",
)


class RuntimeBuildError(RuntimeError):
    """The governed runtime build could not establish an exact result."""


@dataclass(frozen=True)
class SourcePin:
    name: str
    url: str
    filename: str
    byte_count: int
    sha256: str
    sha3_256: str | None = None

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlparse(self.url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.fragment:
            raise ValueError("source URL must be an exact HTTPS URL")
        if Path(self.filename).name != self.filename or not self.filename:
            raise ValueError("source filename must be a basename")
        if type(self.byte_count) is not int or self.byte_count < 1:
            raise ValueError("source byte count must be positive")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("source SHA-256 must be lowercase hexadecimal")
        if self.sha3_256 is not None and _SHA256.fullmatch(self.sha3_256) is None:
            raise ValueError("source SHA3-256 must be lowercase hexadecimal")


PYTHON_SOURCE = SourcePin(
    name="cpython",
    url="https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tar.xz",
    filename="Python-3.11.9.tar.xz",
    byte_count=20_175_816,
    sha256="9b1e896523fc510691126c864406d9360a3d1e986acbda59cda57b5abda45b87",
)
SQLITE_SOURCE = SourcePin(
    name="sqlite-amalgamation",
    url="https://www.sqlite.org/2024/sqlite-amalgamation-3450100.zip",
    filename="sqlite-amalgamation-3450100.zip",
    byte_count=2_730_697,
    sha256="5592243caf28b2cdef41e6ab58d25d653dfc53deded8450eb66072c929f030c4",
    sha3_256="e311198775d5d5b2889d5fabe1d9a490567a14e605591d6a9e4c833804a8b4cb",
)
SOURCE_PINS = (PYTHON_SOURCE, SQLITE_SOURCE)


@dataclass(frozen=True)
class BuildPaths:
    repository_root: Path
    work_root: Path
    prefix: Path
    receipt_path: Path

    @property
    def lock_path(self) -> Path:
        return self.repository_root / LOCK_RELATIVE_PATH

    @property
    def downloads(self) -> Path:
        return self.work_root / "downloads"

    @property
    def sources(self) -> Path:
        return self.work_root / "sources"

    @property
    def sqlite_prefix(self) -> Path:
        # Keep the governed SQLite ABI inside the runtime prefix so the exact
        # runtime never depends on build-root survival.
        return self.prefix

    @property
    def wheelhouse(self) -> Path:
        return self.work_root / "wheelhouse"

    @property
    def build_log(self) -> Path:
        return self.work_root / "build.log"

    @property
    def pip_report(self) -> Path:
        return self.work_root / "pip-install-report.json"

    @property
    def lock_snapshot(self) -> Path:
        return self.work_root / "requirements.lock"


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    return _hash_file(path, "sha256")


def sha3_256_file(path: Path) -> str:
    return _hash_file(path, "sha3_256")


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def assert_no_symlink_ancestors(path: Path) -> Path:
    lexical = _lexical_absolute(path)
    for component in reversed((lexical, *lexical.parents)):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RuntimeBuildError(
                f"cannot inspect governed path component: {component}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeBuildError(
                f"governed path traverses a symlink: {component}"
            )
    return lexical


def _require_fresh_nonoverlapping_paths(paths: BuildPaths) -> BuildPaths:
    normalized = BuildPaths(
        repository_root=assert_no_symlink_ancestors(paths.repository_root).resolve(
            strict=True
        ),
        work_root=assert_no_symlink_ancestors(paths.work_root).resolve(strict=False),
        prefix=assert_no_symlink_ancestors(paths.prefix).resolve(strict=False),
        receipt_path=assert_no_symlink_ancestors(paths.receipt_path).resolve(
            strict=False
        ),
    )
    mutable = (normalized.work_root, normalized.prefix, normalized.receipt_path)
    if any(path.exists() or path.is_symlink() for path in mutable):
        raise RuntimeBuildError("work root, prefix, and receipt must all be fresh")
    if (
        normalized.work_root == normalized.prefix
        or normalized.work_root in normalized.prefix.parents
        or normalized.prefix in normalized.work_root.parents
        or normalized.receipt_path in normalized.work_root.parents
        or normalized.receipt_path in normalized.prefix.parents
        or normalized.receipt_path == normalized.work_root
        or normalized.receipt_path == normalized.prefix
        or normalized.work_root in normalized.receipt_path.parents
        or normalized.prefix in normalized.receipt_path.parents
    ):
        raise RuntimeBuildError(
            "work root, prefix, and receipt path must be mutually non-overlapping"
        )
    if not normalized.lock_path.is_file():
        raise RuntimeBuildError("governed requirements lock is absent")
    return normalized


def verify_lock(path: Path) -> str:
    observed = sha256_file(path)
    if observed != LOCK_SHA256:
        raise RuntimeBuildError("requirements lock differs from the runtime contract")
    return observed


def snapshot_verified_lock(source: Path, destination: Path) -> Mapping[str, object]:
    """Copy the governed lock through one stable handle and verify its bytes."""

    if destination.exists() or destination.is_symlink():
        raise RuntimeBuildError("requirements-lock snapshot destination must be fresh")
    try:
        source_metadata = source.lstat()
    except OSError as error:
        raise RuntimeBuildError("requirements lock cannot be inspected") from error
    if stat.S_ISLNK(source_metadata.st_mode) or not stat.S_ISREG(source_metadata.st_mode):
        raise RuntimeBuildError("requirements lock must be a regular non-symlink file")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with source.open("rb") as opened, destination.open("xb") as copied:
            opened_metadata = os.fstat(opened.fileno())
            if not stat.S_ISREG(opened_metadata.st_mode):
                raise RuntimeBuildError("requirements lock handle is not a regular file")
            for chunk in iter(lambda: opened.read(1024 * 1024), b""):
                byte_count += len(chunk)
                digest.update(chunk)
                copied.write(chunk)
            copied.flush()
            os.fsync(copied.fileno())
    except BaseException:
        if destination.exists():
            destination.unlink()
        raise
    observed = digest.hexdigest()
    if observed != LOCK_SHA256:
        destination.unlink()
        raise RuntimeBuildError("requirements lock differs from the runtime contract")
    if sha256_file(destination) != observed or destination.stat().st_size != byte_count:
        destination.unlink()
        raise RuntimeBuildError("requirements-lock snapshot verification failed")
    return {
        "bytes": byte_count,
        "filename": destination.name,
        "sha256": observed,
        "source_filename": source.name,
    }


def parse_locked_distributions(path: Path) -> tuple[tuple[str, str], ...]:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as error:
        raise RuntimeBuildError("requirements lock is not strict UTF-8") from error
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line or line[0].isspace() or line.startswith("#") or "==" not in line:
            continue
        name, remainder = line.split("==", 1)
        version = remainder.split(None, 1)[0].rstrip("\\")
        if not name or not version:
            raise RuntimeBuildError("requirements lock contains an invalid pin")
        rows.append((name, version))
    normalized = tuple(name.casefold().replace("_", "-") for name, _ in rows)
    if not rows or len(normalized) != len(set(normalized)):
        raise RuntimeBuildError("requirements lock is empty or repeats a distribution")
    return tuple(rows)


def verify_source_file(path: Path, pin: SourcePin) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeBuildError(f"{pin.name} download is not a regular file")
    observed_bytes = path.stat().st_size
    observed_sha256 = sha256_file(path)
    observed_sha3 = sha3_256_file(path) if pin.sha3_256 is not None else None
    if (
        observed_bytes != pin.byte_count
        or observed_sha256 != pin.sha256
        or observed_sha3 != pin.sha3_256
    ):
        raise RuntimeBuildError(f"{pin.name} source bytes differ from their pin")
    return {
        "bytes": observed_bytes,
        "filename": pin.filename,
        "name": pin.name,
        "sha256": observed_sha256,
        "sha3_256": observed_sha3,
        "url": pin.url,
    }


def download_source(pin: SourcePin, destination: Path) -> Mapping[str, object]:
    if destination.exists() or destination.is_symlink():
        raise RuntimeBuildError("source download destination must be fresh")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise RuntimeBuildError("source partial destination must be fresh")
    request = urllib.request.Request(
        pin.url,
        headers={"User-Agent": "WEFT-1-P-A-runtime-builder/1"},
        method="GET",
    )
    byte_count = 0
    sha256 = hashlib.sha256()
    sha3 = hashlib.sha3_256()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.geturl() != pin.url:
                raise RuntimeBuildError(f"{pin.name} source URL redirected")
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) != pin.byte_count:
                raise RuntimeBuildError(
                    f"{pin.name} HTTP byte count differs from its pin"
                )
            with partial.open("xb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    byte_count += len(chunk)
                    if byte_count > pin.byte_count:
                        raise RuntimeBuildError(
                            f"{pin.name} download exceeded its pinned size"
                        )
                    sha256.update(chunk)
                    sha3.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        observed_sha3 = sha3.hexdigest() if pin.sha3_256 is not None else None
        if (
            byte_count != pin.byte_count
            or sha256.hexdigest() != pin.sha256
            or observed_sha3 != pin.sha3_256
        ):
            raise RuntimeBuildError(f"{pin.name} download failed byte verification")
        os.replace(partial, destination)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return verify_source_file(destination, pin)


def _safe_archive_relative_path(raw: str, *, expected_root: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or path.is_absolute()
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
        or not path.parts
        or path.parts[0] != expected_root
    ):
        raise RuntimeBuildError("source archive contains an unsafe path")
    return path


def safe_extract_tar_xz(archive: Path, destination: Path) -> Path:
    expected_root = f"Python-{PYTHON_VERSION}"
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, mode="r:xz") as bundle:
        members = bundle.getmembers()
        if not members:
            raise RuntimeBuildError("CPython source archive is empty")
        for member in members:
            relative = _safe_archive_relative_path(
                member.name.rstrip("/"), expected_root=expected_root
            )
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeBuildError(
                    "CPython source archive contains a link or special file"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise RuntimeBuildError("CPython archive member could not be read")
            with extracted, target.open("xb") as output:
                shutil.copyfileobj(extracted, output, length=1024 * 1024)
            target.chmod(member.mode & 0o777)
    root = destination / expected_root
    if not (root / "configure").is_file():
        raise RuntimeBuildError("CPython source archive lacks configure")
    return root


def safe_extract_sqlite_zip(archive: Path, destination: Path) -> Path:
    expected_root = "sqlite-amalgamation-3450100"
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive, mode="r") as bundle:
        members = bundle.infolist()
        if not members:
            raise RuntimeBuildError("SQLite source archive is empty")
        for member in members:
            relative = _safe_archive_relative_path(
                member.filename.rstrip("/"), expected_root=expected_root
            )
            target = destination.joinpath(*relative.parts)
            unix_mode = member.external_attr >> 16
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise RuntimeBuildError("SQLite source archive contains a symlink")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member, mode="r") as opened, target.open("xb") as output:
                shutil.copyfileobj(opened, output, length=1024 * 1024)
    root = destination / expected_root
    required = ("shell.c", "sqlite3.c", "sqlite3.h", "sqlite3ext.h")
    if any(not (root / name).is_file() for name in required):
        raise RuntimeBuildError("SQLite amalgamation archive is incomplete")
    if sha3_256_file(root / "sqlite3.c") != SQLITE3_C_SHA3_256:
        raise RuntimeBuildError("sqlite3.c differs from the official release SHA3-256")
    return root


def _abstract_recipe() -> Mapping[str, object]:
    sqlite_cflags = (
        "-O2",
        "-g0",
        "-fPIC",
        "-DSQLITE_THREADSAFE=1",
        "-DSQLITE_ENABLE_COLUMN_METADATA=1",
        "-DSQLITE_ENABLE_FTS5=1",
        "-DSQLITE_ENABLE_MATH_FUNCTIONS=1",
        "-DSQLITE_ENABLE_RTREE=1",
    )
    python_configure = (
        "./configure",
        "--prefix=${PREFIX}",
        "--enable-shared",
        "--with-ensurepip=install",
        "--with-system-ffi",
    )
    return {
        "build_dependency_packages": BUILD_DEPENDENCY_PACKAGES,
        "cpython_site_packages_readme_removal": {
            "bytes": CPYTHON_SITE_PACKAGES_README_BYTES,
            "relative_path": CPYTHON_SITE_PACKAGES_README_RELATIVE_PATH,
            "sha256": CPYTHON_SITE_PACKAGES_README_SHA256,
        },
        "deterministic_environment": {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.defpath,
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PIP_ROOT_USER_ACTION": "ignore",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH,
            "TOKENIZERS_PARALLELISM": "false",
            "TZ": "UTC",
        },
        "expected_runtime": {
            "libzstd_version": LIBZSTD_VERSION,
            "python_version": PYTHON_VERSION,
            "sqlite_source_id": SQLITE_SOURCE_ID,
            "sqlite_version": SQLITE_VERSION,
            "unicode_data_version": UNICODE_VERSION,
            "zstandard_package_version": ZSTANDARD_PACKAGE_VERSION,
        },
        "lock_sha256": LOCK_SHA256,
        "pip_download_flags": (
            "--require-hashes",
            "--only-binary=:all:",
            "--no-deps",
            "--no-cache-dir",
        ),
        "pip_install_flags": (
            "--require-hashes",
            "--only-binary=:all:",
            "--no-deps",
            "--no-cache-dir",
            "--no-index",
            "--report=${FRESH_REPORT}",
        ),
        "trusted_installer": {
            "bootstrap_distribution": "pip",
            "bootstrap_version": "24.0",
            "schema": TRUSTED_INSTALLER_CHAIN_SCHEMA,
            "threat_model": TRUSTED_INSTALLER_THREAT_MODEL,
        },
        "python_configure": python_configure,
        "python_make": ("make", "-j${JOBS}", "make install"),
        "runtime_linkage_checks": (
            "target interpreter must start with LD_LIBRARY_PATH absent and -I -B",
            "import sqlite3 and _sqlite3 must succeed",
            f"sqlite_source_id() == {SQLITE_SOURCE_ID}",
            "parsed ldd(_sqlite3) must resolve libsqlite3.so.0 to the governed versioned library",
            "parsed ldd(python) must resolve libpython3.11.so.1.0 inside the governed prefix",
            "readelf must show the governed prefix library directory in RUNPATH/RPATH",
        ),
        "schema": RECIPE_SCHEMA,
        "source_pins": tuple(asdict(pin) for pin in SOURCE_PINS),
        "sqlite3_c_sha3_256": SQLITE3_C_SHA3_256,
        "sqlite_compile_flags": sqlite_cflags,
        "sqlite_link_libraries": ("dl", "pthread", "m"),
        "sqlite_install_location": "${PREFIX}",
        "sqlite_shared_object": "libsqlite3.so.0.8.6",
    }


def recipe_identity_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(_abstract_recipe()))


def dry_run_receipt() -> Mapping[str, object]:
    recipe = _abstract_recipe()
    return {
        "authoritative": False,
        "recipe": recipe,
        "recipe_identity_sha256": sha256_bytes(canonical_json_bytes(recipe)),
        "schema": SCHEMA,
        "status": "PLAN_ONLY_NO_EXECUTION",
    }


def _run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
) -> None:
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise RuntimeBuildError("build command is not an exact string sequence")
    with log_path.open("ab") as log:
        log.write(("$ " + " ".join(command) + "\n").encode("utf-8"))
        log.flush()
        result = subprocess.run(
            tuple(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            shell=False,
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_bytes()[-4000:].decode("utf-8", errors="replace")
        raise RuntimeBuildError(
            f"build command exited {result.returncode}: {' '.join(command)}\n{tail}"
        )


def _capture(command: Sequence[str], *, environment: Mapping[str, str]) -> str:
    result = subprocess.run(
        tuple(command),
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeBuildError(f"inspection command failed: {' '.join(command)}")
    return result.stdout.decode("utf-8", errors="strict").strip()


def _host_preflight() -> Mapping[str, object]:
    if platform.system() != "Linux":
        raise RuntimeBuildError("authoritative runtime builds require Linux")
    machine = platform.machine().casefold()
    if machine not in ALLOWED_MACHINES:
        raise RuntimeBuildError("runtime lock is authorized only for x86-64 Linux")
    tools: dict[str, Mapping[str, str]] = {}
    environment = os.environ.copy()
    for name in REQUIRED_TOOLS:
        resolved = shutil.which(name)
        if resolved is None:
            raise RuntimeBuildError(f"required build tool is absent: {name}")
        path = Path(resolved).resolve(strict=True)
        if not path.is_file() or path.is_symlink():
            raise RuntimeBuildError(f"build tool is not a regular file: {name}")
        version_flag = "--version" if name != "dpkg-query" else "--version"
        version = _capture((str(path), version_flag), environment=environment)
        tools[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "version_first_line": version.splitlines()[0],
        }
    return {
        "libc": platform.libc_ver(),
        "machine": platform.machine(),
        "platform_release": platform.release(),
        "platform_system": platform.system(),
        "tools": tools,
    }


def _dependency_versions(environment: Mapping[str, str]) -> Mapping[str, str]:
    command = (
        "dpkg-query",
        "-W",
        # ``binary:Package`` appends an architecture qualifier for Multi-Arch
        # packages (for example ``libssl-dev:amd64``), which would make the
        # exact dependency-name check below reject an otherwise valid Colab
        # host.  ``Package`` is the canonical unqualified package identity.
        "-f=${Package}\t${Version}\\n",
        *BUILD_DEPENDENCY_PACKAGES,
    )
    output = _capture(command, environment=environment)
    rows: dict[str, str] = {}
    for line in output.splitlines():
        name, separator, version = line.partition("\t")
        if not separator or not name or not version or name in rows:
            raise RuntimeBuildError("dpkg build-dependency evidence is malformed")
        rows[name] = version
    missing = sorted(set(BUILD_DEPENDENCY_PACKAGES) - set(rows))
    if missing:
        raise RuntimeBuildError(
            "build dependency packages are absent: " + ", ".join(missing)
        )
    return dict(sorted(rows.items()))


def _deterministic_environment(paths: BuildPaths) -> dict[str, str]:
    # Do not let Colab's ambient Python/user-site/pip controls participate in
    # either the bootstrap interpreter or the target runtime.  Build commands
    # need only the system executable search path plus the exact values below.
    environment = {
        "ARFLAGS": "rcsD",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.defpath,
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "PIP_ROOT_USER_ACTION": "ignore",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH,
        "TOKENIZERS_PARALLELISM": "false",
        "TZ": "UTC",
        "ZERO_AR_DATE": "1",
    }
    sqlite_lib = paths.sqlite_prefix / "lib"
    include = paths.sqlite_prefix / "include"
    maps = f"{paths.work_root}=/usr/src/weft1-pa"
    environment["CFLAGS"] = f"-O2 -g0 -ffile-prefix-map={maps} -fdebug-prefix-map={maps}"
    environment["CPPFLAGS"] = f"-I{include}"
    environment["LDFLAGS"] = (
        f"-L{sqlite_lib} -Wl,-rpath,{sqlite_lib} -Wl,--enable-new-dtags"
    )
    environment["LD_LIBRARY_PATH"] = str(sqlite_lib)
    environment["PKG_CONFIG_PATH"] = str(sqlite_lib / "pkgconfig")
    return environment


def _self_contained_runtime_environment(
    build_environment: Mapping[str, str],
) -> dict[str, str]:
    """Return the isolated target environment with loader overrides absent."""

    environment = dict(build_environment)
    for key in (
        "LD_LIBRARY_PATH",
        "PYTHONHOME",
        "PYTHONPATH",
    ):
        environment.pop(key, None)
    return environment


def _write_sqlite_pkg_config(paths: BuildPaths) -> None:
    pkgconfig = paths.sqlite_prefix / "lib" / "pkgconfig"
    pkgconfig.mkdir(parents=True, exist_ok=False)
    payload = (
        f"prefix={paths.sqlite_prefix}\n"
        "exec_prefix=${prefix}\n"
        "libdir=${exec_prefix}/lib\n"
        "includedir=${prefix}/include\n\n"
        "Name: SQLite\n"
        "Description: SQL database engine\n"
        f"Version: {SQLITE_VERSION}\n"
        "Libs: -L${libdir} -lsqlite3\n"
        "Libs.private: -ldl -lpthread -lm\n"
        "Cflags: -I${includedir}\n"
    ).encode("utf-8")
    with (pkgconfig / "sqlite3.pc").open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _build_sqlite(
    source_root: Path,
    paths: BuildPaths,
    *,
    environment: Mapping[str, str],
) -> Path:
    include = paths.sqlite_prefix / "include"
    library = paths.sqlite_prefix / "lib"
    binary = paths.sqlite_prefix / "bin"
    for directory in (include, library, binary):
        directory.mkdir(parents=True, exist_ok=False)
    object_path = paths.work_root / "sqlite3.o"
    flags = tuple(_abstract_recipe()["sqlite_compile_flags"])
    _run_logged(
        ("cc", *flags, "-c", str(source_root / "sqlite3.c"), "-o", str(object_path)),
        cwd=paths.work_root,
        environment=environment,
        log_path=paths.build_log,
    )
    versioned_library = library / "libsqlite3.so.0.8.6"
    _run_logged(
        (
            "cc",
            "-shared",
            "-Wl,-soname,libsqlite3.so.0",
            "-o",
            str(versioned_library),
            str(object_path),
            "-ldl",
            "-lpthread",
            "-lm",
        ),
        cwd=paths.work_root,
        environment=environment,
        log_path=paths.build_log,
    )
    _run_logged(
        ("ar", "rcsD", str(library / "libsqlite3.a"), str(object_path)),
        cwd=paths.work_root,
        environment=environment,
        log_path=paths.build_log,
    )
    os.symlink(versioned_library.name, library / "libsqlite3.so.0")
    os.symlink("libsqlite3.so.0", library / "libsqlite3.so")
    for name in ("sqlite3.h", "sqlite3ext.h"):
        shutil.copyfile(source_root / name, include / name)
    _run_logged(
        (
            "cc",
            "-O2",
            "-g0",
            str(source_root / "shell.c"),
            str(source_root / "sqlite3.c"),
            "-o",
            str(binary / "sqlite3"),
            "-ldl",
            "-lpthread",
            "-lm",
        ),
        cwd=paths.work_root,
        environment=environment,
        log_path=paths.build_log,
    )
    _write_sqlite_pkg_config(paths)
    return versioned_library


def _build_python(
    source_root: Path,
    paths: BuildPaths,
    *,
    jobs: int,
    environment: Mapping[str, str],
) -> Path:
    configure = source_root / "configure"
    _run_logged(
        (
            str(configure),
            f"--prefix={paths.prefix}",
            "--enable-shared",
            "--with-ensurepip=install",
            "--with-system-ffi",
        ),
        cwd=source_root,
        environment=environment,
        log_path=paths.build_log,
    )
    _run_logged(
        ("make", f"-j{jobs}"),
        cwd=source_root,
        environment=environment,
        log_path=paths.build_log,
    )
    _run_logged(
        ("make", "install"),
        cwd=source_root,
        environment=environment,
        log_path=paths.build_log,
    )
    executable = paths.prefix / "bin" / "python3.11"
    if not executable.is_file() or executable.is_symlink():
        raise RuntimeBuildError("CPython install did not produce python3.11")
    return executable


def _remove_cpython_site_packages_readme(paths: BuildPaths) -> Mapping[str, object]:
    """Verify and remove CPython's one non-wheel site-packages artifact."""

    candidate = paths.prefix.joinpath(
        *PurePosixPath(CPYTHON_SITE_PACKAGES_README_RELATIVE_PATH).parts
    )
    try:
        lexical = assert_no_symlink_ancestors(candidate)
        metadata = lexical.lstat()
    except (OSError, RuntimeBuildError) as error:
        raise RuntimeBuildError(
            "CPython site-packages README cannot be verified"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeBuildError(
            "CPython site-packages README is not a regular non-symlink file"
        )
    if (
        metadata.st_size != CPYTHON_SITE_PACKAGES_README_BYTES
        or sha256_file(lexical) != CPYTHON_SITE_PACKAGES_README_SHA256
    ):
        raise RuntimeBuildError(
            "CPython site-packages README differs from the pinned source archive"
        )
    lexical.unlink()
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(os.fspath(lexical.parent), directory_flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise RuntimeBuildError(
            "CPython site-packages README removal could not be directory-fsynced"
        ) from error
    if lexical.exists() or lexical.is_symlink():
        raise RuntimeBuildError("CPython site-packages README removal did not persist")
    return {
        "bytes": CPYTHON_SITE_PACKAGES_README_BYTES,
        "directory_fsync": True,
        "relative_path": CPYTHON_SITE_PACKAGES_README_RELATIVE_PATH,
        "sha256": CPYTHON_SITE_PACKAGES_README_SHA256,
    }


def _install_locked_wheels(
    executable: Path,
    paths: BuildPaths,
    *,
    environment: Mapping[str, str],
    lock_path: Path,
    locked_distribution_count: int,
) -> tuple[Mapping[str, object], ...]:
    paths.wheelhouse.mkdir(parents=True, exist_ok=False)
    _run_logged(
        (
            str(executable),
            "-I",
            "-m",
            "pip",
            "download",
            "--require-hashes",
            "--only-binary=:all:",
            "--no-deps",
            "--no-cache-dir",
            "--dest",
            str(paths.wheelhouse),
            "-r",
            str(lock_path),
        ),
        cwd=paths.repository_root,
        environment=environment,
        log_path=paths.build_log,
    )
    wheel_entries = tuple(paths.wheelhouse.iterdir())
    if any(
        entry.is_symlink() or not entry.is_file() or entry.suffix != ".whl"
        for entry in wheel_entries
    ):
        raise RuntimeBuildError("wheelhouse contains a non-wheel or unsafe artifact")
    wheel_rows = tuple(
        {
            "bytes": path.stat().st_size,
            "filename": path.name,
            "sha256": sha256_file(path),
        }
        for path in sorted(wheel_entries, key=lambda item: item.name)
    )
    if len(wheel_rows) != locked_distribution_count:
        raise RuntimeBuildError(
            "hash-locked wheel selection does not contain exactly one wheel "
            "per locked distribution"
        )
    _run_logged(
        (
            str(executable),
            "-I",
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--only-binary=:all:",
            "--no-deps",
            "--no-cache-dir",
            "--no-index",
            "--find-links",
            str(paths.wheelhouse),
            "--report",
            str(paths.pip_report),
            "-r",
            str(lock_path),
        ),
        cwd=paths.repository_root,
        environment=environment,
        log_path=paths.build_log,
    )
    if paths.pip_report.is_symlink() or not paths.pip_report.is_file():
        raise RuntimeBuildError("pip did not emit its fresh install report")
    return wheel_rows


def _parse_ldd_resolution(output: str, *, soname: str) -> Path:
    """Resolve one exact GNU ldd SONAME row rather than substring matching."""

    matches: list[str] = []
    for raw_line in output.splitlines():
        fields = raw_line.strip().split()
        if len(fields) >= 3 and fields[0] == soname and fields[1] == "=>":
            matches.append(fields[2])
    if len(matches) != 1 or matches[0] == "not":
        raise RuntimeBuildError(f"ldd did not resolve exactly one {soname} row")
    raw_path = matches[0]
    if not Path(raw_path).is_absolute():
        raise RuntimeBuildError(f"ldd returned a non-absolute path for {soname}")
    try:
        path = Path(raw_path).resolve(strict=True)
    except OSError as error:
        raise RuntimeBuildError(f"ldd path for {soname} is absent") from error
    if not path.is_file():
        raise RuntimeBuildError(f"ldd path for {soname} is not a regular file")
    return path


def _readelf_search_paths(
    binary: Path,
    *,
    environment: Mapping[str, str],
    expected_directory: Path,
) -> tuple[str, ...]:
    output = _capture(("readelf", "-d", str(binary)), environment=environment)
    rows: list[str] = []
    for line in output.splitlines():
        if "(RUNPATH)" not in line and "(RPATH)" not in line:
            continue
        match = re.search(r"\[([^\]]*)\]", line)
        if match is None:
            raise RuntimeBuildError("readelf emitted a malformed RUNPATH/RPATH row")
        rows.extend(item for item in match.group(1).split(":") if item)
    expected = str(expected_directory.resolve(strict=True))
    if expected not in rows:
        raise RuntimeBuildError(
            f"runtime artifact lacks governed library RUNPATH/RPATH: {binary}"
        )
    return tuple(rows)


_PROBE_SOURCE = r"""
import importlib.metadata
import json
from pathlib import Path
import platform
import sqlite3
import sys
import unicodedata
import zstandard

expected = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
locked = {name.casefold().replace("_", "-"): version for name, version in expected["locked"]}
observed = {
    distribution.metadata["Name"].casefold().replace("_", "-"): distribution.version
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}
extra = sorted(set(observed) - set(locked) - {"pip"})
missing = sorted(set(locked) - set(observed))
wrong = sorted(name for name in locked if observed.get(name) not in {None, locked[name]})
source_id = sqlite3.connect(":memory:").execute("select sqlite_source_id()").fetchone()[0]
payload = {
    "extra_distributions": extra,
    "locked_distributions": [[name, observed[name]] for name in sorted(locked) if name in observed],
    "missing_distributions": missing,
    "python_version": platform.python_version(),
    "python_prefix": sys.prefix,
    "sqlite_extension_path": __import__("_sqlite3").__file__,
    "sqlite_source_id": source_id,
    "sqlite_version": sqlite3.sqlite_version,
    "unicode_data_version": unicodedata.unidata_version,
    "wrong_distributions": wrong,
    "zstandard_package_version": zstandard.__version__,
    "libzstd_version": ".".join(str(value) for value in zstandard.ZSTD_VERSION),
}
print(json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")))
"""


def _runtime_probe(
    executable: Path,
    paths: BuildPaths,
    *,
    environment: Mapping[str, str],
    lock_path: Path,
) -> Mapping[str, object]:
    expected_path = paths.work_root / "runtime-expected.json"
    expected = {"locked": parse_locked_distributions(lock_path)}
    with expected_path.open("xb") as handle:
        handle.write(canonical_json_bytes(expected))
        handle.flush()
        os.fsync(handle.fileno())
    raw = _capture(
        (str(executable), "-I", "-B", "-c", _PROBE_SOURCE, str(expected_path)),
        environment=environment,
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeBuildError("target runtime probe did not emit strict JSON") from error
    expected_values = {
        "extra_distributions": [],
        "libzstd_version": LIBZSTD_VERSION,
        "missing_distributions": [],
        "python_version": PYTHON_VERSION,
        "python_prefix": str(paths.prefix),
        "sqlite_source_id": SQLITE_SOURCE_ID,
        "sqlite_version": SQLITE_VERSION,
        "unicode_data_version": UNICODE_VERSION,
        "wrong_distributions": [],
        "zstandard_package_version": ZSTANDARD_PACKAGE_VERSION,
    }
    if any(payload.get(key) != value for key, value in expected_values.items()):
        raise RuntimeBuildError("target runtime differs from the exact P-A contract")
    sqlite_extension = Path(str(payload.get("sqlite_extension_path"))).resolve(strict=True)
    library_directory = paths.sqlite_prefix / "lib"
    sqlite_ldd = _capture(("ldd", str(sqlite_extension)), environment=environment)
    sqlite_library = _parse_ldd_resolution(
        sqlite_ldd, soname="libsqlite3.so.0"
    )
    expected_sqlite_library = (
        library_directory / "libsqlite3.so.0.8.6"
    ).resolve(strict=True)
    if sqlite_library != expected_sqlite_library:
        raise RuntimeBuildError("CPython _sqlite3 resolved a non-governed SQLite build")
    executable_ldd = _capture(("ldd", str(executable)), environment=environment)
    libpython = _parse_ldd_resolution(
        executable_ldd, soname="libpython3.11.so.1.0"
    )
    expected_libpython = (
        library_directory / "libpython3.11.so.1.0"
    ).resolve(strict=True)
    if libpython != expected_libpython:
        raise RuntimeBuildError("CPython executable resolved a non-governed libpython")
    executable_search_paths = _readelf_search_paths(
        executable,
        environment=environment,
        expected_directory=library_directory,
    )
    sqlite_extension_search_paths = _readelf_search_paths(
        sqlite_extension,
        environment=environment,
        expected_directory=library_directory,
    )
    return {
        **payload,
        "cpython_executable_ldd": executable_ldd.splitlines(),
        "cpython_executable_search_paths": executable_search_paths,
        "libpython_library_path": str(libpython),
        "libpython_library_sha256": sha256_file(libpython),
        "sqlite3_extension_search_paths": sqlite_extension_search_paths,
        "sqlite_extension_ldd": sqlite_ldd.splitlines(),
        "sqlite_extension_sha256": sha256_file(sqlite_extension),
        "sqlite_library_path": str(sqlite_library),
        "sqlite_library_sha256": sha256_file(sqlite_library),
    }


def _repository_attestation(
    executable: Path,
    paths: BuildPaths,
    *,
    environment: Mapping[str, str],
    lock_path: Path,
) -> Mapping[str, object]:
    source = (
        "import json,sys; from pathlib import Path; "
        "sys.path.insert(0,sys.argv[2]); "
        "from training.weft1_corpus_pa import attest_runtime_v3; "
        "a=attest_runtime_v3(requirements_lock=Path(sys.argv[1])); "
        "print(json.dumps({'dependency_lock_sha256':a.dependency_lock_sha256,"
        "'environment_identity_sha256':a.environment_identity_sha256,"
        "'executable_sha256':a.executable_sha256,'environment_payload':a.environment_payload},"
        "allow_nan=False,sort_keys=True,separators=(',',':')))"
    )
    raw = _capture(
        (
            str(executable),
            "-I",
            "-B",
            "-c",
            source,
            str(lock_path),
            str(paths.repository_root),
        ),
        environment=environment,
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeBuildError("repository runtime attestation was not strict JSON") from error
    if payload.get("dependency_lock_sha256") != LOCK_SHA256:
        raise RuntimeBuildError("repository runtime attestation used the wrong lock")
    return payload


def _canonical_distribution_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeBuildError("installer report distribution name is absent")
    normalized = re.sub(r"[-_.]+", "-", value).casefold()
    if not normalized or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
        for character in normalized
    ):
        raise RuntimeBuildError("installer report distribution name is invalid")
    return normalized


def _trusted_installer_chain(
    *,
    report_path: Path,
    wheel_rows: Sequence[Mapping[str, object]],
    locked_distributions: Sequence[tuple[str, str]],
    installed_inventory: Mapping[str, object],
) -> Mapping[str, object]:
    """Bind pip's fresh install report to wheel hashes and the final tree."""

    try:
        raw_report = report_path.read_bytes()
        report = json.loads(raw_report.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeBuildError("pip install report is not strict JSON") from error
    if not isinstance(report, Mapping) or report.get("version") != "1":
        raise RuntimeBuildError("pip install report schema drifted")
    if report.get("pip_version") != "24.0":
        raise RuntimeBuildError("pip install report used a non-bootstrap pip")
    raw_install = report.get("install")
    if not isinstance(raw_install, list):
        raise RuntimeBuildError("pip install report lacks installation rows")
    locked = {
        (_canonical_distribution_name(name), version)
        for name, version in locked_distributions
    }
    selected_by_filename = {
        str(row["filename"]): {
            "bytes": row["bytes"],
            "filename": row["filename"],
            "sha256": row["sha256"],
        }
        for row in wheel_rows
    }
    installations: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for raw_row in raw_install:
        if not isinstance(raw_row, Mapping):
            raise RuntimeBuildError("pip install report row is invalid")
        metadata_value = raw_row.get("metadata")
        download = raw_row.get("download_info")
        if not isinstance(metadata_value, Mapping) or not isinstance(download, Mapping):
            raise RuntimeBuildError("pip install report row lacks metadata")
        name = _canonical_distribution_name(metadata_value.get("name"))
        version = metadata_value.get("version")
        if not isinstance(version, str) or not version:
            raise RuntimeBuildError("pip install report version is absent")
        key = (name, version)
        if key not in locked or key in seen:
            raise RuntimeBuildError("pip install report is outside the lock closure")
        url = download.get("url")
        archive = download.get("archive_info")
        if not isinstance(url, str) or not isinstance(archive, Mapping):
            raise RuntimeBuildError("pip install report lacks archive provenance")
        parsed = urllib.parse.urlparse(url)
        filename = Path(urllib.parse.unquote(parsed.path)).name
        selected = selected_by_filename.get(filename)
        hashes = archive.get("hashes")
        declared_hash = archive.get("hash")
        if (
            selected is None
            or not isinstance(hashes, Mapping)
            or hashes.get("sha256") != selected["sha256"]
            or declared_hash != f"sha256={selected['sha256']}"
        ):
            raise RuntimeBuildError(
                "pip install report archive differs from the selected wheel"
            )
        installations.append(
            {
                "distribution": name,
                "version": version,
                "wheel_bytes": selected["bytes"],
                "wheel_filename": filename,
                "wheel_sha256": selected["sha256"],
            }
        )
        seen.add(key)
    installations.sort(key=lambda row: str(row["distribution"]))
    if seen != locked or len(installations) != len(selected_by_filename):
        raise RuntimeBuildError("pip install report does not cover the lock exactly")

    distributions = installed_inventory.get("distributions")
    files = installed_inventory.get("files")
    if not isinstance(distributions, (list, tuple)) or not isinstance(
        files, (list, tuple)
    ):
        raise RuntimeBuildError("installed inventory is absent from installer chain")
    pip_rows = [
        dict(row)
        for row in distributions
        if isinstance(row, Mapping) and row.get("distribution") == "pip"
    ]
    if (
        len(pip_rows) != 1
        or pip_rows[0].get("version") != "24.0"
        or pip_rows[0].get("source") != "cpython_ensurepip"
    ):
        raise RuntimeBuildError("bootstrap pip inventory differs from CPython ensurepip")
    pip_files = [
        dict(row)
        for row in files
        if isinstance(row, Mapping)
        and isinstance(row.get("owners"), (list, tuple))
        and "pip" in row["owners"]
    ]
    pip_projection = {"distribution": pip_rows[0], "files": pip_files}
    core: dict[str, object] = {
        "bootstrap_pip_inventory_identity_sha256": sha256_bytes(
            canonical_json_bytes(pip_projection)
        ),
        "bootstrap_pip_version": "24.0",
        "installations": installations,
        "pip_report_sha256": sha256_bytes(raw_report),
        "schema": TRUSTED_INSTALLER_CHAIN_SCHEMA,
        "threat_model": TRUSTED_INSTALLER_THREAT_MODEL,
    }
    core["chain_identity_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {"domain": TRUSTED_INSTALLER_CHAIN_SCHEMA, "chain": core}
        )
    )
    return core


def _write_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeBuildError("runtime build receipt path must be fresh")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(receipt)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def verify_build_receipt_payload(receipt: Mapping[str, object]) -> str:
    """Independently verify the canonical receipt's complete internal binding."""

    if not isinstance(receipt, Mapping) or set(receipt) != {
        "authoritative",
        "evidence",
        "receipt_identity_sha256",
        "schema",
        "status",
    }:
        raise RuntimeBuildError("runtime receipt has an unexpected top-level shape")
    if (
        receipt.get("authoritative") is not True
        or receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "PASS"
    ):
        raise RuntimeBuildError("runtime receipt does not assert an authoritative pass")
    evidence = receipt.get("evidence")
    if not isinstance(evidence, Mapping):
        raise RuntimeBuildError("runtime receipt evidence is absent")
    expected_identity = sha256_bytes(
        canonical_json_bytes(
            {"domain": RECEIPT_SCHEMA, "evidence": evidence}
        )
    )
    if receipt.get("receipt_identity_sha256") != expected_identity:
        raise RuntimeBuildError("runtime receipt identity differs from its evidence")

    recipe = evidence.get("recipe")
    if canonical_json_bytes(recipe) != canonical_json_bytes(_abstract_recipe()):
        raise RuntimeBuildError("runtime receipt recipe differs from this builder")
    if evidence.get("recipe_identity_sha256") != recipe_identity_sha256():
        raise RuntimeBuildError("runtime receipt recipe identity is invalid")
    expected_readme_removal = {
        "bytes": CPYTHON_SITE_PACKAGES_README_BYTES,
        "directory_fsync": True,
        "relative_path": CPYTHON_SITE_PACKAGES_README_RELATIVE_PATH,
        "sha256": CPYTHON_SITE_PACKAGES_README_SHA256,
    }
    if evidence.get("cpython_site_packages_readme_removal") != expected_readme_removal:
        raise RuntimeBuildError(
            "runtime receipt lacks the exact CPython README removal"
        )
    lock = evidence.get("requirements_lock")
    if (
        not isinstance(lock, Mapping)
        or lock.get("sha256") != LOCK_SHA256
        or type(lock.get("bytes")) is not int
        or int(lock["bytes"]) < 1
    ):
        raise RuntimeBuildError("runtime receipt does not bind the governed lock")
    locked = evidence.get("locked_distributions")
    if not isinstance(locked, (list, tuple)) or not locked:
        raise RuntimeBuildError("runtime receipt lacks locked distributions")
    normalized_locked: list[tuple[str, str]] = []
    for row in locked:
        if (
            not isinstance(row, (list, tuple))
            or len(row) != 2
            or not all(isinstance(value, str) and value for value in row)
        ):
            raise RuntimeBuildError("runtime receipt has an invalid locked distribution")
        normalized_locked.append((row[0], row[1]))
    normalized_names = tuple(
        name.casefold().replace("_", "-") for name, _ in normalized_locked
    )
    if len(normalized_names) != len(set(normalized_names)):
        raise RuntimeBuildError("runtime receipt repeats a locked distribution")

    source_rows = evidence.get("sources")
    if not isinstance(source_rows, (list, tuple)) or len(source_rows) != len(SOURCE_PINS):
        raise RuntimeBuildError("runtime receipt source evidence is incomplete")
    for row, pin in zip(source_rows, SOURCE_PINS, strict=True):
        expected_source = {
            "bytes": pin.byte_count,
            "filename": pin.filename,
            "name": pin.name,
            "sha256": pin.sha256,
            "sha3_256": pin.sha3_256,
            "url": pin.url,
        }
        if canonical_json_bytes(row) != canonical_json_bytes(expected_source):
            raise RuntimeBuildError("runtime receipt source evidence differs from its pin")

    wheels = evidence.get("wheelhouse")
    if not isinstance(wheels, (list, tuple)) or len(wheels) != len(locked):
        raise RuntimeBuildError("runtime receipt wheel selection is incomplete")
    wheel_names: list[str] = []
    for row in wheels:
        if not isinstance(row, Mapping) or set(row) != {"bytes", "filename", "sha256"}:
            raise RuntimeBuildError("runtime receipt has an invalid wheel row")
        filename = row.get("filename")
        byte_count = row.get("bytes")
        sha256 = row.get("sha256")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".whl")
            or type(byte_count) is not int
            or byte_count < 1
            or not isinstance(sha256, str)
            or _SHA256.fullmatch(sha256) is None
        ):
            raise RuntimeBuildError("runtime receipt wheel row is not canonical")
        wheel_names.append(filename)
    if wheel_names != sorted(wheel_names) or len(wheel_names) != len(set(wheel_names)):
        raise RuntimeBuildError("runtime receipt wheel rows are not unique canonical order")
    if evidence.get("wheelhouse_identity_sha256") != sha256_bytes(
        canonical_json_bytes(wheels)
    ):
        raise RuntimeBuildError("runtime receipt wheelhouse identity is invalid")

    artifacts = evidence.get("artifacts")
    attestation = evidence.get("repository_runtime_attestation")
    probe = evidence.get("runtime_probe")
    if not isinstance(artifacts, Mapping) or not isinstance(attestation, Mapping):
        raise RuntimeBuildError("runtime receipt artifact attestation is absent")
    if not isinstance(probe, Mapping):
        raise RuntimeBuildError("runtime receipt probe is absent")
    if (
        attestation.get("dependency_lock_sha256") != LOCK_SHA256
        or attestation.get("executable_sha256")
        != artifacts.get("cpython_executable_sha256")
    ):
        raise RuntimeBuildError("runtime receipt attestation is not artifact-bound")
    environment_identity = attestation.get("environment_identity_sha256")
    environment_payload = attestation.get("environment_payload")
    if (
        not isinstance(environment_identity, str)
        or _SHA256.fullmatch(environment_identity) is None
        or not isinstance(environment_payload, Mapping)
        or environment_payload.get("dependency_lock_sha256") != LOCK_SHA256
        or environment_payload.get("python_executable_sha256")
        != artifacts.get("cpython_executable_sha256")
    ):
        raise RuntimeBuildError("runtime receipt environment attestation is incomplete")
    runtime_linkage = environment_payload.get("runtime_linkage")
    if not isinstance(runtime_linkage, Mapping):
        raise RuntimeBuildError("runtime receipt lacks live linkage attestation")
    linkage_rows = {
        "executable": "cpython_executable_sha256",
        "libpython_library": "libpython_library_sha256",
        "sqlite_extension": "sqlite3_extension_sha256",
        "sqlite_library": "sqlite3_library_sha256",
    }
    for linkage_name, artifact_name in linkage_rows.items():
        row = runtime_linkage.get(linkage_name)
        if (
            not isinstance(row, Mapping)
            or row.get("sha256") != artifacts.get(artifact_name)
            or not isinstance(row.get("path"), str)
            or not row.get("path")
            or type(row.get("bytes")) is not int
            or int(row["bytes"]) < 1
        ):
            raise RuntimeBuildError(
                f"runtime receipt {linkage_name} is not artifact-bound"
            )
    installed_inventory = evidence.get("installed_distribution_inventory")
    if (
        not isinstance(installed_inventory, Mapping)
        or canonical_json_bytes(installed_inventory)
        != canonical_json_bytes(
            environment_payload.get("installed_distribution_inventory")
        )
        or installed_inventory.get("schema")
        != INSTALLED_DISTRIBUTION_INVENTORY_SCHEMA
    ):
        raise RuntimeBuildError(
            "runtime receipt installed-distribution inventory is not attestation-bound"
        )
    inventory_core = dict(installed_inventory)
    claimed_inventory_identity = inventory_core.pop(
        "inventory_identity_sha256", None
    )
    if (
        not isinstance(claimed_inventory_identity, str)
        or _SHA256.fullmatch(claimed_inventory_identity) is None
        or claimed_inventory_identity
        != sha256_bytes(
            canonical_json_bytes(
                {
                    "domain": INSTALLED_DISTRIBUTION_INVENTORY_SCHEMA,
                    "inventory": inventory_core,
                }
            )
        )
    ):
        raise RuntimeBuildError(
            "runtime receipt installed-distribution inventory identity is invalid"
        )
    distribution_rows = environment_payload.get("distributions")
    if not isinstance(distribution_rows, (list, tuple)):
        raise RuntimeBuildError("runtime receipt lacks full lock-artifact observations")
    allowed_wheels: dict[tuple[str, str], frozenset[str]] = {}
    for row in distribution_rows:
        if not isinstance(row, Mapping):
            raise RuntimeBuildError("runtime receipt has an invalid distribution observation")
        name = row.get("distribution")
        version = row.get("version")
        hashes = row.get("artifact_sha256s")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            or not isinstance(hashes, (list, tuple))
            or not hashes
            or any(
                not isinstance(item, str) or _SHA256.fullmatch(item) is None
                for item in hashes
            )
        ):
            raise RuntimeBuildError("runtime receipt has an invalid lock-artifact row")
        key = (name.casefold().replace("_", "-"), version)
        if key in allowed_wheels:
            raise RuntimeBuildError("runtime receipt repeats a lock-artifact row")
        allowed_wheels[key] = frozenset(hashes)
    locked_keys = {
        (name.casefold().replace("_", "-"), version)
        for name, version in normalized_locked
    }
    if set(allowed_wheels) != locked_keys:
        raise RuntimeBuildError("runtime receipt lock-artifact rows are incomplete")
    selected_keys: set[tuple[str, str]] = set()
    for row in wheels:
        filename = str(row["filename"])
        wheel_parts = filename[:-4].split("-")
        if len(wheel_parts) < 5:
            raise RuntimeBuildError("runtime receipt has a malformed wheel filename")
        key = (
            wheel_parts[0].casefold().replace("_", "-").replace(".", "-"),
            wheel_parts[1],
        )
        if (
            key not in allowed_wheels
            or str(row["sha256"]) not in allowed_wheels[key]
            or key in selected_keys
        ):
            raise RuntimeBuildError("selected wheel is not uniquely authorized by the lock")
        selected_keys.add(key)
    if selected_keys != locked_keys:
        raise RuntimeBuildError("selected wheels do not cover the complete lock")
    installer_chain = evidence.get("trusted_installer_chain")
    if not isinstance(installer_chain, Mapping) or set(installer_chain) != {
        "bootstrap_pip_inventory_identity_sha256",
        "bootstrap_pip_version",
        "chain_identity_sha256",
        "installations",
        "pip_report_sha256",
        "schema",
        "threat_model",
    }:
        raise RuntimeBuildError("runtime receipt trusted-installer chain is incomplete")
    chain_core = dict(installer_chain)
    claimed_chain_identity = chain_core.pop("chain_identity_sha256", None)
    if (
        installer_chain.get("schema") != TRUSTED_INSTALLER_CHAIN_SCHEMA
        or installer_chain.get("threat_model") != TRUSTED_INSTALLER_THREAT_MODEL
        or installer_chain.get("bootstrap_pip_version") != "24.0"
        or not isinstance(claimed_chain_identity, str)
        or claimed_chain_identity
        != sha256_bytes(
            canonical_json_bytes(
                {"domain": TRUSTED_INSTALLER_CHAIN_SCHEMA, "chain": chain_core}
            )
        )
    ):
        raise RuntimeBuildError("runtime receipt trusted-installer identity is invalid")
    pip_report_sha256 = installer_chain.get("pip_report_sha256")
    if not isinstance(pip_report_sha256, str) or _SHA256.fullmatch(
        pip_report_sha256
    ) is None:
        raise RuntimeBuildError("runtime receipt pip report SHA-256 is invalid")
    raw_installations = installer_chain.get("installations")
    if not isinstance(raw_installations, (list, tuple)):
        raise RuntimeBuildError("runtime receipt lacks installer rows")
    installation_keys: set[tuple[str, str]] = set()
    selected_by_filename = {str(row["filename"]): row for row in wheels}
    prior_distribution = ""
    for row in raw_installations:
        if not isinstance(row, Mapping) or set(row) != {
            "distribution",
            "version",
            "wheel_bytes",
            "wheel_filename",
            "wheel_sha256",
        }:
            raise RuntimeBuildError("runtime receipt has an invalid installer row")
        distribution = str(row.get("distribution"))
        version = str(row.get("version"))
        filename = str(row.get("wheel_filename"))
        selected = selected_by_filename.get(filename)
        key = (distribution, version)
        if (
            distribution <= prior_distribution
            or key not in locked_keys
            or key in installation_keys
            or selected is None
            or row.get("wheel_bytes") != selected.get("bytes")
            or row.get("wheel_sha256") != selected.get("sha256")
        ):
            raise RuntimeBuildError("runtime receipt installer coverage drifted")
        prior_distribution = distribution
        installation_keys.add(key)
    if installation_keys != locked_keys:
        raise RuntimeBuildError("runtime receipt installer closure is incomplete")
    inventory_distributions = installed_inventory.get("distributions")
    inventory_files = installed_inventory.get("files")
    if not isinstance(inventory_distributions, (list, tuple)) or not isinstance(
        inventory_files, (list, tuple)
    ):
        raise RuntimeBuildError("runtime receipt installed inventory is incomplete")
    pip_rows = [
        dict(row)
        for row in inventory_distributions
        if isinstance(row, Mapping) and row.get("distribution") == "pip"
    ]
    pip_files = [
        dict(row)
        for row in inventory_files
        if isinstance(row, Mapping)
        and isinstance(row.get("owners"), (list, tuple))
        and "pip" in row["owners"]
    ]
    if (
        len(pip_rows) != 1
        or pip_rows[0].get("version") != "24.0"
        or pip_rows[0].get("source") != "cpython_ensurepip"
        or installer_chain.get("bootstrap_pip_inventory_identity_sha256")
        != sha256_bytes(
        canonical_json_bytes({"distribution": pip_rows[0], "files": pip_files})
        )
    ):
        raise RuntimeBuildError("runtime receipt bootstrap pip inventory drifted")
    expected_probe = {
        "libzstd_version": LIBZSTD_VERSION,
        "python_version": PYTHON_VERSION,
        "sqlite_source_id": SQLITE_SOURCE_ID,
        "sqlite_version": SQLITE_VERSION,
        "unicode_data_version": UNICODE_VERSION,
        "zstandard_package_version": ZSTANDARD_PACKAGE_VERSION,
    }
    if any(probe.get(key) != value for key, value in expected_probe.items()):
        raise RuntimeBuildError("runtime receipt probe differs from authority")
    if (
        probe.get("libpython_library_sha256")
        != artifacts.get("libpython_library_sha256")
        or probe.get("sqlite_extension_sha256")
        != artifacts.get("sqlite3_extension_sha256")
        or probe.get("sqlite_library_sha256")
        != artifacts.get("sqlite3_library_sha256")
        or probe.get("libpython_library_path")
        != runtime_linkage.get("libpython_library", {}).get("path")
        or probe.get("sqlite_extension_path")
        != runtime_linkage.get("sqlite_extension", {}).get("path")
        or probe.get("sqlite_library_path")
        != runtime_linkage.get("sqlite_library", {}).get("path")
    ):
        raise RuntimeBuildError("runtime receipt probe linkage differs from attestation")
    for key in ("builder_sha256", "runtime_contract_sha256"):
        value = evidence.get(key)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise RuntimeBuildError(f"runtime receipt lacks a canonical {key}")
    return expected_identity


def build_runtime(paths: BuildPaths, *, jobs: int) -> Mapping[str, object]:
    if type(jobs) is not int or jobs < 1 or jobs > 256:
        raise RuntimeBuildError("jobs must be an integer in [1, 256]")
    paths = _require_fresh_nonoverlapping_paths(paths)
    builder_path = Path(__file__).resolve(strict=True)
    runtime_contract_path = (
        paths.repository_root / RUNTIME_CONTRACT_RELATIVE_PATH
    ).resolve(strict=True)
    builder_sha256 = sha256_file(builder_path)
    runtime_contract_sha256 = sha256_file(runtime_contract_path)
    host = _host_preflight()
    environment = _deterministic_environment(paths)
    dependency_versions = _dependency_versions(environment)
    paths.work_root.mkdir(parents=True, exist_ok=False)
    paths.downloads.mkdir()
    paths.sources.mkdir()
    lock_receipt = snapshot_verified_lock(paths.lock_path, paths.lock_snapshot)
    locked_distributions = parse_locked_distributions(paths.lock_snapshot)

    source_receipts = []
    for pin in SOURCE_PINS:
        print(f"[weft1-runtime] downloading and verifying {pin.name}", flush=True)
        source_receipts.append(
            download_source(pin, paths.downloads / pin.filename)
        )
    python_source_root = safe_extract_tar_xz(
        paths.downloads / PYTHON_SOURCE.filename,
        paths.sources / "cpython",
    )
    sqlite_source_root = safe_extract_sqlite_zip(
        paths.downloads / SQLITE_SOURCE.filename,
        paths.sources / "sqlite",
    )
    print("[weft1-runtime] building pinned SQLite", flush=True)
    sqlite_library = _build_sqlite(
        sqlite_source_root,
        paths,
        environment=environment,
    )
    print("[weft1-runtime] building CPython 3.11.9", flush=True)
    executable = _build_python(
        python_source_root,
        paths,
        jobs=jobs,
        environment=environment,
    )
    readme_removal = _remove_cpython_site_packages_readme(paths)
    runtime_environment = _self_contained_runtime_environment(environment)
    print("[weft1-runtime] resolving hash-locked wheelhouse", flush=True)
    wheel_rows = _install_locked_wheels(
        executable,
        paths,
        environment=runtime_environment,
        lock_path=paths.lock_snapshot,
        locked_distribution_count=len(locked_distributions),
    )
    print("[weft1-runtime] running exact runtime attestation", flush=True)
    probe = _runtime_probe(
        executable,
        paths,
        environment=runtime_environment,
        lock_path=paths.lock_snapshot,
    )
    attestation = _repository_attestation(
        executable,
        paths,
        environment=runtime_environment,
        lock_path=paths.lock_snapshot,
    )
    if sha256_file(builder_path) != builder_sha256:
        raise RuntimeBuildError("runtime builder changed during execution")
    if sha256_file(runtime_contract_path) != runtime_contract_sha256:
        raise RuntimeBuildError("runtime contract changed during execution")
    if verify_lock(paths.lock_snapshot) != LOCK_SHA256:
        raise RuntimeBuildError("requirements-lock snapshot changed during execution")
    sqlite_extension = Path(str(probe["sqlite_extension_path"]))
    libpython_library = Path(str(probe["libpython_library_path"]))
    recipe = _abstract_recipe()
    wheelhouse_identity = sha256_bytes(canonical_json_bytes(wheel_rows))
    attested_environment = attestation.get("environment_payload")
    if not isinstance(attested_environment, Mapping) or not isinstance(
        attested_environment.get("installed_distribution_inventory"), Mapping
    ):
        raise RuntimeBuildError(
            "repository attestation lacks installed-distribution integrity"
        )
    trusted_installer_chain = _trusted_installer_chain(
        report_path=paths.pip_report,
        wheel_rows=wheel_rows,
        locked_distributions=locked_distributions,
        installed_inventory=attested_environment["installed_distribution_inventory"],
    )
    evidence = {
        "artifacts": {
            "build_log_sha256": sha256_file(paths.build_log),
            "cpython_executable_sha256": sha256_file(executable),
            "libpython_library_sha256": sha256_file(libpython_library),
            "sqlite3_extension_sha256": sha256_file(sqlite_extension),
            "sqlite3_library_sha256": sha256_file(sqlite_library),
        },
        "build_dependency_versions": dependency_versions,
        "builder_sha256": builder_sha256,
        "cpython_site_packages_readme_removal": readme_removal,
        "host": host,
        "installed_distribution_inventory": attested_environment[
            "installed_distribution_inventory"
        ],
        "jobs": jobs,
        "locked_distributions": locked_distributions,
        "requirements_lock": lock_receipt,
        "prefix": str(paths.prefix),
        "recipe": recipe,
        "recipe_identity_sha256": sha256_bytes(canonical_json_bytes(recipe)),
        "repository_runtime_attestation": attestation,
        "runtime_contract_sha256": runtime_contract_sha256,
        "runtime_probe": probe,
        "sources": source_receipts,
        "trusted_installer_chain": trusted_installer_chain,
        "wheelhouse": wheel_rows,
        "wheelhouse_identity_sha256": wheelhouse_identity,
    }
    receipt = {
        "authoritative": True,
        "evidence": evidence,
        "receipt_identity_sha256": sha256_bytes(
            canonical_json_bytes(
                {
                    "domain": "weft1_pa_runtime_build_receipt_v1",
                    "evidence": evidence,
                }
            )
        ),
        "schema": RECEIPT_SCHEMA,
        "status": "PASS",
    }
    verify_build_receipt_payload(receipt)
    _write_receipt(paths.receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--work-root", type=Path, default=Path("/content/weft1-pa-build"))
    parser.add_argument("--prefix", type=Path, default=Path("/content/weft1-pa-runtime"))
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("/content/weft1-pa-runtime-receipt.json"),
    )
    parser.add_argument("--jobs", type=int, default=max(1, min(os.cpu_count() or 1, 16)))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact non-authoritative recipe without side effects",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        sys.stdout.buffer.write(canonical_json_bytes(dry_run_receipt()))
        sys.stdout.buffer.flush()
        return 0
    try:
        receipt = build_runtime(
            BuildPaths(
                repository_root=args.repository_root,
                work_root=args.work_root,
                prefix=args.prefix,
                receipt_path=args.receipt,
            ),
            jobs=args.jobs,
        )
    except (OSError, RuntimeBuildError, subprocess.SubprocessError) as error:
        failure = {
            "authoritative": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "failed_closed": True,
            "schema": RECEIPT_SCHEMA,
            "status": "FAIL",
        }
        sys.stderr.buffer.write(canonical_json_bytes(failure))
        sys.stderr.buffer.flush()
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(receipt))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
