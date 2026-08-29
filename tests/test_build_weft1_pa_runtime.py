from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from scripts import build_weft1_pa_runtime as builder
from training.weft1_corpus_pa import (
    DEFAULT_REQUIREMENTS_LOCK_SHA256,
    RuntimeExpectationV3,
)


ROOT = Path(__file__).resolve().parents[1]


def test_builder_literals_match_runtime_contract_and_checked_in_lock() -> None:
    expectation = RuntimeExpectationV3()
    assert builder.PYTHON_VERSION == expectation.python_version
    assert builder.UNICODE_VERSION == expectation.unicode_data_version
    assert builder.SQLITE_VERSION == expectation.sqlite_version
    assert builder.ZSTANDARD_PACKAGE_VERSION == expectation.zstandard_package_version
    assert builder.LIBZSTD_VERSION == expectation.libzstd_version
    assert builder.LOCK_SHA256 == DEFAULT_REQUIREMENTS_LOCK_SHA256
    lock = ROOT / builder.LOCK_RELATIVE_PATH
    assert hashlib.sha256(lock.read_bytes()).hexdigest() == builder.LOCK_SHA256
    assert builder.verify_lock(lock) == builder.LOCK_SHA256
    rows = builder.parse_locked_distributions(lock)
    assert ("zstandard", "0.25.0") in rows
    assert ("fasttext-wheel", "0.9.2") in rows
    assert len(rows) == len({name.casefold().replace("_", "-") for name, _ in rows})


def test_official_source_pins_and_sqlite_release_identity_are_exact() -> None:
    assert builder.PYTHON_SOURCE.url == (
        "https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tar.xz"
    )
    assert builder.PYTHON_SOURCE.sha256 == (
        "9b1e896523fc510691126c864406d9360a3d1e986acbda59cda57b5abda45b87"
    )
    assert builder.SQLITE_SOURCE.url == (
        "https://www.sqlite.org/2024/sqlite-amalgamation-3450100.zip"
    )
    assert builder.SQLITE3_C_SHA3_256 == (
        "0474604df9e1b69a5544295dd046aad954749279780d557da80f44b958100295"
    )
    assert builder.SQLITE_SOURCE_ID.endswith(
        "e876e51a0ed5c5b3126f52e532044363a014bc594cfefa87ffb5b82257cc467a"
    )
    for pin in builder.SOURCE_PINS:
        assert pin.url.startswith("https://")
        assert len(pin.sha256) == 64
        assert pin.byte_count > 1_000_000


def test_recipe_is_hash_stable_and_requires_offline_hash_locked_install() -> None:
    first = builder._abstract_recipe()
    second = builder._abstract_recipe()
    assert first == second
    assert builder.recipe_identity_sha256() == hashlib.sha256(
        builder.canonical_json_bytes(first)
    ).hexdigest()
    assert first["python_configure"] == (
        "./configure",
        "--prefix=${PREFIX}",
        "--enable-shared",
        "--with-ensurepip=install",
        "--with-system-ffi",
    )
    assert "--require-hashes" in first["pip_download_flags"]
    assert "--only-binary=:all:" in first["pip_download_flags"]
    assert "--no-deps" in first["pip_download_flags"]
    assert "--no-index" in first["pip_install_flags"]
    assert first["expected_runtime"]["sqlite_source_id"] == builder.SQLITE_SOURCE_ID
    assert first["sqlite_shared_object"] == "libsqlite3.so.0.8.6"
    assert first["sqlite_install_location"] == "${PREFIX}"
    assert any("import sqlite3 and _sqlite3" in row for row in first["runtime_linkage_checks"])
    assert any("ldd(_sqlite3)" in row for row in first["runtime_linkage_checks"])


def test_dry_run_is_canonical_and_performs_no_network_or_subprocess(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("dry-run attempted execution")

    monkeypatch.setattr(builder.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(builder.subprocess, "run", forbidden)
    assert builder.main(["--dry-run"]) == 0
    stdout, stderr = capfd.readouterr()
    assert stderr == ""
    raw = stdout.encode("utf-8")
    payload = json.loads(raw)
    assert raw == builder.canonical_json_bytes(payload)
    assert payload["status"] == "PLAN_ONLY_NO_EXECUTION"
    assert payload["authoritative"] is False


def test_source_verifier_rejects_size_and_hash_drift(tmp_path: Path) -> None:
    payload = b"source fixture"
    path = tmp_path / "source.tar.xz"
    path.write_bytes(payload)
    pin = builder.SourcePin(
        name="fixture",
        url="https://example.invalid/source.tar.xz",
        filename=path.name,
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert builder.verify_source_file(path, pin)["sha256"] == pin.sha256
    path.write_bytes(payload + b"!")
    with pytest.raises(builder.RuntimeBuildError, match="differ from their pin"):
        builder.verify_source_file(path, pin)


def test_lock_snapshot_is_one_fresh_hash_verified_artifact(tmp_path: Path) -> None:
    source = ROOT / builder.LOCK_RELATIVE_PATH
    destination = tmp_path / "requirements.lock"
    receipt = builder.snapshot_verified_lock(source, destination)
    assert receipt == {
        "bytes": len(source.read_bytes()),
        "filename": "requirements.lock",
        "sha256": builder.LOCK_SHA256,
        "source_filename": source.name,
    }
    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(builder.RuntimeBuildError, match="must be fresh"):
        builder.snapshot_verified_lock(source, destination)


def test_tar_extractor_rejects_links_before_writing_link_target(tmp_path: Path) -> None:
    archive = tmp_path / "python.tar.xz"
    with tarfile.open(archive, "w:xz") as bundle:
        root = tarfile.TarInfo("Python-3.11.9/")
        root.type = tarfile.DIRTYPE
        bundle.addfile(root)
        link = tarfile.TarInfo("Python-3.11.9/configure")
        link.type = tarfile.SYMTYPE
        link.linkname = "/bin/sh"
        bundle.addfile(link)
    destination = tmp_path / "tar-out"
    with pytest.raises(builder.RuntimeBuildError, match="link or special"):
        builder.safe_extract_tar_xz(archive, destination)
    assert not (destination / "Python-3.11.9" / "configure").exists()


def test_sqlite_zip_extractor_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "sqlite.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("sqlite-amalgamation-3450100/../../escape", b"bad")
    destination = tmp_path / "zip-out"
    with pytest.raises(builder.RuntimeBuildError, match="unsafe path"):
        builder.safe_extract_sqlite_zip(archive, destination)
    assert not (tmp_path / "escape").exists()


def test_receipt_writer_is_canonical_fresh_only(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    value = {"schema": builder.RECEIPT_SCHEMA, "status": "PASS"}
    builder._write_receipt(receipt, value)
    assert receipt.read_bytes() == builder.canonical_json_bytes(value)
    with pytest.raises(builder.RuntimeBuildError, match="must be fresh"):
        builder._write_receipt(receipt, value)


def _valid_build_receipt() -> dict[str, object]:
    wheel_rows = (
        {
            "bytes": 123,
            "filename": "fixture-1.0-py3-none-any.whl",
            "sha256": "a" * 64,
        },
    )
    sources = tuple(
        {
            "bytes": pin.byte_count,
            "filename": pin.filename,
            "name": pin.name,
            "sha256": pin.sha256,
            "sha3_256": pin.sha3_256,
            "url": pin.url,
        }
        for pin in builder.SOURCE_PINS
    )
    inventory_core = {
        "bootstrap_distributions": (
            {"distribution": "pip", "version": "24.0"},
        ),
        "distributions": (
            {
                "distribution": "fixture",
                "file_count": 1,
                "record_path": "lib/site-packages/fixture-1.0.dist-info/RECORD",
                "record_sha256": "6" * 64,
                "source": "hash_locked_wheel",
                "version": "1.0",
            },
            {
                "distribution": "pip",
                "file_count": 1,
                "record_path": "lib/site-packages/pip-24.0.dist-info/RECORD",
                "record_sha256": "7" * 64,
                "source": "cpython_ensurepip",
                "version": "24.0",
            },
        ),
        "files": (
            {
                "bytes": 1,
                "owners": ("fixture",),
                "relative_path": "lib/site-packages/fixture-1.0.dist-info/RECORD",
                "sha256": "6" * 64,
            },
            {
                "bytes": 1,
                "owners": ("pip",),
                "relative_path": "lib/site-packages/pip-24.0.dist-info/RECORD",
                "sha256": "7" * 64,
            },
        ),
        "installation_prefix": "/content/runtime",
        "schema": builder.INSTALLED_DISTRIBUTION_INVENTORY_SCHEMA,
        "site_roots": ("lib/site-packages",),
    }
    installed_inventory = {
        **inventory_core,
        "inventory_identity_sha256": builder.sha256_bytes(
            builder.canonical_json_bytes(
                {
                    "domain": builder.INSTALLED_DISTRIBUTION_INVENTORY_SCHEMA,
                    "inventory": inventory_core,
                }
            )
        ),
    }
    runtime_linkage_core = {
        "executable": {
            "bytes": 1,
            "path": "/content/runtime/bin/python3.11",
            "sha256": "b" * 64,
        },
        "libpython_library": {
            "bytes": 1,
            "path": "/content/runtime/lib/libpython3.11.so.1.0",
            "sha256": "1" * 64,
        },
        "schema": "weft1_runtime_linkage_v3",
        "sqlite_extension": {
            "bytes": 1,
            "path": "/content/runtime/lib/python3.11/lib-dynload/_sqlite3.so",
            "sha256": "2" * 64,
        },
        "sqlite_library": {
            "bytes": 1,
            "path": "/content/runtime/lib/libsqlite3.so.0.8.6",
            "sha256": "3" * 64,
        },
    }
    from training.weft1_corpus_a2 import execution_authority_v3_bound_sha256

    runtime_linkage = {
        **runtime_linkage_core,
        "linkage_identity_sha256": execution_authority_v3_bound_sha256(
            "weft1_runtime_linkage_v3", runtime_linkage_core
        ),
    }
    pip_projection = {
        "distribution": installed_inventory["distributions"][1],
        "files": [installed_inventory["files"][1]],
    }
    installer_core = {
        "bootstrap_pip_inventory_identity_sha256": builder.sha256_bytes(
            builder.canonical_json_bytes(pip_projection)
        ),
        "bootstrap_pip_version": "24.0",
        "installations": [
            {
                "distribution": "fixture",
                "version": "1.0",
                "wheel_bytes": 123,
                "wheel_filename": "fixture-1.0-py3-none-any.whl",
                "wheel_sha256": "a" * 64,
            }
        ],
        "pip_report_sha256": "8" * 64,
        "schema": builder.TRUSTED_INSTALLER_CHAIN_SCHEMA,
        "threat_model": builder.TRUSTED_INSTALLER_THREAT_MODEL,
    }
    installer_chain = {
        **installer_core,
        "chain_identity_sha256": builder.sha256_bytes(
            builder.canonical_json_bytes(
                {
                    "domain": builder.TRUSTED_INSTALLER_CHAIN_SCHEMA,
                    "chain": installer_core,
                }
            )
        ),
    }
    evidence = {
        "artifacts": {
            "cpython_executable_sha256": "b" * 64,
            "libpython_library_sha256": "1" * 64,
            "sqlite3_extension_sha256": "2" * 64,
            "sqlite3_library_sha256": "3" * 64,
        },
        "builder_sha256": "c" * 64,
        "cpython_site_packages_readme_removal": {
            "bytes": builder.CPYTHON_SITE_PACKAGES_README_BYTES,
            "directory_fsync": True,
            "relative_path": builder.CPYTHON_SITE_PACKAGES_README_RELATIVE_PATH,
            "sha256": builder.CPYTHON_SITE_PACKAGES_README_SHA256,
        },
        "locked_distributions": (("fixture", "1.0"),),
        "recipe": builder._abstract_recipe(),
        "recipe_identity_sha256": builder.recipe_identity_sha256(),
        "repository_runtime_attestation": {
            "dependency_lock_sha256": builder.LOCK_SHA256,
            "environment_identity_sha256": "e" * 64,
            "environment_payload": {
                "dependency_lock_sha256": builder.LOCK_SHA256,
                "distributions": (
                    {
                        "artifact_sha256s": ("a" * 64,),
                        "distribution": "fixture",
                        "version": "1.0",
                    },
                ),
                "installed_distribution_inventory": installed_inventory,
                "python_executable_sha256": "b" * 64,
                "runtime_linkage": runtime_linkage,
            },
            "executable_sha256": "b" * 64,
        },
        "installed_distribution_inventory": installed_inventory,
        "requirements_lock": {
            "bytes": 123,
            "filename": "requirements.lock",
            "sha256": builder.LOCK_SHA256,
            "source_filename": builder.LOCK_RELATIVE_PATH.name,
        },
        "runtime_contract_sha256": "d" * 64,
        "runtime_probe": {
            "libpython_library_path": runtime_linkage["libpython_library"]["path"],
            "libpython_library_sha256": "1" * 64,
            "libzstd_version": builder.LIBZSTD_VERSION,
            "python_version": builder.PYTHON_VERSION,
            "sqlite_source_id": builder.SQLITE_SOURCE_ID,
            "sqlite_extension_path": runtime_linkage["sqlite_extension"]["path"],
            "sqlite_extension_sha256": "2" * 64,
            "sqlite_library_path": runtime_linkage["sqlite_library"]["path"],
            "sqlite_library_sha256": "3" * 64,
            "sqlite_version": builder.SQLITE_VERSION,
            "unicode_data_version": builder.UNICODE_VERSION,
            "zstandard_package_version": builder.ZSTANDARD_PACKAGE_VERSION,
        },
        "sources": sources,
        "trusted_installer_chain": installer_chain,
        "wheelhouse": wheel_rows,
        "wheelhouse_identity_sha256": builder.sha256_bytes(
            builder.canonical_json_bytes(wheel_rows)
        ),
    }
    identity = builder.sha256_bytes(
        builder.canonical_json_bytes(
            {"domain": builder.RECEIPT_SCHEMA, "evidence": evidence}
        )
    )
    return {
        "authoritative": True,
        "evidence": evidence,
        "receipt_identity_sha256": identity,
        "schema": builder.RECEIPT_SCHEMA,
        "status": "PASS",
    }


def test_receipt_verifier_binds_exact_wheel_rows_and_recipe() -> None:
    receipt = _valid_build_receipt()
    assert builder.verify_build_receipt_payload(receipt) == receipt[
        "receipt_identity_sha256"
    ]
    evidence = receipt["evidence"]
    assert isinstance(evidence, dict)
    wheels = evidence["wheelhouse"]
    assert isinstance(wheels, tuple)
    wheels[0]["sha256"] = "e" * 64
    evidence["wheelhouse_identity_sha256"] = builder.sha256_bytes(
        builder.canonical_json_bytes(wheels)
    )
    receipt["receipt_identity_sha256"] = builder.sha256_bytes(
        builder.canonical_json_bytes(
            {"domain": builder.RECEIPT_SCHEMA, "evidence": evidence}
        )
    )
    with pytest.raises(builder.RuntimeBuildError, match="wheelhouse identity"):
        evidence["wheelhouse_identity_sha256"] = "f" * 64
        receipt["receipt_identity_sha256"] = builder.sha256_bytes(
            builder.canonical_json_bytes(
                {"domain": builder.RECEIPT_SCHEMA, "evidence": evidence}
            )
        )
        builder.verify_build_receipt_payload(receipt)


def test_receipt_verifier_binds_installed_inventory_to_runtime_attestation() -> None:
    receipt = _valid_build_receipt()
    evidence = receipt["evidence"]
    assert isinstance(evidence, dict)
    inventory = evidence["installed_distribution_inventory"]
    assert isinstance(inventory, dict)
    files = inventory["files"]
    assert isinstance(files, tuple)
    files[0]["sha256"] = "9" * 64
    receipt["receipt_identity_sha256"] = builder.sha256_bytes(
        builder.canonical_json_bytes(
            {"domain": builder.RECEIPT_SCHEMA, "evidence": evidence}
        )
    )
    with pytest.raises(builder.RuntimeBuildError, match="inventory identity is invalid"):
        builder.verify_build_receipt_payload(receipt)


def test_missing_sqlite_extension_fails_runtime_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = builder.BuildPaths(
        repository_root=ROOT,
        work_root=tmp_path,
        prefix=tmp_path / "runtime",
        receipt_path=tmp_path / "receipt.json",
    )
    assert paths.sqlite_prefix == paths.prefix

    def missing_sqlite(*_args: object, **_kwargs: object) -> str:
        raise builder.RuntimeBuildError("inspection command failed: no _sqlite3")

    monkeypatch.setattr(builder, "_capture", missing_sqlite)
    with pytest.raises(builder.RuntimeBuildError, match="no _sqlite3"):
        builder._runtime_probe(
            tmp_path / "python3.11",
            paths,
            environment={},
            lock_path=ROOT / builder.LOCK_RELATIVE_PATH,
        )


def test_colab_wrapper_and_python_builder_share_dependency_set() -> None:
    wrapper = (ROOT / "scripts/run_weft1_pa_runtime_builder_colab.sh").read_text(
        encoding="utf-8"
    )
    for package in builder.BUILD_DEPENDENCY_PACKAGES:
        assert f"  {package}\n" in wrapper
    assert '"${APT[@]}" install -y --no-install-recommends' in wrapper
    assert "env -u LD_LIBRARY_PATH -u PYTHONHOME -u PYTHONPATH" in wrapper
    assert 'python3 -I -B "${BUILDER}" --repository-root' in wrapper


def test_dpkg_provenance_uses_unqualified_canonical_package_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_capture(
        command: tuple[str, ...], *, environment: dict[str, str]
    ) -> str:
        assert "-f=${Package}\t${Version}\\n" in command
        assert environment == {"LANG": "C.UTF-8"}
        return "\n".join(
            f"{package}\tfixture-version" for package in builder.BUILD_DEPENDENCY_PACKAGES
        )

    monkeypatch.setattr(builder, "_capture", fake_capture)
    observed = builder._dependency_versions({"LANG": "C.UTF-8"})
    assert observed == {
        package: "fixture-version" for package in builder.BUILD_DEPENDENCY_PACKAGES
    }


def test_exact_cpython_site_packages_readme_is_removed_and_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = builder.BuildPaths(
        repository_root=ROOT,
        work_root=tmp_path / "work",
        prefix=tmp_path / "runtime",
        receipt_path=tmp_path / "receipt.json",
    )
    readme = paths.prefix / builder.CPYTHON_SITE_PACKAGES_README_RELATIVE_PATH
    readme.parent.mkdir(parents=True)
    readme.write_bytes(
        b"This directory exists so that 3rd party packages can be installed\n"
        b"here.  Read the source for site.py for more details.\n"
    )
    fsynced: list[int] = []
    monkeypatch.setattr(builder.os, "open", lambda *_args, **_kwargs: 91)
    monkeypatch.setattr(builder.os, "fsync", fsynced.append)
    monkeypatch.setattr(builder.os, "close", lambda descriptor: None)
    receipt = builder._remove_cpython_site_packages_readme(paths)
    assert receipt["sha256"] == builder.CPYTHON_SITE_PACKAGES_README_SHA256
    assert receipt["bytes"] == 119
    assert fsynced == [91]
    assert not readme.exists()

    readme.write_bytes(b"x" * 119)
    with pytest.raises(builder.RuntimeBuildError, match="differs from the pinned"):
        builder._remove_cpython_site_packages_readme(paths)
    assert readme.exists()


def test_ldd_parser_requires_the_exact_soname_row(tmp_path: Path) -> None:
    versioned = tmp_path / "libsqlite3.so.0.8.6"
    versioned.write_bytes(b"sqlite")
    output = f"libsqlite3.so.0 => {versioned} (0x0001)\n"
    assert builder._parse_ldd_resolution(
        output, soname="libsqlite3.so.0"
    ) == versioned.resolve()
    with pytest.raises(builder.RuntimeBuildError, match="exactly one"):
        builder._parse_ldd_resolution(
            f"unrelated => {versioned} (0x0001)\n",
            soname="libsqlite3.so.0",
        )


def test_builder_environment_drops_hostile_python_and_loader_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = builder.BuildPaths(
        repository_root=ROOT,
        work_root=tmp_path / "work",
        prefix=tmp_path / "runtime",
        receipt_path=tmp_path / "receipt.json",
    )
    monkeypatch.setenv("PYTHONPATH", "/hostile/python")
    monkeypatch.setenv("PYTHONHOME", "/hostile/home")
    monkeypatch.setenv("PIP_INDEX_URL", "https://hostile.invalid/simple")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/hostile/lib")
    build_environment = builder._deterministic_environment(paths)
    assert "PYTHONPATH" not in build_environment
    assert "PYTHONHOME" not in build_environment
    assert "PIP_INDEX_URL" not in build_environment
    assert build_environment["PIP_CONFIG_FILE"] == builder.os.devnull
    runtime_environment = builder._self_contained_runtime_environment(
        build_environment
    )
    assert "LD_LIBRARY_PATH" not in runtime_environment
    assert runtime_environment["PYTHONNOUSERSITE"] == "1"
    assert runtime_environment["PYTHONSAFEPATH"] == "1"


def test_trusted_installer_report_is_exactly_lock_and_wheel_bound(
    tmp_path: Path,
) -> None:
    evidence = _valid_build_receipt()["evidence"]
    assert isinstance(evidence, dict)
    wheels = evidence["wheelhouse"]
    inventory = evidence["installed_distribution_inventory"]
    assert isinstance(wheels, tuple)
    assert isinstance(inventory, dict)
    report = {
        "version": "1",
        "pip_version": "24.0",
        "install": [
            {
                "download_info": {
                    "url": "file:///content/wheelhouse/fixture-1.0-py3-none-any.whl",
                    "archive_info": {
                        "hash": f"sha256={'a' * 64}",
                        "hashes": {"sha256": "a" * 64},
                    },
                },
                "metadata": {"name": "fixture", "version": "1.0"},
            }
        ],
    }
    report_path = tmp_path / "pip-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    chain = builder._trusted_installer_chain(
        report_path=report_path,
        wheel_rows=wheels,
        locked_distributions=(("fixture", "1.0"),),
        installed_inventory=inventory,
    )
    assert chain["schema"] == builder.TRUSTED_INSTALLER_CHAIN_SCHEMA
    assert chain["installations"][0]["wheel_sha256"] == "a" * 64

    report["install"][0]["download_info"]["archive_info"]["hashes"][
        "sha256"
    ] = "f" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(builder.RuntimeBuildError, match="selected wheel"):
        builder._trusted_installer_chain(
            report_path=report_path,
            wheel_rows=wheels,
            locked_distributions=(("fixture", "1.0"),),
            installed_inventory=inventory,
        )
