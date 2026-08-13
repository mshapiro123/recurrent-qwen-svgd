"""Autonomous CLI transport and durability wrapper for one P3.4 campaign arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

try:
    import requests
    import zstandard
except ImportError:  # pragma: no cover - exercised only on a fresh Colab image.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "requests", "zstandard"],
        check=True,
    )
    os.execv(sys.executable, [sys.executable, *sys.argv])


KIND = "paper2_phase3_p34_cli_campaign_v1"
PUBLIC_REPO = "https://github.com/mshapiro123/recurrent-qwen-svgd.git"
SOURCE_BRANCH = "codex/phase3-opening-build"
PRIVATE_REPO = "mshapiro123/recurrent-qwen-svgd-runtime-private"
TRANSPORT_TAG = "p34-transport-20260813"
CAMPAIGN_TAG = "p34-campaign-20260813"
ROOT = Path("/content/recurrent-qwen-svgd")
SCRATCH = next(
    (
        path / "recurrent-qwen-svgd-stage"
        for path in (Path("/mnt/local-scratch"), Path("/content/local-scratch"), Path("/content"))
        if path.exists() and shutil.disk_usage(path).free >= 30 * 1024**3
    ),
    None,
)
MIGRATED_SHA = {
    0: "d0f2b735825d29ab9801a5200493ca9aa65294778aea2fb7f728eb8e85dfc519",
    1: "3ca1cdf8dd16bf4f435e81a675d7514778144c5c881af52a70171659f7734b4f",
}
P33_SHA = {
    0: "84dc0fb2d1f69114b20888acd95101d6b31c810974a536dc36358b69fe13c70e",
    1: "e80ad205eb3c4712fdee5303a4887260488f67ff858a2b4b005d724675e52067",
}
EXPECTED_TIER_S_LOOKS = 4
EXPECTED_TIER_W_LOOKS = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class PrivateRelease:
    def __init__(self, *, token_file: Path, tag: str) -> None:
        token = token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise RuntimeError("P3.4 private release token is empty")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        response = self.session.get(
            f"https://api.github.com/repos/{PRIVATE_REPO}/releases", timeout=60
        )
        response.raise_for_status()
        matches = [release for release in response.json() if release["tag_name"] == tag]
        if len(matches) != 1:
            raise RuntimeError(f"P3.4 private release tag resolution failed: {tag}")
        self.release = matches[0]

    def assets(self) -> dict[str, dict[str, Any]]:
        response = self.session.get(
            self.release["assets_url"], params={"per_page": 100}, timeout=60
        )
        response.raise_for_status()
        return {asset["name"]: asset for asset in response.json()}

    def download(self, name: str, destination: Path, *, required: bool = True) -> bool:
        asset = self.assets().get(name)
        if asset is None:
            if required:
                raise FileNotFoundError(f"private release asset missing: {name}")
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".download")
        with self.session.get(
            asset["url"], headers={"Accept": "application/octet-stream"},
            stream=True, timeout=(60, 600),
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for block in response.iter_content(16 * 1024 * 1024):
                    if block:
                        output.write(block)
        temporary.replace(destination)
        return True

    def upload(self, source: Path, name: str) -> None:
        """Publish without deleting the previous durable asset before upload succeeds."""
        existing = self.assets().get(name)
        temporary_name = f"{name}.uploading-{uuid.uuid4().hex}"
        url = self.release["upload_url"].split("{", 1)[0]
        with source.open("rb") as handle:
            response = self.session.post(
                url,
                params={"name": temporary_name},
                headers={"Content-Type": "application/octet-stream"},
                data=handle,
                timeout=(60, 1_800),
            )
        response.raise_for_status()
        uploaded = response.json()
        expected_digest = f"sha256:{sha256_file(source)}"
        if int(uploaded["size"]) != source.stat().st_size:
            raise RuntimeError(f"P3.4 private release upload size mismatch: {name}")
        if uploaded.get("digest") not in (None, expected_digest):
            raise RuntimeError(f"P3.4 private release upload digest mismatch: {name}")
        previous_name = None
        if existing is not None:
            previous_name = f"{name}.previous-{uuid.uuid4().hex}"
            park = self.session.patch(
                existing["url"], json={"name": previous_name}, timeout=60
            )
            park.raise_for_status()
        try:
            rename = self.session.patch(
                uploaded["url"], json={"name": name}, timeout=60
            )
            rename.raise_for_status()
        except Exception:
            if existing is not None and previous_name is not None:
                self.session.patch(existing["url"], json={"name": name}, timeout=60)
            raise
        if existing is not None:
            delete = self.session.delete(existing["url"], timeout=60)
            delete.raise_for_status()


def verify_sha256s(root: Path) -> None:
    ledger = root / "SHA256SUMS"
    for line in ledger.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"P3.4 transport payload SHA mismatch: {relative}")


def stage_transport(*, release: PrivateRelease, cache: Path) -> Path:
    extracted = cache / "p34-private-transport-stage"
    if extracted.is_dir():
        try:
            verify_sha256s(extracted)
            return extracted
        except (FileNotFoundError, RuntimeError):
            shutil.rmtree(extracted)
    parts_dir = cache / "parts"
    manifest_path = parts_dir / "transport_parts_manifest.json"
    release.download("transport_parts_manifest.json", manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts = []
    combined = hashlib.sha256()
    for receipt in manifest["parts"]:
        path = parts_dir / receipt["name"]
        valid_cached_part = (
            path.is_file()
            and path.stat().st_size == int(receipt["bytes"])
            and sha256_file(path) == receipt["sha256"]
        )
        if not valid_cached_part:
            release.download(receipt["name"], path)
        if path.stat().st_size != int(receipt["bytes"]) or sha256_file(path) != receipt["sha256"]:
            raise RuntimeError(f"P3.4 private transport part mismatch: {path.name}")
        parts.append(path)
        with path.open("rb") as handle:
            while block := handle.read(16 * 1024 * 1024):
                combined.update(block)
    if combined.hexdigest() != manifest["concatenated_zstd_sha256"]:
        raise RuntimeError("P3.4 concatenated transport digest mismatch")
    compressed = cache / "p34-private-inputs.tar.zst"
    with compressed.open("wb") as output:
        for part in parts:
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, output, length=16 * 1024 * 1024)
    with compressed.open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as decoded:
            with tarfile.open(fileobj=decoded, mode="r|") as archive:
                archive.extractall(cache, filter="data")
    verify_sha256s(extracted)
    return extracted


def prepare_repo(ref: str) -> None:
    if not ROOT.is_dir():
        subprocess.run(
            ["git", "clone", "--branch", SOURCE_BRANCH, PUBLIC_REPO, str(ROOT)], check=True
        )
    subprocess.run(["git", "fetch", "origin", SOURCE_BRANCH], cwd=ROOT, check=True)
    subprocess.run(["git", "reset", "--hard", ref], cwd=ROOT, check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        cwd=ROOT, check=True,
    )
    subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q",
            "tests/test_paper2_phase3_p34.py",
            "tests/test_paper2_phase3_p34_runner.py",
            "tests/test_paper2_phase3_p34_guardrail_collision.py",
            "tests/test_paper2_phase3_p34_cli.py",
        ],
        cwd=ROOT, check=True,
    )


def assert_training_amendment(*, lock_path: Path, expected_sha256: str) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    amendment = lock.get("guardrail_amendment")
    if not isinstance(amendment, dict) or amendment.get("sha256") != expected_sha256:
        raise RuntimeError("P3.4 ratified guardrail amendment is absent or mismatched")
    guardrails = lock["guardrails"]
    if int(guardrails["tier_s_consecutive_looks"]) != EXPECTED_TIER_S_LOOKS:
        raise RuntimeError("P3.4 Tier-S must require four consecutive looks")
    if int(guardrails["tier_w_consecutive_looks"]) != EXPECTED_TIER_W_LOOKS:
        raise RuntimeError("P3.4 Tier-W must require two consecutive looks")


def package_receipts(*, output_dir: Path, private_dir: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    excluded = {destination.resolve(), temporary.resolve()}
    with temporary.open("wb") as raw:
        with zstandard.ZstdCompressor(level=3, threads=-1).stream_writer(raw) as encoded:
            with tarfile.open(fileobj=encoded, mode="w|") as archive:
                if output_dir.exists():
                    archive.add(output_dir, arcname="outputs")
                if private_dir.exists():
                    for path in private_dir.rglob("*"):
                        if (
                            path.is_file()
                            and path.name != "resume.pt"
                            and path.resolve() not in excluded
                            and not path.name.endswith("-receipts.tar.zst")
                            and not path.name.endswith("-receipts.tar.zst.tmp")
                        ):
                            archive.add(path, arcname=str(Path("private") / path.relative_to(private_dir)))
    temporary.replace(destination)


def run_campaign(args: argparse.Namespace) -> int:
    if SCRATCH is None:
        raise RuntimeError("P3.4 CLI campaign requires at least 30 GiB local scratch")
    label = f"{args.arm}_seed_{args.seed}"
    token_file = args.token_file
    transport_release = PrivateRelease(token_file=token_file, tag=TRANSPORT_TAG)
    campaign_release = PrivateRelease(token_file=token_file, tag=CAMPAIGN_TAG)
    stage = stage_transport(release=transport_release, cache=SCRATCH / "p34-transport")
    prepare_repo(args.ref)
    if args.transport_only:
        receipt = {
            "kind": KIND,
            "status": "transport_verified",
            "ref": args.ref,
            "stage": str(stage),
            "training_steps": 0,
        }
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        return 0
    assert_training_amendment(
        lock_path=ROOT / "training/paper2_phase3_p34_preregistration.json",
        expected_sha256=args.guardrail_amendment_sha256,
    )
    output_dir = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_20260813" / label
    private_dir = SCRATCH / "p34-runs" / label
    private_dir.mkdir(parents=True, exist_ok=True)
    resume_name = f"{label}-resume.pt"
    campaign_release.download(resume_name, private_dir / "resume.pt", required=False)
    command = [
        sys.executable, "-u", "-m", "training.run_paper2_phase3_p34",
        "--seed", str(args.seed), "--arm", args.arm,
        "--old_summary", str(ROOT / "outputs/stage5/stage5_paper2_phase2_stage0a_20260803/summary.json"),
        "--old_private", str(stage / "old"),
        "--new_summary", str(stage / "new/full_cache_summary.json"),
        "--new_private", str(stage / "new"),
        "--staged_labels", str(stage / "preflight/p33_prep/p33_staged_labels.jsonl"),
        "--positive_audit", str(stage / "preflight/p33_prep/p33_audit_slice.jsonl"),
        "--negative_audit", str(stage / "preflight/p33_prep/p33_negative_audit_slice.jsonl"),
        "--retention_panel", str(stage / "preflight/p33_prep/p33_retention_panel.jsonl"),
        "--direction_cache", str(stage / "shared/agreement_oracle_directions.pt"),
        "--dev_panel", str(ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"),
        "--base_scores", str(ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_panel_base_scores.jsonl"),
        "--share_rows", str(ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/share_calibration/p34_share_calibration_rows.jsonl"),
        "--migrated", str(stage / f"seed_{args.seed}/migrated.pt"),
        "--migrated_sha256", MIGRATED_SHA[args.seed],
        "--p33", str(stage / f"seed_{args.seed}/p33_step_1000.pt"),
        "--p33_sha256", P33_SHA[args.seed],
        "--i1", str(stage / f"seed_{args.seed}/i1_resume.pt"),
        "--lock", str(ROOT / "training/paper2_phase3_p34_preregistration.json"),
        "--output_dir", str(output_dir), "--private_dir", str(private_dir),
        "--device", "cuda",
    ]
    log_path = private_dir / "campaign.log"
    status_path = private_dir / "cli_status.json"
    write_json(status_path, {"kind": KIND, "status": "starting", "label": label, "ref": args.ref})
    campaign_release.upload(status_path, f"{label}-status.json")
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        last_resume = None
        upload_error: Exception | None = None
        while process.poll() is None:
            resume = private_dir / "resume.pt"
            signature = (resume.stat().st_mtime_ns, resume.stat().st_size) if resume.is_file() else None
            if signature is not None and signature != last_resume:
                try:
                    with tempfile.TemporaryDirectory(dir=private_dir) as temporary:
                        snapshot = Path(temporary) / "resume.pt"
                        shutil.copy2(resume, snapshot)
                        campaign_release.upload(snapshot, resume_name)
                    package_receipts(
                        output_dir=output_dir, private_dir=private_dir,
                        destination=private_dir / "latest-receipts.tar.zst",
                    )
                    campaign_release.upload(
                        private_dir / "latest-receipts.tar.zst", f"{label}-latest-receipts.tar.zst"
                    )
                    last_resume = signature
                except Exception as error:  # Preserve the latest local checkpoint, then stop safely.
                    upload_error = error
                    process.terminate()
                    break
            time.sleep(10)
        try:
            return_code = process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait(timeout=30)
    if upload_error is not None:
        raise RuntimeError(f"P3.4 durable checkpoint publication failed: {upload_error}")
    final_bundle = private_dir / "final-receipts.tar.zst"
    package_receipts(output_dir=output_dir, private_dir=private_dir, destination=final_bundle)
    campaign_release.upload(final_bundle, f"{label}-final-receipts.tar.zst")
    if (private_dir / "resume.pt").is_file():
        campaign_release.upload(private_dir / "resume.pt", resume_name)
    campaign_summary_path = output_dir / "summary.json"
    campaign_summary = (
        json.loads(campaign_summary_path.read_text(encoding="utf-8"))
        if campaign_summary_path.is_file() else {}
    )
    final_status = {
        "kind": KIND,
        "status": campaign_summary.get("status", "failed"),
        "landed": return_code in (0, 2),
        "label": label, "ref": args.ref, "return_code": return_code,
        "step": campaign_summary.get("step"),
        "stop_reason": campaign_summary.get("stop_reason"),
        "resume_sha256": sha256_file(private_dir / "resume.pt") if (private_dir / "resume.pt").is_file() else None,
        "training_release": CAMPAIGN_TAG,
    }
    write_json(status_path, final_status)
    campaign_release.upload(status_path, f"{label}-status.json")
    print(json.dumps(final_status, indent=2, sort_keys=True), flush=True)
    return 0 if return_code in (0, 2) else return_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--arm", choices=("main", "slot"), required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--token_file", type=Path, required=True)
    parser.add_argument("--transport_only", action="store_true")
    parser.add_argument("--guardrail_amendment_sha256")
    args = parser.parse_args()
    if args.arm == "slot" and args.seed != 0:
        parser.error("slot arm is authorized only for seed zero")
    if not args.transport_only and not args.guardrail_amendment_sha256:
        parser.error("campaign execution requires --guardrail_amendment_sha256")
    return args


def main() -> int:
    try:
        return run_campaign(parse_args())
    except Exception as error:
        print(json.dumps({
            "kind": KIND, "status": "failed", "exception_type": type(error).__name__,
            "exception": str(error), "traceback": traceback.format_exc(),
        }, indent=2, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
