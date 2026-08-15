"""Run the authorized score-only P3.4 fixed-ceiling DEV probe."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage5_paper2_phase3_p34_fixed_ceiling_probe_20260815"
SOURCE_RUN_ID = "stage5_paper2_phase3_p34_a2_20260814"
DRIVE_STAGE5 = Path("/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5")
MIGRATION_ID = "stage5_paper2_phase3_p31_p32_receipts_20260810"
P33_ID = "stage5_paper2_phase3_p33_20260811"
I1_ID = "stage5_paper2_phase3_p33_i1_20260812"
MIGRATED_SHA = {
    0: "d0f2b735825d29ab9801a5200493ca9aa65294778aea2fb7f728eb8e85dfc519",
    1: "3ca1cdf8dd16bf4f435e81a675d7514778144c5c881af52a70171659f7734b4f",
}
P33_SHA = {
    0: "84dc0fb2d1f69114b20888acd95101d6b31c810974a536dc36358b69fe13c70e",
    1: "e80ad205eb3c4712fdee5303a4887260488f67ff858a2b4b005d724675e52067",
}
P34_SHA = {
    0: "381955ec5b78d0a00883c29e9f940feac8cfc8665f7a3a4446c79734532f4ed7",
    1: "97ad532a5bffd72b2563799047b517e531e00115793bf4808f060148dfffc1ec",
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    drive_run = DRIVE_STAGE5 / RUN_ID
    private = drive_run / "private/fixed_ceiling_probe"
    receipts = drive_run / "receipts"
    local = ROOT / "outputs/stage5" / RUN_ID
    status_path = receipts / "status.json"

    def status(value: str, **details: object) -> None:
        write_json(status_path, {
            "kind": "paper2_phase3_p34_fixed_ceiling_probe_status_v1",
            "status": value,
            "updated_at_unix": time.time(),
            **details,
        })
        print(f"p34_fixed_ceiling_status={value} details={details}", flush=True)

    try:
        status("staging_inputs")
        lock = json.loads(
            (ROOT / "training/paper2_phase3_p34_preregistration.json").read_text(encoding="utf-8")
        )
        panel = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_task_panel.jsonl"
        base_scores = ROOT / "outputs/stage5/stage5_paper2_phase3_p34_lock_20260812/panel/p34_panel_base_scores.jsonl"
        if not panel.is_file() or not base_scores.is_file():
            raise FileNotFoundError("P3.4 frozen DEV panel or base-score receipt is missing")

        conditions = []
        for seed in (0, 1):
            migrated = DRIVE_STAGE5 / MIGRATION_ID / f"private/migrated_checkpoints/seed_{seed}_full_a2_phase3_migrated.pt"
            p33 = DRIVE_STAGE5 / P33_ID / f"private/seed_{seed}/checkpoint_step_1000.pt"
            i1 = DRIVE_STAGE5 / I1_ID / f"private/seed_{seed}/resume.pt"
            p34 = DRIVE_STAGE5 / SOURCE_RUN_ID / f"private/main_seed_{seed}/checkpoint_step_4000.pt"
            expected_i1 = str(lock["initialization"][f"seed_{seed}"]["sha256"])
            for path, expected in (
                (migrated, MIGRATED_SHA[seed]),
                (p33, P33_SHA[seed]),
                (i1, expected_i1),
                (p34, P34_SHA[seed]),
            ):
                if not path.is_file():
                    raise FileNotFoundError(path)
                observed = sha256_file(path)
                if observed != expected:
                    raise RuntimeError(f"fixed-ceiling source SHA mismatch path={path} observed={observed}")
            for ceiling, suffix in ((0.02, "0p02"), (0.08, "0p08")):
                name = f"seed_{seed}_ceiling_{suffix}"
                row_path = private / f"{name}.jsonl"
                summary_path = private / f"{name}_summary.json"
                conditions.append((seed, ceiling, name, row_path, summary_path))
                if summary_path.is_file():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if (
                        summary.get("status") == "complete_dev_only"
                        and int(summary.get("rows", 0)) == 1_024
                        and float(summary.get("evaluation_gate_ceiling", -1)) == ceiling
                    ):
                        print(f"p34_fixed_ceiling_resume condition={name} status=already_complete", flush=True)
                        continue
                status("scoring", condition=name)
                run([
                    sys.executable, "-u", "-m", "eval.eval_paper2_phase3_p34_task_trajectory",
                    "--panel", str(panel),
                    "--base_scores", str(base_scores),
                    "--output_jsonl", str(row_path),
                    "--summary", str(summary_path),
                    "--condition", name,
                    "--look", "20",
                    "--seed", str(seed),
                    "--migrated", str(migrated),
                    "--migrated_sha256", MIGRATED_SHA[seed],
                    "--p33", str(p33),
                    "--p33_sha256", P33_SHA[seed],
                    "--i1", str(i1),
                    "--i1_sha256", expected_i1,
                    "--p34", str(p34),
                    "--p34_sha256", P34_SHA[seed],
                    "--gate_ceiling_override", str(ceiling),
                ])

        status("analyzing")
        run([
            sys.executable, "-u", "-m", "analysis.build_paper2_phase3_p34_fixed_ceiling_probe",
            "--input_dir", str(private),
            "--output_summary", str(local / "summary.json"),
            "--output_manifest", str(local / "artifact_manifest.json"),
        ])
        for source in (local / "summary.json", local / "artifact_manifest.json"):
            target = receipts / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            if sha256_file(target) != sha256_file(source):
                raise RuntimeError(f"Drive publication SHA mismatch for {source.name}")
        status(
            "complete",
            summary_sha256=sha256_file(receipts / "summary.json"),
            conditions=[name for _seed, _ceiling, name, _rows, _summary in conditions],
            optimizer_steps=0,
            confirm_scored=False,
            eval_e_scored=False,
        )
        print("P3.4 fixed-ceiling score probe landed; release this GPU session.", flush=True)
        return 0
    except Exception as error:
        status(
            "failed",
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
