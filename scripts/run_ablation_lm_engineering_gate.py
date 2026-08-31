"""Run every repository test and enforce the exact ablation-engineering quarantine.

This is not an xfail or deselection list. Every test executes. The engineering
gate passes only while pytest reports exactly the named legacy failures: a new,
resolved, or renamed failure makes this runner red and requires a new receipt.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "training"
    / "ablation_lm_engineering_quarantine_20260831_gtok_s2.json"
)
FAILED_NODE = re.compile(r"^FAILED\s+(\S+)", re.MULTILINE)


SUMMARY_COUNT = re.compile(
    r"(?P<count>\d+)\s+(?P<label>failed|passed|skipped|deselected|xfailed|xpassed|errors?)"
)


def _load_gate_contract(
    receipt_path: Path = RECEIPT,
    *,
    today: date | None = None,
) -> tuple[set[str], int]:
    """Load an active, unexpired exact-node quarantine contract."""

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "exact_node_quarantine_build_only":
        raise RuntimeError("ablation engineering quarantine is not active")
    if receipt.get("scope") != "ablation_lm_engineering_gate_only":
        raise RuntimeError("ablation engineering quarantine scope differs")
    if receipt.get("training_authorized") is not False:
        raise RuntimeError("engineering quarantine may not authorize training")
    review_due_raw = receipt.get("review_due_on")
    if type(review_due_raw) is not str:
        raise RuntimeError("engineering quarantine requires an ISO review_due_on date")
    try:
        review_due = date.fromisoformat(review_due_raw)
    except ValueError as error:
        raise RuntimeError(
            "engineering quarantine review_due_on must be an ISO YYYY-MM-DD date"
        ) from error
    if review_due.isoformat() != review_due_raw:
        raise RuntimeError(
            "engineering quarantine review_due_on must use canonical ISO YYYY-MM-DD"
        )
    review_day = date.today() if today is None else today
    if type(review_day) is not date:
        raise TypeError("today must be an exact datetime.date")
    if review_day >= review_due:
        raise RuntimeError(
            "engineering quarantine review is stale as of "
            f"{review_due.isoformat()}"
        )
    expected = [row["node_id"] for row in receipt["expected_failures"]]
    if not expected or len(expected) != len(set(expected)):
        raise RuntimeError("expected failure node IDs must be nonempty and unique")
    expected_passed = receipt["last_observed_full_suite"]["passed"]
    if type(expected_passed) is not int or expected_passed < 1:
        raise RuntimeError("receipt must lock a positive full-suite pass count")
    return {node.replace("\\", "/") for node in expected}, expected_passed


def _summary_line(output: str) -> str:
    candidates = [
        line.strip("= \r")
        for line in output.splitlines()
        if " passed" in line and (" failed" in line or " warnings" in line)
    ]
    return candidates[-1] if candidates else "pytest summary unavailable"


def main() -> int:
    try:
        expected, expected_passed = _load_gate_contract()
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"ablation engineering gate receipt error: {error}", file=sys.stderr)
        return 2

    clean_environment = os.environ.copy()
    clean_environment.pop("PYTEST_ADDOPTS", None)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--color=no", "-o", "addopts="],
        cwd=ROOT,
        env=clean_environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = completed.stdout + completed.stderr
    observed = {node.replace("\\", "/") for node in FAILED_NODE.findall(output)}
    summary = _summary_line(output)
    counts = {
        match.group("label"): int(match.group("count"))
        for match in SUMMARY_COUNT.finditer(summary)
    }
    added = tuple(sorted(observed - expected))
    resolved_or_renamed = tuple(sorted(expected - observed))
    forbidden_outcomes = {
        label: counts.get(label, 0)
        for label in ("skipped", "deselected", "xfailed", "xpassed", "error", "errors")
        if counts.get(label, 0)
    }
    exact = (
        completed.returncode == 1
        and not added
        and not resolved_or_renamed
        and counts.get("failed") == len(expected)
        and counts.get("passed") == expected_passed
        and not forbidden_outcomes
    )

    if exact:
        print(
            "ablation engineering gate PASS: all tests ran and exactly "
            f"{len(expected)} quarantined legacy nodes failed"
        )
        print(f"full repository suite remains RED: {summary}")
        return 0

    print("ablation engineering gate FAIL", file=sys.stderr)
    print(f"pytest exit code: {completed.returncode}", file=sys.stderr)
    print(f"added failures: {added!r}", file=sys.stderr)
    print(f"resolved or renamed failures: {resolved_or_renamed!r}", file=sys.stderr)
    print(f"expected passed: {expected_passed}; observed counts: {counts!r}", file=sys.stderr)
    print(f"forbidden outcomes: {forbidden_outcomes!r}", file=sys.stderr)
    print("pytest output tail:", file=sys.stderr)
    print("\n".join(output.splitlines()[-120:]), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
