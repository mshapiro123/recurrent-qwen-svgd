"""Launch or execute the forward-only WEFT-1 hermetic P-B DECON screen."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.weft1_corpus_decon import (  # noqa: E402
    DECON_PARENT_WATCHDOG_SECONDS,
    launch_hermetic_decon,
    run_hermetic_decon,
)


def _add_private_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confirm-seal", required=True, action="append", type=Path
    )
    parser.add_argument("--confirm-seal-ledger", required=True, type=Path)
    parser.add_argument("--confirm-private-rows", required=True, type=Path)
    parser.add_argument("--eval-e-index", required=True, type=Path)
    parser.add_argument("--eval-e-lock", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    child = commands.add_parser("child")
    child.add_argument("--materialization-root", required=True, type=Path)
    _add_private_inputs(child)
    child.add_argument("--output-root", required=True, type=Path)

    launch = commands.add_parser("launch")
    launch.add_argument("--materialization-root", required=True, type=Path)
    _add_private_inputs(launch)
    launch.add_argument("--output-root", required=True, type=Path)
    launch.add_argument("--local-work-parent", required=True, type=Path)
    launch.add_argument("--python-executable", type=Path)
    launch.add_argument("--unshare-executable", type=Path)
    launch.add_argument(
        "--timeout-seconds",
        type=int,
        default=DECON_PARENT_WATCHDOG_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    common = {
        "materialization_root": arguments.materialization_root,
        "confirm_seal_paths": tuple(arguments.confirm_seal),
        "confirm_seal_ledger_path": arguments.confirm_seal_ledger,
        "confirm_private_rows_path": arguments.confirm_private_rows,
        "eval_e_index_path": arguments.eval_e_index,
        "eval_e_lock_path": arguments.eval_e_lock,
        "output_root": arguments.output_root,
    }
    if arguments.command == "child":
        run_hermetic_decon(**common)
    elif arguments.command == "launch":
        launch_hermetic_decon(
            **common,
            local_work_parent=arguments.local_work_parent,
            python_executable=arguments.python_executable,
            unshare_executable=arguments.unshare_executable,
            timeout_seconds=arguments.timeout_seconds,
        )
    else:  # pragma: no cover - argparse owns this branch
        raise AssertionError("unknown DECON command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
