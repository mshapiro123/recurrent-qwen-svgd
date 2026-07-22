"""Build Paper One Figure 5 from locked accuracy and latency receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "docs/figures/figure5_data.json"
DEFAULT_OUTPUT_BASE = ROOT / "docs/figures/figure5_accuracy_depth_wall_clock"
ARM_ORDER = ("A", "E", "C", "B", "D")
COLORS = {
    "A": "#0B6E4F",
    "E": "#D48C00",
    "C": "#2667FF",
    "B": "#C44536",
    "D": "#6F4E7C",
}
MARKERS = {"A": "o", "E": "s", "C": "^", "B": "D", "D": "P"}
LABEL_OFFSETS = {
    "A": (4, 4, "left"),
    "E": (4, 4, "left"),
    "C": (5, 3, "left"),
    "B": (-7, 5, "right"),
    "D": (7, -8, "left"),
}
LABEL_OFFSET_OVERRIDES = {
    ("B", 8): (-7, 14, "right"),
    ("B", 11): (-7, 0, "right"),
    ("B", 14): (-7, -14, "right"),
    ("D", 1): (7, -15, "left"),
    ("D", 2): (7, -2, "left"),
    ("D", 4): (7, 8, "left"),
    ("D", 11): (7, -16, "left"),
    ("D", 14): (7, 14, "left"),
}


def load_and_validate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    depths = [int(value) for value in payload["depths"]]
    if depths != [1, 2, 4, 8, 11, 14]:
        raise ValueError(f"Unexpected Figure 5 depths: {depths}")
    if bool(payload.get("placeholder_data_present")):
        raise ValueError("Figure 5 still contains placeholder data")
    total = int(payload["rows_per_depth"])
    if total != 128:
        raise ValueError(f"Expected 128 rows per depth, observed {total}")
    if tuple(payload["arms"]) != ARM_ORDER:
        raise ValueError(f"Expected arm order {ARM_ORDER}, observed {tuple(payload['arms'])}")
    for arm in ARM_ORDER:
        values = payload["arms"][arm]
        if bool(values.get("placeholder", True)):
            raise ValueError(f"Arm {arm} is still marked as placeholder")
        for field in ("correct", "wall_clock_ms"):
            missing = set(map(str, depths)) - set(values[field])
            if missing:
                raise ValueError(f"Arm {arm} {field} is missing depths {sorted(missing)}")
        for depth in depths:
            correct = int(values["correct"][str(depth)])
            latency = float(values["wall_clock_ms"][str(depth)])
            if not 0 <= correct <= total:
                raise ValueError(f"Arm {arm} depth {depth} has invalid correct count {correct}")
            if latency <= 0:
                raise ValueError(f"Arm {arm} depth {depth} has invalid latency {latency}")
    return payload


def build_figure(payload: dict[str, Any], output_base: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter, LogLocator
    except ImportError as exc:  # pragma: no cover - environment-specific guidance
        raise RuntimeError(
            "Figure 5 requires matplotlib. Install docs/requirements-figures.txt."
        ) from exc

    depths = [int(value) for value in payload["depths"]]
    total = int(payload["rows_per_depth"])
    fig, (depth_ax, latency_ax) = plt.subplots(1, 2, figsize=(12.8, 5.35), dpi=180)
    fig.patch.set_facecolor("white")

    depth_ax.axvspan(0.7, 8.3, color="#E8F2ED", alpha=0.75, zorder=0)
    depth_ax.text(
        4.5,
        4.0,
        "trained support\n(depths 1-8)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#315E4D",
    )

    for arm in ARM_ORDER:
        item = payload["arms"][arm]
        accuracy = [100.0 * int(item["correct"][str(depth)]) / total for depth in depths]
        latency = [float(item["wall_clock_ms"][str(depth)]) for depth in depths]
        label = item["label"]
        style = dict(
            color=COLORS[arm],
            marker=MARKERS[arm],
            linewidth=2.3,
            markersize=6.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=label,
        )
        depth_ax.plot(depths, accuracy, **style)
        latency_ax.plot(latency, accuracy, **style)
        for depth, x_value, y_value in zip(depths, latency, accuracy):
            x_offset, y_offset, alignment = LABEL_OFFSET_OVERRIDES.get(
                (arm, depth), LABEL_OFFSETS[arm]
            )
            latency_ax.annotate(
                str(depth),
                (x_value, y_value),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=alignment,
                fontsize=7,
                color=COLORS[arm],
            )

    depth_ax.set_title("A. Accuracy by composition depth", loc="left", fontweight="bold")
    depth_ax.set_xlabel("Composition depth")
    depth_ax.set_ylabel("Accuracy")
    depth_ax.set_xticks(depths)
    depth_ax.set_xlim(0.7, 14.3)

    latency_ax.set_title("B. Accuracy by registered-path latency", loc="left", fontweight="bold")
    latency_ax.set_xlabel("Median model-call latency per row (ms, log scale)")
    latency_ax.set_xscale("log")
    latency_ax.set_xlim(22, 2900)
    latency_ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    latency_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

    for axis in (depth_ax, latency_ax):
        axis.set_ylim(-1.5, 103)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        axis.grid(axis="x", color="#EEEEEE", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles, labels = depth_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        fontsize=9,
    )
    fig.suptitle(
        "Accuracy, recurrent depth, and wall-clock cost on the frozen Phase A family",
        x=0.07,
        y=0.995,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.94,
        "Same 128 rows per depth; complete synchronized model-call latency on one A100, batch size 1",
        ha="left",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0.04, 0.08, 0.99, 0.91), w_pad=2.5)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_base.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png")]
    for path in outputs:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    args = parser.parse_args()
    payload = load_and_validate(args.data)
    for path in build_figure(payload, args.output_base):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
