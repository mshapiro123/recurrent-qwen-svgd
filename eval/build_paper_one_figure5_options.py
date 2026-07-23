"""Export copy-ready two- and three-panel Paper One Figure 5 options."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.build_paper_one_figure5 import (
    ARM_ORDER,
    COLORS,
    MARKERS,
    build_figure,
    load_and_validate,
)


DEFAULT_DATA = ROOT / "docs/figures/figure5_data.json"
DEFAULT_OUTPUT_DIR = ROOT / "docs/figures"


def _style_axis(axis: Any) -> None:
    from matplotlib.ticker import FuncFormatter

    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    axis.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8.5)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))


def build_three_panel(payload: dict[str, Any], output_base: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter, LogLocator
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Figure export requires matplotlib") from exc

    depths = [int(value) for value in payload["depths"]]
    total = int(payload["rows_per_depth"])
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.2), dpi=200)
    depth_ax, cost_ax, pareto_ax = axes
    fig.patch.set_facecolor("white")

    for axis in (depth_ax, cost_ax):
        axis.axvspan(0.7, 8.3, color="#E8F2ED", alpha=0.75, zorder=0)

    for arm in ARM_ORDER:
        item = payload["arms"][arm]
        accuracy = [100.0 * int(item["correct"][str(depth)]) / total for depth in depths]
        latency = [float(item["wall_clock_ms"][str(depth)]) for depth in depths]
        style = {
            "color": COLORS[arm],
            "marker": MARKERS[arm],
            "linewidth": 2.2,
            "markersize": 5.8,
            "markeredgecolor": "white",
            "markeredgewidth": 0.7,
            "label": item["label"],
        }
        depth_ax.plot(depths, accuracy, **style)
        cost_ax.plot(depths, latency, **style)
        pareto_ax.plot(latency, accuracy, **style)

        for index, (x_value, y_value) in enumerate(zip(latency, accuracy)):
            if depths[index] not in {1, 8, 11, 14}:
                continue
            x_offset = -6 if arm == "C" else 6
            alignment = "right" if arm == "C" else "left"
            y_offset = 5 if depths[index] in {1, 11} else -10
            pareto_ax.annotate(
                str(depths[index]),
                (x_value, y_value),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=alignment,
                fontsize=7,
                color=COLORS[arm],
            )

    depth_ax.axvline(10.5, color="#555555", linestyle=(0, (3, 3)), linewidth=1)
    depth_ax.text(10.35, 42, "scratchpad\nhorizon", ha="right", va="center", fontsize=8, color="#444444")

    depth_ax.set_title("A. Accuracy by depth", loc="left", fontweight="bold", fontsize=11)
    depth_ax.set_xlabel("Composition depth")
    depth_ax.set_ylabel("Accuracy")
    depth_ax.set_xticks(depths)
    depth_ax.set_xlim(0.7, 14.3)
    depth_ax.set_ylim(-1.5, 103)

    cost_ax.set_title("B. Latency by depth", loc="left", fontweight="bold", fontsize=11)
    cost_ax.set_xlabel("Composition depth")
    cost_ax.set_ylabel("Median model-call latency (ms, log scale)")
    cost_ax.set_xticks(depths)
    cost_ax.set_xlim(0.7, 14.3)
    cost_ax.set_yscale("log")
    cost_ax.set_ylim(22, 2900)
    cost_ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    cost_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

    pareto_ax.set_title("C. Accuracy by latency", loc="left", fontweight="bold", fontsize=11)
    pareto_ax.set_xlabel("Median model-call latency (ms, log scale)")
    pareto_ax.set_ylabel("Accuracy")
    pareto_ax.set_xscale("log")
    pareto_ax.set_xlim(22, 2900)
    pareto_ax.set_ylim(-1.5, 103)
    pareto_ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    pareto_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

    _style_axis(depth_ax)
    cost_ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    cost_ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    cost_ax.spines["top"].set_visible(False)
    cost_ax.spines["right"].set_visible(False)
    cost_ax.tick_params(labelsize=8.5)
    _style_axis(pareto_ax)

    handles, labels = depth_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=8.5,
    )
    fig.suptitle(
        "Accuracy, depth, and wall-clock cost on the frozen Phase A family",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.945,
        "First completed response; 128 identical rows per depth; one A100; batch size 1",
        ha="left",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0.035, 0.085, 0.995, 0.91), w_pad=2.1)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_base.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png")]
    for path in outputs:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    payload = load_and_validate(args.data)

    outputs = build_three_panel(
        payload,
        args.output_dir / "figure5_option_a_three_panel",
    )
    outputs.extend(
        build_figure(
            payload,
            args.output_dir / "figure5_option_b_two_panel",
        )
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
