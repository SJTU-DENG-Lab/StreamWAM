#!/usr/bin/env python3
"""Generate the static Stream-WAM latency figure used by the project page."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
import numpy as np


PAPER = "#f7f5ef"
PANEL = "#fffdf8"
INK = "#25231e"
MUTED = "#706b62"
GRID = "#ded9cf"
BASELINES = ("#b9aa9c", "#a87b5c", "#c39370")
STREAM = "#168b7c"
STREAM_EDGE = "#0b6159"
ABLATION = "#d7ddd8"
ABLATION_EDGE = "#7f8d87"

LIBERO_METHODS = (
    "FastWAM",
    "Joint-CD",
    "RTC",
    "Stream-WAM",
    "w/o Action\nConditioning",
    "w/o Slot",
)
LIBERO_CHUNK = (493.0, 114.2, 142.3, 41.0, 35.1, 36.3)
LIBERO_LONG = (16.31, 6.89, 6.23, 5.36, 5.20, 5.31)
LIBERO_SHORT = (8.25, 3.74, 3.20, 3.15, 2.92, 3.01)

ROBOTWIN_METHODS = ("StarWAM-Joint", "StarWAM-CD", "Stream-WAM")
ROBOTWIN_CHUNK = (190.17, 81.21, 47.09)
ROBOTWIN_EPISODE = (110.22, 102.59, 77.48)
ROBOCASA_METHODS = ("X-WAM", "X-WAM-CD", "Stream-WAM")
ROBOCASA_CHUNK = (504.00, 135.21, 136.76)
ROBOCASA_EPISODE = (37.31, 33.60, 11.76)
LIBERO_CHUNK_YMAX = 520


def _style_axis(axis: Axes, *, ylabel: str) -> None:
    axis.set_facecolor(PANEL)
    axis.set_ylabel(ylabel, color=MUTED, fontsize=9, labelpad=8)
    axis.tick_params(axis="both", colors=MUTED, labelsize=8)
    axis.grid(axis="y", color=GRID, linewidth=0.75, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)


def _panel_heading(axis: Axes, benchmark: str, metric: str) -> None:
    title = f"{benchmark}  ·  {metric}" if benchmark else metric
    axis.set_title(
        title,
        loc="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
        pad=13,
    )
    axis.text(
        1,
        1.04,
        "LOWER IS BETTER",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=6.5,
        fontweight="bold",
        alpha=0.9,
    )


def _method_styles(count: int) -> tuple[list[str], list[str | None], list[str]]:
    colors = [BASELINES[index % len(BASELINES)] for index in range(count)]
    edges = ["none"] * count
    hatches: list[str | None] = [None] * count
    colors[3] = STREAM
    edges[3] = STREAM_EDGE
    for index in range(4, count):
        colors[index] = ABLATION
        edges[index] = ABLATION_EDGE
        hatches[index] = "////"
    return colors, hatches, edges


def _annotate(
    axis: Axes,
    bars,
    values: tuple[float, ...],
    *,
    precision: int,
    offset: float,
) -> None:
    for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.{precision}f}",
            ha="center",
            va="bottom",
            color=STREAM_EDGE if index == 3 else INK,
            fontsize=7.2,
            fontweight="bold" if index == 3 else "normal",
        )


def _mark_ablation_region(axis: Axes, *, top: float) -> None:
    axis.axvspan(3.55, 5.45, color="#edf0ed", alpha=0.85, zorder=0)
    axis.text(
        4.5,
        top,
        "ABLATIONS",
        ha="center",
        va="top",
        color=ABLATION_EDGE,
        fontsize=6.2,
        fontweight="bold",
    )


def _draw_libero_chunk(axis: Axes) -> None:
    _style_axis(axis, ylabel="Milliseconds")
    _panel_heading(axis, "LIBERO", "Chunk Time")
    positions = np.arange(len(LIBERO_METHODS))
    colors, hatches, edges = _method_styles(len(LIBERO_METHODS))
    _mark_ablation_region(axis, top=LIBERO_CHUNK_YMAX - 10)

    bars = axis.bar(
        positions,
        LIBERO_CHUNK,
        width=0.62,
        color=colors,
        edgecolor=edges,
        linewidth=1.2,
        zorder=3,
    )
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)

    _annotate(axis, bars, LIBERO_CHUNK, precision=1, offset=7)
    axis.set_ylim(0, LIBERO_CHUNK_YMAX)
    axis.set_xticks(positions, LIBERO_METHODS, rotation=19, ha="right")
    axis.tick_params(axis="x", colors=MUTED, labelsize=7.4, length=0, pad=7)

def _draw_libero_episode(axis: Axes) -> None:
    _style_axis(axis, ylabel="Seconds")
    _panel_heading(axis, "LIBERO", "Episode Time")
    positions = np.arange(len(LIBERO_METHODS))
    width = 0.31
    colors, hatches, edges = _method_styles(len(LIBERO_METHODS))
    _mark_ablation_region(axis, top=17.8)

    long_bars = axis.bar(positions - width / 2, LIBERO_LONG, width, color=colors, edgecolor=edges, linewidth=1.1, zorder=3)
    short_bars = axis.bar(positions + width / 2, LIBERO_SHORT, width, color=colors, edgecolor=edges, linewidth=1.1, alpha=0.48, zorder=3)
    for index, hatch in enumerate(hatches):
        long_bars[index].set_hatch(hatch)
        short_bars[index].set_hatch(hatch)

    for bars, values in ((long_bars, LIBERO_LONG), (short_bars, LIBERO_SHORT)):
        _annotate(axis, bars, values, precision=2, offset=0.28)

    axis.set_ylim(0, 18.4)
    axis.set_xticks(positions, LIBERO_METHODS, rotation=19, ha="right")
    axis.tick_params(axis="x", length=0, pad=7, labelsize=7.4)
    axis.legend(
        handles=(
            Patch(facecolor="#9b7358", label="Long"),
            Patch(facecolor="#9b7358", alpha=0.48, label="Short"),
        ),
        loc="upper right",
        bbox_to_anchor=(1, 0.94),
        frameon=False,
        fontsize=7.5,
        ncols=2,
        handlelength=1.2,
    )


def _draw_robotwin_and_robocasa(
    axis: Axes,
    robotwin_values: tuple[float, ...],
    robocasa_values: tuple[float, ...],
    *,
    metric: str,
    ylabel: str,
    ceiling: float,
) -> None:
    _style_axis(axis, ylabel=ylabel)
    _panel_heading(axis, "", metric)
    positions = np.array((0, 1, 2, 4, 5, 6))
    methods = ROBOTWIN_METHODS + ROBOCASA_METHODS
    values = robotwin_values + robocasa_values
    colors = [BASELINES[0], BASELINES[1], STREAM, BASELINES[0], BASELINES[1], STREAM]
    edges = ["none", "none", STREAM_EDGE, "none", "none", STREAM_EDGE]

    axis.axvspan(-0.6, 2.6, color="#f4efe8", alpha=0.7, zorder=0)
    axis.axvspan(3.4, 6.6, color="#edf1ee", alpha=0.75, zorder=0)
    axis.axvline(3, color=GRID, linewidth=1, zorder=1)
    bars = axis.bar(positions, values, width=0.58, color=colors, edgecolor=edges, linewidth=1.2, zorder=3)

    for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
        is_streamwam = index in (2, 5)
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + ceiling * 0.022,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            color=STREAM_EDGE if is_streamwam else INK,
            fontsize=7.4,
            fontweight="bold" if is_streamwam else "normal",
        )

    axis.text(1, ceiling * 0.96, "ROBOTWIN 2.0", ha="center", va="top", color=MUTED, fontsize=6.5, fontweight="bold")
    axis.text(5, ceiling * 0.96, "ROBOCASA", ha="center", va="top", color=MUTED, fontsize=6.5, fontweight="bold")
    axis.set_ylim(0, ceiling)
    axis.set_xticks(positions, methods, rotation=16, ha="right")
    axis.tick_params(axis="x", length=0, pad=7, labelsize=7.2)

def render(output_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
    })
    figure = plt.figure(figsize=(12, 9), dpi=200, facecolor=PAPER)
    grid = figure.add_gridspec(2, 2, left=0.075, right=0.97, top=0.84, bottom=0.105, hspace=0.42, wspace=0.22)

    _draw_libero_chunk(figure.add_subplot(grid[0, 0]))
    _draw_libero_episode(figure.add_subplot(grid[0, 1]))
    _draw_robotwin_and_robocasa(
        figure.add_subplot(grid[1, 0]),
        ROBOTWIN_CHUNK,
        ROBOCASA_CHUNK,
        metric="Chunk Time",
        ylabel="Milliseconds",
        ceiling=550,
    )
    _draw_robotwin_and_robocasa(
        figure.add_subplot(grid[1, 1]),
        ROBOTWIN_EPISODE,
        ROBOCASA_EPISODE,
        metric="Episode Time",
        ylabel="Seconds",
        ceiling=125,
    )

    figure.suptitle("Streaming efficiency", x=0.075, y=0.965, ha="left", color=INK, fontsize=21, fontweight="bold")
    figure.text(
        0.075,
        0.908,
        "Vertical-bar comparison across LIBERO, RoboTwin 2.0, and RoboCasa  ·  exact values shown above each bar",
        ha="left",
        color=MUTED,
        fontsize=9.5,
    )
    figure.text(
        0.97,
        0.045,
        "Stream-WAM highlighted in teal  ·  hatched bars denote LIBERO ablations",
        ha="right",
        color=MUTED,
        fontsize=7.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, facecolor=PAPER)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("assets") / "stream-wam-latency.png",
    )
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
