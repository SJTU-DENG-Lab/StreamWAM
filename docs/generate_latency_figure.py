#!/usr/bin/env python3
"""Generate the two static Stream-WAM latency figures used by the project page."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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
    "Stream-WAM\nw/o Action\nConditioning",
    "Stream-WAM\nw/o Slot\nEncoder",
)
LIBERO_CHUNK = (493.0, 114.2, 142.3, 41.0, 35.1, 36.3)
LIBERO_LONG = (16.31, 6.89, 6.23, 5.36, 5.20, 5.31)
LIBERO_SHORT = (8.25, 3.74, 3.20, 3.15, 2.92, 3.01)

ROBOTWIN_METHODS = ("StarWAM\nJoint", "StarWAM\nCD", "Stream-WAM")
ROBOTWIN_CHUNK = (190.17, 81.21, 47.09)
ROBOTWIN_EPISODE = (110.22, 102.59, 77.48)
ROBOCASA_METHODS = ("X-WAM", "X-WAM\nCD", "Stream-WAM")
ROBOCASA_CHUNK = (374.07, 134.37, 115.98)
ROBOCASA_EPISODE = (17.36, 13.04, 9.49)
LIBERO_CHUNK_YMAX = 520
ROBOCASA_CHUNK_YMAX = 410
ROBOCASA_EPISODE_YMAX = 20


def _style_axis(axis: Axes, *, ylabel: str, title: str) -> None:
    axis.set_facecolor(PANEL)
    axis.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=14)
    axis.set_ylabel(ylabel, color=MUTED, fontsize=8.5, labelpad=7)
    axis.tick_params(axis="both", colors=MUTED, labelsize=7.5)
    axis.tick_params(axis="x", length=0, pad=8)
    axis.grid(axis="y", color=GRID, linewidth=0.75, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)


def _libero_styles() -> tuple[list[str], list[str], list[str | None]]:
    colors = [BASELINES[0], BASELINES[1], BASELINES[2], STREAM, ABLATION, ABLATION]
    edges = ["none", "none", "none", STREAM_EDGE, ABLATION_EDGE, ABLATION_EDGE]
    hatches: list[str | None] = [None, None, None, None, "////", "\\\\\\\\"]
    return colors, edges, hatches


def _apply_hatches(bars, hatches: list[str | None]) -> None:
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)


def _annotate(
    axis: Axes,
    bars,
    values: tuple[float, ...],
    *,
    ceiling: float,
    precision: int,
    stream_indices: tuple[int, ...],
) -> None:
    for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
        is_stream = index in stream_indices
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ceiling * 0.021,
            f"{value:.{precision}f}",
            ha="center",
            va="bottom",
            color=STREAM_EDGE if is_stream else INK,
            fontsize=7,
            fontweight="bold" if is_stream else "normal",
        )


def _draw_libero_chunk(axis: Axes) -> None:
    _style_axis(axis, ylabel="Milliseconds", title="LIBERO")
    positions = np.arange(len(LIBERO_METHODS))
    colors, edges, hatches = _libero_styles()
    axis.axvspan(3.55, 5.45, color="#edf0ed", alpha=0.9, zorder=0)
    bars = axis.bar(
        positions,
        LIBERO_CHUNK,
        width=0.62,
        color=colors,
        edgecolor=edges,
        linewidth=1.15,
        zorder=3,
    )
    _apply_hatches(bars, hatches)
    _annotate(axis, bars, LIBERO_CHUNK, ceiling=LIBERO_CHUNK_YMAX, precision=1, stream_indices=(3,))
    axis.set_ylim(0, LIBERO_CHUNK_YMAX)
    axis.set_xticks(positions, LIBERO_METHODS)


def _draw_libero_episode(axis: Axes) -> None:
    _style_axis(axis, ylabel="Seconds", title="LIBERO")
    positions = np.arange(len(LIBERO_METHODS))
    width = 0.3
    ceiling = 18.4
    colors, edges, hatches = _libero_styles()
    axis.axvspan(3.55, 5.45, color="#edf0ed", alpha=0.9, zorder=0)
    long_bars = axis.bar(
        positions - width / 2,
        LIBERO_LONG,
        width,
        color=colors,
        edgecolor=edges,
        linewidth=1.05,
        zorder=3,
    )
    short_bars = axis.bar(
        positions + width / 2,
        LIBERO_SHORT,
        width,
        color=colors,
        edgecolor=edges,
        linewidth=1.05,
        alpha=0.48,
        zorder=3,
    )
    _apply_hatches(long_bars, hatches)
    _apply_hatches(short_bars, hatches)
    _annotate(axis, long_bars, LIBERO_LONG, ceiling=ceiling, precision=2, stream_indices=(3,))
    _annotate(axis, short_bars, LIBERO_SHORT, ceiling=ceiling, precision=2, stream_indices=(3,))
    axis.set_ylim(0, ceiling)
    axis.set_xticks(positions, LIBERO_METHODS)
    axis.legend(
        handles=(
            Patch(facecolor="#9b7358", label="Long"),
            Patch(facecolor="#9b7358", alpha=0.48, label="Short"),
        ),
        loc="upper right",
        frameon=False,
        fontsize=7,
        ncols=2,
        handlelength=1.1,
    )


def _draw_three_method_panel(
    axis: Axes,
    *,
    title: str,
    ylabel: str,
    methods: tuple[str, ...],
    values: tuple[float, ...],
    ceiling: float,
) -> None:
    _style_axis(axis, ylabel=ylabel, title=title)
    positions = np.arange(3)
    bars = axis.bar(
        positions,
        values,
        width=0.58,
        color=(BASELINES[0], BASELINES[1], STREAM),
        edgecolor=("none", "none", STREAM_EDGE),
        linewidth=1.15,
        zorder=3,
    )
    _annotate(axis, bars, values, ceiling=ceiling, precision=2, stream_indices=(2,))
    axis.set_ylim(0, ceiling)
    axis.set_xticks(positions, methods)


def _new_figure(metric: str) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
    })
    figure = plt.figure(figsize=(12, 4.5), dpi=200, facecolor=PAPER)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(2, 1, 1),
        left=0.06,
        right=0.98,
        top=0.79,
        bottom=0.29,
        wspace=0.31,
    )
    axes = tuple(figure.add_subplot(grid[0, index]) for index in range(3))
    figure.suptitle(metric, x=0.06, y=0.93, ha="left", color=INK, fontsize=14, fontweight="bold")
    figure.text(
        0.98,
        0.055,
        "Stream-WAM highlighted in teal  ·  hatched bars are Stream-WAM ablations",
        ha="right",
        color=MUTED,
        fontsize=7,
    )
    return figure, axes


def render_chunk_time(output_path: Path) -> None:
    figure, axes = _new_figure("Chunk Time")
    _draw_libero_chunk(axes[0])
    _draw_three_method_panel(
        axes[1], title="RoboTwin 2.0", ylabel="Milliseconds", methods=ROBOTWIN_METHODS,
        values=ROBOTWIN_CHUNK, ceiling=210,
    )
    _draw_three_method_panel(
        axes[2], title="RoboCasa", ylabel="Milliseconds", methods=ROBOCASA_METHODS,
        values=ROBOCASA_CHUNK, ceiling=ROBOCASA_CHUNK_YMAX,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, facecolor=PAPER)
    plt.close(figure)


def render_episode_time(output_path: Path) -> None:
    figure, axes = _new_figure("Episode Time")
    _draw_libero_episode(axes[0])
    _draw_three_method_panel(
        axes[1], title="RoboTwin 2.0", ylabel="Seconds", methods=ROBOTWIN_METHODS,
        values=ROBOTWIN_EPISODE, ceiling=125,
    )
    _draw_three_method_panel(
        axes[2], title="RoboCasa", ylabel="Seconds", methods=ROBOCASA_METHODS,
        values=ROBOCASA_EPISODE, ceiling=ROBOCASA_EPISODE_YMAX,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, facecolor=PAPER)
    plt.close(figure)


def render(output_dir: Path) -> None:
    render_chunk_time(output_dir / "stream-wam-chunk-time.png")
    render_episode_time(output_dir / "stream-wam-episode-time.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("assets"))
    args = parser.parse_args()
    render(args.output_dir)


if __name__ == "__main__":
    main()
