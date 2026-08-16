"""Shared dark-panel visual system for the admin app.

One palette + figure factory reused by ui/indexing.py, ui/journey_tab.py, and
ui/metrics_tab.py so the three tabs read as one system rather than three
independently-styled charts.
"""

from __future__ import annotations

BG     = "#111827"   # panel background
GRID   = "#1f2937"   # gridlines / borders
TEXT   = "#9ca3af"   # axis labels / secondary text
WHITE  = "#f3f4f6"   # emphasized text

INDIGO = "#6366f1"   # primary series
CYAN   = "#22d3ee"   # secondary series
AMBER  = "#f59e0b"   # tertiary / warning
RED    = "#ef4444"   # danger / P99
GREEN  = "#22c55e"   # ok / P50
MUTED  = "#4b5563"   # de-emphasized


def dark_figure(w: float = 6, h: float = 2.6, dpi: int = 110):
    """A Figure + single Axes pre-styled with the shared dark palette."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(w, h), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.spines[["top", "right", "left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    return fig, ax
