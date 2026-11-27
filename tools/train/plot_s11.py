#!/usr/bin/env python3
"""Plot the S11 anytime curve (eta-field MSE vs cost per 60 s episode) as an SVG.

Usage (from repo root):
    .venv-train/bin/python tools/train/plot_s11.py <output.svg>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

INPUT_JSON = Path("tools/train/results/s11_anytime.json")

# Canvas and plot area geometry (px)
WIDTH = 920
HEIGHT = 620
PLOT_LEFT = 86.0
PLOT_RIGHT = 884.0
PLOT_TOP = 72.0
PLOT_BOTTOM = 532.0

# Log-log axis bounds
X_MIN = 0.01
X_MAX = 4.0
Y_MIN = 1e-4
Y_MAX = 1.0

LOG_X_MIN = float(np.log10(X_MIN))
LOG_X_MAX = float(np.log10(X_MAX))
LOG_Y_MIN = float(np.log10(Y_MIN))
LOG_Y_MAX = float(np.log10(Y_MAX))

# U-Net and persistence reference points cited from S10-GENERALIZATION.md
# (evaluated over all 8 heldout episodes in S10).
UNET_ONE_STEP = (1.45, 0.037384, "U-Net one-step 0.0374")
UNET_ROLLOUT_4S = (1.45, 0.240828, "U-Net rollout 4 s 0.241")
PERSISTENCE_ONE_STEP = (0.012, 0.034480, "persistence one-step 0.0345 (cost ~ 0)")

TITLE = (
    "S11 anytime curve \u2014 heldout swell (H0 2.1 m, T 10 s, +6 deg), "
    "eta MSE vs cost per 60 s episode"
)
CAPTION = (
    "Error definitions differ by family: teacher = full 60 s trajectory field MSE "
    "vs full-res truth; U-Net/persistence = S10 one-step / 4 s-rollout MSE over all "
    "8 heldout episodes."
)

# Palette
COLOR_BG = "#ffffff"
COLOR_AXIS = "#1f2937"
COLOR_TEXT = "#111827"
COLOR_MUTED = "#4b5563"
COLOR_GRID_MAJOR = "#d1d5db"
COLOR_GRID_MINOR = "#f1f5f9"
COLOR_TEACHER = "#1d4ed8"
COLOR_TEACHER_EP = "#60a5fa"
COLOR_UNET = "#d95f02"
COLOR_PERSIST = "#15803d"


def map_x(val: float) -> float:
    """Map positive x value in [X_MIN, X_MAX] to SVG x coordinate."""
    clamped = float(np.clip(val, X_MIN, X_MAX))
    t = (float(np.log10(clamped)) - LOG_X_MIN) / (LOG_X_MAX - LOG_X_MIN)
    return PLOT_LEFT + t * (PLOT_RIGHT - PLOT_LEFT)


def map_y(val: float) -> float:
    """Map positive y value in [Y_MIN, Y_MAX] to SVG y coordinate."""
    clamped = float(np.clip(val, Y_MIN, Y_MAX))
    t = (float(np.log10(clamped)) - LOG_Y_MIN) / (LOG_Y_MAX - LOG_Y_MIN)
    return PLOT_BOTTOM - t * (PLOT_BOTTOM - PLOT_TOP)


def fmt_mse(val: float) -> str:
    """Format an MSE value cleanly for inline plot labels."""
    if val < 1e-3:
        return f"{val:.2e}"
    if val < 0.1:
        return f"{val:.4f}"
    return f"{val:.3f}"


def diamond_points(cx: float, cy: float, r: float) -> str:
    """Return SVG polygon points for a diamond centred at (cx, cy)."""
    pts = [
        (cx, cy - r),
        (cx + r, cy),
        (cx, cy + r),
        (cx - r, cy),
    ]
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def triangle_points(cx: float, cy: float, r: float) -> str:
    """Return SVG polygon points for an upward triangle centred at (cx, cy)."""
    dx = r * 0.92
    pts = [
        (cx, cy - r),
        (cx + dx, cy + r * 0.72),
        (cx - dx, cy + r * 0.72),
    ]
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def svg_text(
    x: float,
    y: float,
    content: str,
    *,
    size: float = 12,
    weight: str = "normal",
    anchor: str = "start",
    fill: str = COLOR_TEXT,
    transform: str | None = None,
) -> str:
    attrs = [
        f'x="{x:.2f}"',
        f'y="{y:.2f}"',
        f'font-size="{size}"',
        f'font-weight="{weight}"',
        f'text-anchor="{anchor}"',
        f'fill="{fill}"',
    ]
    if transform:
        attrs.append(f'transform="{escape(transform)}"')
    return f"<text {' '.join(attrs)}>{escape(content)}</text>"


def build_svg(data: dict) -> str:
    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" '
        'font-family="ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, '
        'Segoe UI, Roboto, Helvetica, Arial, sans-serif">'
    )
    lines.append(f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="{COLOR_BG}"/>')

    # Title
    lines.append(
        "  "
        + svg_text(
            WIDTH / 2,
            28,
            TITLE,
            size=14.5,
            weight="bold",
            anchor="middle",
            fill=COLOR_TEXT,
        )
    )

    # Compact legend bar above plot frame
    ly = 51.0
    lines.append('  <g id="legend" font-size="11.5" fill="' + COLOR_TEXT + '">')
    # Teacher episode circle
    lines.append(
        f'    <circle cx="135" cy="{ly:.1f}" r="4" fill="{COLOR_TEACHER_EP}" '
        f'fill-opacity="0.7" stroke="{COLOR_TEACHER}" stroke-width="1"/>'
    )
    lines.append("    " + svg_text(145, ly + 4, "teacher episode", size=11.5))
    # Teacher mean diamond + line
    lines.append(
        f'    <line x1="255" y1="{ly:.1f}" x2="283" y2="{ly:.1f}" '
        f'stroke="{COLOR_TEACHER}" stroke-width="2"/>'
    )
    lines.append(
        f'    <polygon points="{diamond_points(269, ly, 6)}" '
        f'fill="{COLOR_TEACHER}" stroke="#ffffff" stroke-width="1"/>'
    )
    lines.append("    " + svg_text(290, ly + 4, "teacher mean (4 / 8 / 16 m)", size=11.5))
    # U-Net square
    lines.append(
        f'    <rect x="478" y="{ly - 5:.1f}" width="10" height="10" '
        f'fill="{COLOR_UNET}" stroke="#7c2d12" stroke-width="1"/>'
    )
    lines.append("    " + svg_text(495, ly + 4, "U-Net (S10)", size=11.5))
    # Persistence triangle
    lines.append(
        f'    <polygon points="{triangle_points(598, ly + 0.5, 6.5)}" '
        f'fill="{COLOR_PERSIST}" stroke="#14532d" stroke-width="1"/>'
    )
    lines.append("    " + svg_text(611, ly + 4, "persistence (S10)", size=11.5))
    lines.append("  </g>")

    # Gridlines: minor then major
    lines.append('  <g id="grid">')
    # X minor gridlines
    for exp in (-2, -1, 0):
        base = 10.0**exp
        for mult in range(2, 10):
            xv = mult * base
            if X_MIN < xv < X_MAX:
                px = map_x(xv)
                lines.append(
                    f'    <line x1="{px:.2f}" y1="{PLOT_TOP:.2f}" '
                    f'x2="{px:.2f}" y2="{PLOT_BOTTOM:.2f}" '
                    f'stroke="{COLOR_GRID_MINOR}" stroke-width="1"/>'
                )
    # Y minor gridlines
    for exp in (-4, -3, -2, -1):
        base = 10.0**exp
        for mult in range(2, 10):
            yv = mult * base
            if Y_MIN < yv < Y_MAX:
                py = map_y(yv)
                lines.append(
                    f'    <line x1="{PLOT_LEFT:.2f}" y1="{py:.2f}" '
                    f'x2="{PLOT_RIGHT:.2f}" y2="{py:.2f}" '
                    f'stroke="{COLOR_GRID_MINOR}" stroke-width="1"/>'
                )

    # X major decade gridlines + ticks
    x_major_ticks = [
        (0.01, "0.01"),
        (0.1, "0.1"),
        (1.0, "1"),
    ]
    x_extra_ticks = [
        (0.02, "0.02"),
        (0.05, "0.05"),
        (0.2, "0.2"),
        (0.5, "0.5"),
        (2.0, "2"),
        (4.0, "4"),
    ]
    for xv, _ in x_major_ticks:
        px = map_x(xv)
        lines.append(
            f'    <line x1="{px:.2f}" y1="{PLOT_TOP:.2f}" '
            f'x2="{px:.2f}" y2="{PLOT_BOTTOM:.2f}" '
            f'stroke="{COLOR_GRID_MAJOR}" stroke-width="1.2"/>'
        )

    # Y major decade gridlines + ticks
    y_major_ticks = [
        (1e-4, "1e-4"),
        (1e-3, "1e-3"),
        (1e-2, "1e-2"),
        (1e-1, "1e-1"),
        (1.0, "1"),
    ]
    for yv, _ in y_major_ticks:
        py = map_y(yv)
        lines.append(
            f'    <line x1="{PLOT_LEFT:.2f}" y1="{py:.2f}" '
            f'x2="{PLOT_RIGHT:.2f}" y2="{py:.2f}" '
            f'stroke="{COLOR_GRID_MAJOR}" stroke-width="1.2"/>'
        )
    lines.append("  </g>")

    # Plot frame
    lines.append(
        f'  <rect x="{PLOT_LEFT:.2f}" y="{PLOT_TOP:.2f}" '
        f'width="{PLOT_RIGHT - PLOT_LEFT:.2f}" height="{PLOT_BOTTOM - PLOT_TOP:.2f}" '
        f'fill="none" stroke="{COLOR_AXIS}" stroke-width="1.4"/>'
    )

    # Axis ticks and tick labels
    lines.append('  <g id="axes">')
    for xv, lbl in x_major_ticks:
        px = map_x(xv)
        lines.append(
            f'    <line x1="{px:.2f}" y1="{PLOT_BOTTOM:.2f}" '
            f'x2="{px:.2f}" y2="{PLOT_BOTTOM + 6:.2f}" '
            f'stroke="{COLOR_AXIS}" stroke-width="1.2"/>'
        )
        lines.append(
            "    "
            + svg_text(
                px,
                PLOT_BOTTOM + 20,
                lbl,
                size=11.5,
                weight="bold",
                anchor="middle",
                fill=COLOR_TEXT,
            )
        )
    for xv, lbl in x_extra_ticks:
        px = map_x(xv)
        lines.append(
            f'    <line x1="{px:.2f}" y1="{PLOT_BOTTOM:.2f}" '
            f'x2="{px:.2f}" y2="{PLOT_BOTTOM + 4:.2f}" '
            f'stroke="{COLOR_MUTED}" stroke-width="1"/>'
        )
        lines.append(
            "    "
            + svg_text(
                px,
                PLOT_BOTTOM + 19,
                lbl,
                size=10.5,
                anchor="middle",
                fill=COLOR_MUTED,
            )
        )

    for yv, lbl in y_major_ticks:
        py = map_y(yv)
        lines.append(
            f'    <line x1="{PLOT_LEFT - 6:.2f}" y1="{py:.2f}" '
            f'x2="{PLOT_LEFT:.2f}" y2="{py:.2f}" '
            f'stroke="{COLOR_AXIS}" stroke-width="1.2"/>'
        )
        lines.append(
            "    "
            + svg_text(
                PLOT_LEFT - 10,
                py + 4,
                lbl,
                size=11.5,
                anchor="end",
                fill=COLOR_TEXT,
            )
        )

    # Axis titles
    x_mid = 0.5 * (PLOT_LEFT + PLOT_RIGHT)
    y_mid = 0.5 * (PLOT_TOP + PLOT_BOTTOM)
    lines.append(
        "    "
        + svg_text(
            x_mid,
            574,
            "Cost (seconds per 60 s episode)",
            size=13,
            weight="bold",
            anchor="middle",
            fill=COLOR_TEXT,
        )
    )
    lines.append(
        "    "
        + svg_text(
            24,
            y_mid,
            "eta-field MSE (m^2)",
            size=13,
            weight="bold",
            anchor="middle",
            fill=COLOR_TEXT,
            transform=f"rotate(-90 24 {y_mid:.2f})",
        )
    )
    lines.append("  </g>")

    # Extract teacher family levels and episodes
    episodes = data.get("episodes", [])
    aggregate = data.get("aggregate", {})
    cell_levels = sorted(int(k) for k in aggregate.keys())

    # Teacher family: episode circles
    lines.append('  <g id="teacher-episodes">')
    for ep in episodes:
        for lvl in ep.get("levels", []):
            px = map_x(float(lvl["costMedianS"]))
            py = map_y(float(lvl["fieldMSE"]))
            lines.append(
                f'    <circle cx="{px:.2f}" cy="{py:.2f}" r="4.2" '
                f'fill="{COLOR_TEACHER_EP}" fill-opacity="0.65" '
                f'stroke="{COLOR_TEACHER}" stroke-width="1"/>'
            )
    lines.append("  </g>")

    # Teacher family: mean diamonds connected by a polyline
    mean_points: list[tuple[int, float, float, float, float]] = []
    for cm in cell_levels:
        agg = aggregate[str(cm)]
        cost = float(agg["meanCostMedianS"])
        mse = float(agg["meanFieldMSE"])
        mean_points.append((cm, cost, mse, map_x(cost), map_y(mse)))

    lines.append('  <g id="teacher-means">')
    if len(mean_points) >= 2:
        poly_pts = " ".join(f"{px:.2f},{py:.2f}" for _, _, _, px, py in mean_points)
        lines.append(
            f'    <polyline points="{poly_pts}" fill="none" '
            f'stroke="{COLOR_TEACHER}" stroke-width="2.2"/>'
        )

    for cm, _cost, mse, px, py in mean_points:
        lines.append(
            f'    <polygon points="{diamond_points(px, py, 7.5)}" '
            f'fill="{COLOR_TEACHER}" stroke="#ffffff" stroke-width="1.3"/>'
        )
        label = f"teacher {cm} m ({fmt_mse(mse)})"
        # Place labels cleanly inside the plot frame without clipping
        if px > PLOT_RIGHT - 170:
            lx = px - 12
            anchor = "end"
        else:
            lx = px + 12
            anchor = "start"
        ly_lbl = py - 8 if py > PLOT_BOTTOM - 22 else py + 4
        if cm == 16 and py < PLOT_TOP + 22:
            ly_lbl = py + 16
        lines.append(
            "    "
            + svg_text(
                lx,
                ly_lbl,
                label,
                size=11.5,
                weight="bold",
                anchor=anchor,
                fill=COLOR_TEACHER,
            )
        )
    lines.append("  </g>")

    # U-Net points (squares)
    lines.append('  <g id="unet-points">')
    sq_half = 5.8
    for cost, mse, label in (UNET_ONE_STEP, UNET_ROLLOUT_4S):
        px = map_x(cost)
        py = map_y(mse)
        lines.append(
            f'    <rect x="{px - sq_half:.2f}" y="{py - sq_half:.2f}" '
            f'width="{sq_half * 2:.2f}" height="{sq_half * 2:.2f}" '
            f'fill="{COLOR_UNET}" stroke="#7c2d12" stroke-width="1.2"/>'
        )
        lines.append(
            "    "
            + svg_text(
                px - 11,
                py + 4,
                label,
                size=11.5,
                weight="bold",
                anchor="end",
                fill="#9a3412",
            )
        )
    lines.append("  </g>")

    # Persistence one-step point (triangle)
    lines.append('  <g id="persistence-point">')
    p_cost, p_mse, p_label = PERSISTENCE_ONE_STEP
    p_px = map_x(p_cost)
    p_py = map_y(p_mse)
    lines.append(
        f'    <polygon points="{triangle_points(p_px, p_py, 7.5)}" '
        f'fill="{COLOR_PERSIST}" stroke="#14532d" stroke-width="1.2"/>'
    )
    lines.append(
        "    "
        + svg_text(
            p_px + 12,
            p_py + 4,
            p_label,
            size=11.5,
            weight="bold",
            anchor="start",
            fill=COLOR_PERSIST,
        )
    )
    lines.append("  </g>")

    # Caption line at the bottom
    lines.append(
        "  "
        + svg_text(
            WIDTH / 2,
            604,
            CAPTION,
            size=10,
            anchor="middle",
            fill=COLOR_MUTED,
        )
    )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output.svg>", file=sys.stderr)
        return 2

    if not INPUT_JSON.is_file():
        print(f"Error: input JSON not found at {INPUT_JSON}", file=sys.stderr)
        return 1

    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    svg = build_svg(data)

    out_path = Path(sys.argv[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
