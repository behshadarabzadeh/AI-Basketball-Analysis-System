"""
Court zone classification and shot-chart rendering.

Classifies a court-space (x, y) point in metres into one of 13 named scoring
zones (paint / mid-range / three-point, split into corner-wing-top sectors)
and renders a shot chart with per-zone attempt/make/percentage labels. Shot
records passed into plot_auto_chart are opaque dicts describing where a shot
was taken from, who is credited with it, and whether it went in; deciding
those three things is not this module's job. Attempt detection and player
attribution are handled by private upstream subsystems, while make/miss
classification is handled by the separately published
make_miss_classifier.py module. This file only classifies court zones and
renders the resulting shot records.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc, Polygon

from config import (
    HALF_W, Y_MIN, Y_MAX, LANE_HALF, LANE_W, Y_FT_LINE, FT_CIRCLE_D,
    LOWMID_R, THREE_ARC_R, CORNER_3_X, CORNER_3_Y_INT, theta_right, theta_left,
    TOP_5, WING_5, TOP_3, COURT_COLOR, COURT_LW, ZONE_LINE_COLOR, ZONE_LW,
    ZONE_ORDER, ZONE_FILL_ORDER,
)


def is_made_result(value):
    return str(value).strip().upper() == "MAKE"


def phi_deg(x, y):
    return math.degrees(math.atan2(x, y))


def sector_3(phi):
    if phi < -TOP_3:
        return "LEFT"
    elif phi > TOP_3:
        return "RIGHT"
    return "CENTER"


def sector_5(phi):
    if phi < -WING_5:
        return "LEFT_CORNER"
    elif phi < -TOP_5:
        return "LEFT_WING"
    elif phi <= TOP_5:
        return "TOP"
    elif phi <= WING_5:
        return "RIGHT_WING"
    return "RIGHT_CORNER"


def is_three_point(x, y):
    ax = abs(x)
    r = math.hypot(x, y)

    if Y_MIN <= y <= CORNER_3_Y_INT:
        if CORNER_3_X <= ax <= HALF_W:
            return True, ("LEFT_CORNER" if x < 0 else "RIGHT_CORNER")

    elif CORNER_3_Y_INT < y <= THREE_ARC_R * math.cos(math.radians(WING_5)):
        phi = phi_deg(x, y)
        if (r >= THREE_ARC_R) and (abs(phi) >= WING_5):
            return True, ("LEFT_CORNER" if x < 0 else "RIGHT_CORNER")

    if (r >= THREE_ARC_R) and (y >= CORNER_3_Y_INT):
        return True, sector_5(phi_deg(x, y))

    return False, None


def classify_zone(x, y):
    if x < -HALF_W or x > HALF_W or y < Y_MIN or y > Y_MAX:
        return "OUT_OF_BOUNDS"

    r = math.hypot(x, y)
    phi = phi_deg(x, y)
    is3, three_area = is_three_point(x, y)

    if is3:
        return f"3PT_{three_area}"
    elif r <= LOWMID_R:
        return f"LOWMID_{sector_3(phi)}"
    else:
        return f"HIGHMID_{sector_5(phi)}"


def zone_fill_color(pct):
    if pct is None:
        return (0.95, 0.95, 0.95, 0.35)
    if pct < 30:
        return (1.00, 0.78, 0.78, 0.60)
    elif pct < 45:
        return (1.00, 0.86, 0.68, 0.62)
    elif pct < 65:
        return (1.00, 0.95, 0.58, 0.68)
    else:
        return (0.72, 0.90, 0.72, 0.68)


def polar_to_xy(r, phi_deg_val):
    rad = math.radians(phi_deg_val)
    x = r * math.sin(rad)
    y = r * math.cos(rad)
    return x, y


def arc_points(r, phi1, phi2, n=80):
    vals = np.linspace(phi1, phi2, n)
    return [polar_to_xy(r, p) for p in vals]


def point_on_circle_by_y(r, y, side="left"):
    x_abs = math.sqrt(max(0.0, r * r - y * y))
    return (-x_abs, y) if side == "left" else (x_abs, y)


def corner_phi_deg():
    return math.degrees(math.atan2(CORNER_3_X, CORNER_3_Y_INT))


def annulus_sector_polygon(r_inner, r_outer, phi1, phi2, n=120):
    outer = arc_points(r_outer, phi1, phi2, n=n)
    inner = arc_points(r_inner, phi2, phi1, n=n)
    return outer + inner


def wedge_polygon(r, phi1, phi2, n=120):
    return [(0.0, 0.0)] + arc_points(r, phi1, phi2, n=n)


def build_zone_polygons():
    zones = {}

    corner_phi = corner_phi_deg()
    outer_cap_r = 9.6

    left_inner_base = point_on_circle_by_y(LOWMID_R, Y_MIN, side="left")
    right_inner_base = point_on_circle_by_y(LOWMID_R, Y_MIN, side="right")

    left_inner_base_phi = math.degrees(math.atan2(left_inner_base[0], left_inner_base[1]))
    right_inner_base_phi = math.degrees(math.atan2(right_inner_base[0], right_inner_base[1]))

    outer_sideline_y = math.sqrt(max(0.0, outer_cap_r**2 - HALF_W**2))
    left_outer_sideline_pt = (-HALF_W, outer_sideline_y)
    right_outer_sideline_pt = (HALF_W, outer_sideline_y)

    left_outer_sideline_phi = math.degrees(math.atan2(left_outer_sideline_pt[0], left_outer_sideline_pt[1]))
    right_outer_sideline_phi = math.degrees(math.atan2(right_outer_sideline_pt[0], right_outer_sideline_pt[1]))

    zones["LOWMID_LEFT"] = wedge_polygon(LOWMID_R, left_inner_base_phi, -TOP_3, n=140)
    zones["LOWMID_CENTER"] = wedge_polygon(LOWMID_R, -TOP_3, TOP_3, n=140)
    zones["LOWMID_RIGHT"] = wedge_polygon(LOWMID_R, TOP_3, right_inner_base_phi, n=140)

    zones["HIGHMID_LEFT_CORNER"] = (
        [(-CORNER_3_X, Y_MIN), (-CORNER_3_X, CORNER_3_Y_INT)] +
        arc_points(THREE_ARC_R, -corner_phi, -WING_5, n=120) +
        arc_points(LOWMID_R, -WING_5, left_inner_base_phi, n=120) +
        [left_inner_base]
    )

    zones["HIGHMID_LEFT_WING"] = annulus_sector_polygon(LOWMID_R, THREE_ARC_R, -WING_5, -TOP_5, n=140)
    zones["HIGHMID_TOP"] = annulus_sector_polygon(LOWMID_R, THREE_ARC_R, -TOP_5, TOP_5, n=160)
    zones["HIGHMID_RIGHT_WING"] = annulus_sector_polygon(LOWMID_R, THREE_ARC_R, TOP_5, WING_5, n=140)

    zones["HIGHMID_RIGHT_CORNER"] = (
        [(CORNER_3_X, Y_MIN), right_inner_base] +
        arc_points(LOWMID_R, right_inner_base_phi, WING_5, n=120) +
        arc_points(THREE_ARC_R, WING_5, corner_phi, n=120) +
        [(CORNER_3_X, CORNER_3_Y_INT)]
    )

    zones["3PT_LEFT_CORNER"] = (
        [(-HALF_W, Y_MIN), (-CORNER_3_X, Y_MIN), (-CORNER_3_X, CORNER_3_Y_INT)] +
        arc_points(THREE_ARC_R, -corner_phi, -WING_5, n=120) +
        arc_points(outer_cap_r, -WING_5, left_outer_sideline_phi, n=120) +
        [left_outer_sideline_pt]
    )

    zones["3PT_RIGHT_CORNER"] = (
        [(CORNER_3_X, Y_MIN), (HALF_W, Y_MIN), right_outer_sideline_pt] +
        arc_points(outer_cap_r, right_outer_sideline_phi, WING_5, n=120) +
        arc_points(THREE_ARC_R, WING_5, corner_phi, n=120) +
        [(CORNER_3_X, CORNER_3_Y_INT)]
    )

    zones["3PT_LEFT_WING"] = annulus_sector_polygon(THREE_ARC_R, 9.6, -WING_5, -TOP_5, n=140)
    zones["3PT_TOP"] = annulus_sector_polygon(THREE_ARC_R, 9.6, -TOP_5, TOP_5, n=160)
    zones["3PT_RIGHT_WING"] = annulus_sector_polygon(THREE_ARC_R, 9.6, TOP_5, WING_5, n=140)

    return zones


def zone_label_position(zone_name):
    positions = {
        "LOWMID_LEFT": (-1.7, 2.7),
        "LOWMID_CENTER": (0.0, 3.1),
        "LOWMID_RIGHT": (1.7, 2.7),
        "HIGHMID_LEFT_CORNER": (-5.2, 2.1),
        "HIGHMID_LEFT_WING": (-4.4, 5.0),
        "HIGHMID_TOP": (0.0, 5.5),
        "HIGHMID_RIGHT_WING": (4.4, 5.0),
        "HIGHMID_RIGHT_CORNER": (5.2, 2.1),
        "3PT_LEFT_CORNER": (-6.9, 2.7),
        "3PT_LEFT_WING": (-5.8, 7.0),
        "3PT_TOP": (0.0, 7.5),
        "3PT_RIGHT_WING": (5.8, 7.0),
        "3PT_RIGHT_CORNER": (6.9, 2.7),
    }
    return positions[zone_name]


def add_filled_zone(ax, pts, color):
    poly = Polygon(pts, closed=True, facecolor=color, edgecolor="none", zorder=0)
    ax.add_patch(poly)


def ray(ax, phi, r1, r2, color, lw):
    rad = math.radians(phi)
    ax.plot(
        [r1 * math.sin(rad), r2 * math.sin(rad)],
        [r1 * math.cos(rad), r2 * math.cos(rad)],
        color=color, lw=lw
    )


def draw_angle_lines(ax, rmax=9.5):
    for a in (-TOP_3, TOP_3):
        ray(ax, a, 0.0, LOWMID_R, ZONE_LINE_COLOR, ZONE_LW)
    for a in (-WING_5, -TOP_5, TOP_5, WING_5):
        ray(ax, a, LOWMID_R, rmax, ZONE_LINE_COLOR, ZONE_LW)


def draw_base_court(ax):
    ax.plot([-HALF_W, HALF_W], [Y_MIN, Y_MIN], color=COURT_COLOR, lw=COURT_LW)
    ax.plot([-HALF_W, -HALF_W], [Y_MIN, Y_MAX], color=COURT_COLOR, lw=COURT_LW)
    ax.plot([HALF_W, HALF_W], [Y_MIN, Y_MAX], color=COURT_COLOR, lw=COURT_LW)
    ax.plot([-HALF_W, HALF_W], [Y_MAX, Y_MAX], color=COURT_COLOR, lw=COURT_LW)

    ax.add_patch(Rectangle(
        (-LANE_HALF, Y_MIN),
        LANE_W,
        (Y_FT_LINE - Y_MIN),
        fill=False,
        color=COURT_COLOR,
        lw=COURT_LW
    ))

    ax.add_patch(Arc(
        (0, Y_FT_LINE),
        FT_CIRCLE_D,
        FT_CIRCLE_D,
        theta1=0,
        theta2=180,
        color=COURT_COLOR,
        lw=COURT_LW
    ))

    ax.plot([CORNER_3_X, CORNER_3_X], [Y_MIN, CORNER_3_Y_INT], color=COURT_COLOR, lw=COURT_LW)
    ax.plot([-CORNER_3_X, -CORNER_3_X], [Y_MIN, CORNER_3_Y_INT], color=COURT_COLOR, lw=COURT_LW)

    ax.add_patch(Arc(
        (0, 0),
        2 * THREE_ARC_R,
        2 * THREE_ARC_R,
        theta1=theta_right,
        theta2=theta_left,
        color=COURT_COLOR,
        lw=COURT_LW
    ))

    ax.add_patch(Circle((0, 0), 0.24, fill=False, color=COURT_COLOR, lw=3.0))

    ax.add_patch(Arc(
        (0, 0),
        2 * LOWMID_R,
        2 * LOWMID_R,
        theta1=-10,
        theta2=190,
        color=ZONE_LINE_COLOR,
        lw=ZONE_LW
    ))

    draw_angle_lines(ax, rmax=9.5)

    ax.set_xlim(-HALF_W, HALF_W)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")


def compute_zone_stats(shots):
    stats = {z: {"attempts": 0, "made": 0, "pct": None} for z in ZONE_ORDER}

    for sh in shots:
        z = sh["zone"]
        if z in stats:
            stats[z]["attempts"] += 1
            if is_made_result(sh["result"]):
                stats[z]["made"] += 1

    for z in ZONE_ORDER:
        att = stats[z]["attempts"]
        made = stats[z]["made"]
        if att > 0:
            stats[z]["pct"] = 100.0 * made / att

    return stats


def plot_auto_chart(player_name, shots):
    """
    shots: list of {"player": str, "zone": str, "result": "MAKE"|"MISS"
    (case-insensitive), "court_xy": (x, y)} records, already resolved
    upstream.
    """
    stats = compute_zone_stats(shots)
    zone_polys = build_zone_polygons()

    total_attempts = len(shots)
    total_made = sum(1 for s in shots if is_made_result(s["result"]))
    total_pct = 100.0 * total_made / total_attempts if total_attempts > 0 else 0.0

    fig, ax = plt.subplots(figsize=(11, 10))

    for z in ZONE_FILL_ORDER:
        pts = zone_polys.get(z, None)
        if pts is not None and len(pts) >= 3:
            add_filled_zone(ax, pts, zone_fill_color(stats[z]["pct"]))

    draw_base_court(ax)

    for z in ZONE_ORDER:
        x, y = zone_label_position(z)
        att = stats[z]["attempts"]
        made = stats[z]["made"]
        pct = stats[z]["pct"]
        pct_txt = "0%" if pct is None else f"{pct:.0f}%"

        txt = f"{made}/{att}\n{pct_txt}"
        ax.text(
            x, y, txt,
            ha="center", va="center",
            fontsize=9, color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.55),
            zorder=10
        )

    for sh in shots:
        x, y = sh["court_xy"]
        if is_made_result(sh["result"]):
            ax.scatter([x], [y], s=110, marker="o", color="gray", zorder=20)
        else:
            ax.scatter([x], [y], s=110, marker="x", color="gray", linewidths=2.2, zorder=20)

        ax.text(x + 0.12, y + 0.12, sh["player"], fontsize=7, color="black", zorder=25)

    title = f"{player_name} | Shots: {total_attempts} | Made: {total_made} | FG%: {total_pct:.1f}%"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()
    plt.show()
