"""Dual-validation make/miss classifier for basketball shot detection."""

from __future__ import annotations

from typing import TypedDict

import cv2
import numpy as np


class ShotResult(TypedDict):
    result: str             # "MAKE" or "MISS"
    rim_crossing_ok: bool   # True if the rim crossing check passed
    net_motion_ok: bool     # True if the net-motion check passed (or was skipped)
    sum_dist: float | None  # last_above_dist + first_below_dist, or None
    net_motion_max: float   # peak net-motion score


def compute_net_motion(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    net_box: tuple[int, int, int, int],
    net_diff_threshold: int,
    ball_box: tuple[float, float, float, float] | None = None,
) -> float:
    """Estimate the fraction of net pixels that changed between two frames.

    Crops both frames to the net ROI, computes a per-pixel absolute difference,
    binarises the result, and returns the ratio of changed pixels to total valid
    pixels. The ball region is optionally masked out so only net-fabric ripple
    is measured, not the ball itself moving through the ROI.

    Args:
        prev_frame:         BGR image from the previous frame (H × W × 3, uint8).
        curr_frame:         BGR image from the current frame (H × W × 3, uint8).
        net_box:            Net ROI as (x1, y1, x2, y2) in image-pixel coordinates.
        net_diff_threshold: Per-pixel absolute-difference threshold for binarising
                            the grayscale frame-diff. Suppresses sensor noise.
        ball_box:           Optional ball bounding box (x1, y1, x2, y2). Its overlap
                            with the net ROI is zeroed so ball pixels are ignored.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 when either frame is None, the ROI
        is empty, or no valid pixels remain after masking.
    """
    if prev_frame is None or curr_frame is None or net_box is None:
        return 0.0

    x1, y1, x2, y2 = net_box
    prev_roi = prev_frame[y1:y2, x1:x2]
    curr_roi = curr_frame[y1:y2, x1:x2]

    if prev_roi.size == 0 or curr_roi.size == 0:
        return 0.0

    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, net_diff_threshold, 255, cv2.THRESH_BINARY)

    mask = np.ones(thresh.shape, dtype=np.uint8) * 255

    if ball_box is not None:
        bx1, by1, bx2, by2 = (int(v) for v in ball_box)
        mx1 = max(0, bx1 - x1)
        my1 = max(0, by1 - y1)
        mx2 = min(x2 - x1, bx2 - x1)
        my2 = min(y2 - y1, by2 - y1)
        if mx2 > mx1 and my2 > my1:
            mask[my1:my2, mx1:mx2] = 0

    valid_pixels = cv2.countNonZero(mask)
    if valid_pixels == 0:
        return 0.0

    motion_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
    return motion_pixels / valid_pixels


def evaluate_make_miss(
    last_above_dist: float | None,
    first_below_dist: float | None,
    net_motion_max: float,
    make_sum_dist_max_px: float,
    net_motion_threshold: float,
    use_net_motion: bool,
) -> ShotResult:
    """Apply the dual-validation rule and return a labelled result.

    Dual-validation logic
    ---------------------
    A shot is a MAKE if and only if both checks pass:

    Check 1 — Rim crossing validation
        last_above_dist + first_below_dist <= make_sum_dist_max_px

        last_above_dist is the Euclidean distance (px) from the ball centre to
        the rim centre at the last frame the ball was above the rim plane.
        first_below_dist is the same measurement for the first frame the ball
        dropped below. Their sum is small when the ball passes directly through
        the rim and large when it arcs wide or clangs off the backboard.

    Check 2 — Net motion  (optional, controlled by use_net_motion)
        net_motion_max >= net_motion_threshold

        The peak frame-diff score recorded over the observation window must
        exceed the threshold. When use_net_motion is False this check always
        passes, making geometry the sole arbiter.

    Args:
        last_above_dist:      Rim distance (px) at the last above-rim frame.
                              None if never observed.
        first_below_dist:     Rim distance (px) at the first below-rim frame.
                              None if descent was not captured.
        net_motion_max:       Peak net-motion fraction from compute_net_motion
                              calls over the observation window.
        make_sum_dist_max_px: Max allowed sum of distances (px) for a MAKE.
        net_motion_threshold: Minimum net-motion fraction required for a MAKE.
        use_net_motion:       If False, the net-motion check always passes.

    Returns:
        ShotResult with keys: result, rim_crossing_ok, net_motion_ok,
        sum_dist, net_motion_max.
    """
    sum_dist = (
        None
        if (last_above_dist is None or first_below_dist is None)
        else last_above_dist + first_below_dist
    )

    rim_crossing_ok = sum_dist is not None and sum_dist <= make_sum_dist_max_px
    net_motion_ok = net_motion_max >= net_motion_threshold if use_net_motion else True

    return ShotResult(
        result="MAKE" if (rim_crossing_ok and net_motion_ok) else "MISS",
        rim_crossing_ok=rim_crossing_ok,
        net_motion_ok=net_motion_ok,
        sum_dist=sum_dist,
        net_motion_max=net_motion_max,
    )
