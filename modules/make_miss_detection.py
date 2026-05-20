"""
Basketball Shot Outcome Classification — Make / Miss

Part of the AI Basketball Analysis System
https://github.com/behshadarabzadeh/AI-Basketball-Analysis-System-

This module classifies a basketball shot as MAKE or MISS using two signals:

    1. Distance Analysis
       The ball's last tracked position above the rim and first tracked
       position below the rim are used to estimate how closely the ball
       passed to the rim centre during the shot trajectory.

    2. Net Motion Analysis
       After the ball drops below the rim, pixel-level motion is measured
       inside a net region of interest (ROI).

       A made shot typically creates stronger and more consistent
       net movement than a missed shot.

Both signals must agree for a MAKE classification.

Author: Behshad Arabzadeh
"""

import cv2
import math
import numpy as np


# =========================================================
# CONFIGURATION
# =========================================================

# Maximum combined distance between:
#   - last ball position above rim
#   - first ball position below rim
# Smaller values indicate the ball passed closer to the rim centre.
MAKE_SUM_DIST_MAX_PX = 30.0

# Minimum fraction of moving net pixels required
# for valid net motion confirmation.
NET_MOTION_THRESHOLD = 0.17

# Pixel intensity difference threshold for motion detection.
NET_DIFF_THRESHOLD = 25

# Margins used for above/below rim transition logic.
RIM_ABOVE_MARGIN_PX = 2.0
RIM_BELOW_MARGIN_PX = 2.0


# =========================================================
# HELPERS
# =========================================================

def euclidean(p1, p2):
    """
    Computes Euclidean distance between two 2D points.
    """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# =========================================================
# NET MOTION
# =========================================================

def compute_net_motion(prev_frame, curr_frame, net_box, ball_box=None):
    """
    Measures pixel motion inside the net ROI between two frames.

    The basketball bounding box is excluded from the motion calculation
    so only the net contributes to the final motion score.

    Args:
        prev_frame:
            Previous BGR video frame.

        curr_frame:
            Current BGR video frame.

        net_box:
            [x1, y1, x2, y2] coordinates of the net ROI.

        ball_box:
            Optional basketball bounding box to exclude from motion analysis.

    Returns:
        float:
            Fraction of net ROI pixels with detected motion (0.0 → 1.0).
    """

    if prev_frame is None or curr_frame is None or net_box is None:
        return 0.0

    x1, y1, x2, y2 = net_box

    # Extract net ROI from frames
    prev_roi = prev_frame[y1:y2, x1:x2]
    curr_roi = curr_frame[y1:y2, x1:x2]

    if prev_roi.size == 0 or curr_roi.size == 0:
        return 0.0

    # Convert ROIs to grayscale
    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)

    # Frame difference
    diff = cv2.absdiff(prev_gray, curr_gray)

    # Threshold motion areas
    _, thresh = cv2.threshold(
        diff,
        NET_DIFF_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # Valid analysis mask
    mask = np.ones(thresh.shape, dtype=np.uint8) * 255

    # Exclude basketball area from motion calculation
    if ball_box is not None:

        bx1, by1, bx2, by2 = map(int, ball_box)

        mx1 = max(0, bx1 - x1)
        my1 = max(0, by1 - y1)

        mx2 = min(x2 - x1, bx2 - x1)
        my2 = min(y2 - y1, by2 - y1)

        if mx2 > mx1 and my2 > my1:
            mask[my1:my2, mx1:mx2] = 0

    valid_pixels = cv2.countNonZero(mask)

    if valid_pixels == 0:
        return 0.0

    motion_pixels = cv2.countNonZero(
        cv2.bitwise_and(thresh, thresh, mask=mask)
    )

    # Fraction of moving pixels inside valid ROI
    net_motion_score = motion_pixels / valid_pixels

    return net_motion_score


# =========================================================
# MAKE / MISS CLASSIFICATION
# =========================================================

def classify_shot(ball_positions, rim_center, net_motion_values):
    """
    Classifies a basketball shot as MAKE, MISS, or UNCONFIRMED.

    Pipeline:
        1. Detect above-to-below rim transition
        2. Measure ball distance relative to rim centre
        3. Verify net motion after rim crossing

    Args:
        ball_positions:
            List of (x, y) basketball positions ordered by frame time.

        rim_center:
            (cx, cy) image coordinates of the rim centre.

        net_motion_values:
            List of net motion scores collected after the ball
            drops below rim level.

    Returns:
        str:
            "MAKE", "MISS", or "UNCONFIRMED"
    """

    if len(ball_positions) < 2:
        return "UNCONFIRMED"

    last_above_dist = None
    first_below_dist = None

    # Analyse ball trajectory relative to rim
    for pos in ball_positions:

        x, y = pos

        # Distance from ball to rim centre
        dist = euclidean(pos, rim_center)

        # Ball above rim
        if y < rim_center[1] - RIM_ABOVE_MARGIN_PX:
            last_above_dist = dist

        # Ball below rim after previously being above
        if y > rim_center[1] + RIM_BELOW_MARGIN_PX and last_above_dist is not None:
            first_below_dist = dist
            break

    # Rim crossing not detected
    if last_above_dist is None or first_below_dist is None:
        return "UNCONFIRMED"

    # =====================================================
    # SIGNAL 1 — DISTANCE CHECK
    # =====================================================

    distance_ok = (
        last_above_dist + first_below_dist
    ) <= MAKE_SUM_DIST_MAX_PX

    # =====================================================
    # SIGNAL 2 — NET MOTION CHECK
    # =====================================================

    max_net_motion = max(net_motion_values) if net_motion_values else 0.0

    net_motion_ok = max_net_motion >= NET_MOTION_THRESHOLD

    # =====================================================
    # FINAL CLASSIFICATION
    # =====================================================

    if distance_ok and net_motion_ok:
        return "MAKE"

    return "MISS"
