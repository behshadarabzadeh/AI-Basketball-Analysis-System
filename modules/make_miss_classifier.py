"""
Shot outcome (make/miss) classification once a ball is already descending
near the rim.

Given a stream of ball centre positions relative to the rim line, plus a
frame-differencing motion signal from a fixed net region of interest, this
module decides whether a shot went in: it records the last position seen
above the rim line and the first position seen below it, waits a short
window, and checks whether enough net motion occurred to confirm a make.

Deliberately excluded: the upward-plus-toward-rim velocity heuristic that
first decides a ball is being shot at all, and any attribution of a shot to
a specific player. Both belong to a separate private workflow that hands a
ball to this classifier only after it has already decided a shot attempt is
under way.
"""

import math

import cv2
import numpy as np

from config import (
    RIM_ABOVE_MARGIN_PX, RIM_BELOW_MARGIN_PX, MAKE_SUM_DIST_MAX_PX,
    USE_NET_MOTION_FOR_MAKE, NET_CHECK_AFTER_BELOW_FRAMES,
    NET_MOTION_THRESHOLD, NET_DIFF_THRESHOLD,
)


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_net_motion(prev_frame, curr_frame, net_box, ball_box=None):
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
    _, thresh = cv2.threshold(diff, NET_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    mask = np.ones(thresh.shape, dtype=np.uint8) * 255

    if ball_box is not None:
        bx1, by1, bx2, by2 = map(int, ball_box)

        mx1 = max(0, bx1 - x1)
        my1 = max(0, by1 - y1)
        mx2 = min(x2 - x1, bx2 - x1)
        my2 = min(y2 - y1, by2 - y1)

        if mx2 > mx1 and my2 > my1:
            mask[my1:my2, mx1:mx2] = 0

    valid_pixels = cv2.countNonZero(mask)
    if valid_pixels <= 0:
        return 0.0

    motion_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
    return motion_pixels / valid_pixels


class ShotOutcomeTracker:
    """
    Tracks one in-flight ball around the rim line and produces a MAKE/MISS
    verdict. The caller decides when a shot attempt has started (out of
    scope for this module) and creates one tracker per attempt.
    """

    def __init__(self):
        self.last_above_center = None
        self.last_above_dist = None
        self.last_above_frame = None

        self.first_below_center = None
        self.first_below_dist = None
        self.first_below_frame = None

        self.waiting_net_check = False
        self.net_check_start_frame = None
        self.net_motion_max = 0.0

    def update(self, frame_idx, ball_center, rim_center):
        """
        Feed one frame's ball position. Call this every frame until
        waiting_net_check becomes True, then switch to check_net_motion().
        """
        curr_dist = euclidean(ball_center, rim_center)
        is_above = ball_center[1] < rim_center[1] - RIM_ABOVE_MARGIN_PX
        is_below = ball_center[1] > rim_center[1] + RIM_BELOW_MARGIN_PX

        if is_above and not self.waiting_net_check:
            self.last_above_center = ball_center
            self.last_above_dist = curr_dist
            self.last_above_frame = frame_idx

        elif is_below and not self.waiting_net_check:
            self.first_below_center = ball_center
            self.first_below_dist = curr_dist
            self.first_below_frame = frame_idx

            self.waiting_net_check = True
            self.net_check_start_frame = frame_idx
            self.net_motion_max = 0.0

    def check_net_motion(self, frame_idx, prev_frame, curr_frame, net_box, ball_box):
        """
        Call once per frame while waiting_net_check is True. Returns a
        result dict once NET_CHECK_AFTER_BELOW_FRAMES have elapsed since the
        ball first appeared below the rim line, else None.
        """
        if not self.waiting_net_check:
            return None

        net_motion = compute_net_motion(prev_frame, curr_frame, net_box, ball_box=ball_box)
        self.net_motion_max = max(self.net_motion_max, net_motion)

        if frame_idx - self.net_check_start_frame < NET_CHECK_AFTER_BELOW_FRAMES:
            return None

        distance_make_ok = (
            self.last_above_dist is not None and
            self.first_below_dist is not None and
            (self.last_above_dist + self.first_below_dist) <= MAKE_SUM_DIST_MAX_PX
        )

        net_motion_ok = self.net_motion_max >= NET_MOTION_THRESHOLD if USE_NET_MOTION_FOR_MAKE else True
        result = "MAKE" if distance_make_ok and net_motion_ok else "MISS"

        result_info = {
            "result": result,
            "last_above_dist": self.last_above_dist,
            "first_below_dist": self.first_below_dist,
            "sum_dist": None if self.last_above_dist is None or self.first_below_dist is None
            else self.last_above_dist + self.first_below_dist,
            "net_motion": self.net_motion_max,
            "net_motion_ok": net_motion_ok,
            "distance_make_ok": distance_make_ok,
        }

        self.waiting_net_check = False
        self.net_check_start_frame = None
        self.net_motion_max = 0.0

        return result_info
