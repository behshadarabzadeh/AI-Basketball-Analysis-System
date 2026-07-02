"""
Shared configuration constants for the public module set.

Only constants that are actually referenced by court.py, homography_mapping.py,
make_miss_classifier.py, ball_track.py, and bib_color.py are defined here.
Anything path-, model-, or venue-specific from the private system (video/model
file locations, trained weight files, per-venue calibration tables) has been
left out of this public release on purpose.
"""

import math
import numpy as np

# ---------------------------------------------------------
# Used by: homography_mapping.py (display scaling for the
# interactive calibration window)
# ---------------------------------------------------------
MAX_DISPLAY_W = 1400
MAX_DISPLAY_H = 900

# ---------------------------------------------------------
# Used by: court.py, homography_mapping.py
# Real-world court geometry, in metres, relative to the rim.
# ---------------------------------------------------------
COURT_W = 15.0
HALF_W = COURT_W / 2.0

HOOP_TO_BASELINE = 1.575
HALFCOURT_FROM_BASELINE = 14.0

Y_MIN = -HOOP_TO_BASELINE
Y_MAX = HALFCOURT_FROM_BASELINE - HOOP_TO_BASELINE

LANE_W = 4.9
LANE_HALF = LANE_W / 2.0

FT_FROM_BASELINE = 5.8
Y_FT_LINE = -HOOP_TO_BASELINE + FT_FROM_BASELINE
FT_CIRCLE_D = 3.6

LOWMID_R = 4.30

THREE_ARC_R = 6.75
CORNER_3_X = 6.60
CORNER_3_Y_INT = math.sqrt(max(0.0, THREE_ARC_R**2 - CORNER_3_X**2))

theta_right = math.degrees(math.atan2(CORNER_3_Y_INT, CORNER_3_X))
theta_left = 180.0 - theta_right

TOP_5 = 25.0
WING_5 = 60.0
TOP_3 = 30.0

# ---------------------------------------------------------
# Used by: homography_mapping.py
# The four image-space reference points collected during
# calibration are mapped onto these known court-space
# coordinates (left/right free-throw-lane elbows, and the
# two baseline corners at the sideline).
# ---------------------------------------------------------
POINT_CLICK_NAMES = [
    "LEFT ELBOW (lane corner @ FT line)",
    "RIGHT ELBOW (lane corner @ FT line)",
]

LINE_CLICK_NAMES = [
    "LEFT SIDELINE",
    "RIGHT SIDELINE",
    "BASELINE",
]

COURT_PTS_4 = np.array([
    [-LANE_HALF, Y_FT_LINE],
    [LANE_HALF, Y_FT_LINE],
    [-HALF_W, Y_MIN],
    [HALF_W, Y_MIN],
], dtype=np.float32)

# ---------------------------------------------------------
# Used by: court.py (drawing style + zone iteration order)
# ---------------------------------------------------------
COURT_COLOR = "black"
COURT_LW = 2.8
ZONE_LINE_COLOR = "gray"
ZONE_LW = 1.2

ZONE_ORDER = [
    "LOWMID_LEFT", "LOWMID_CENTER", "LOWMID_RIGHT",
    "HIGHMID_LEFT_CORNER", "HIGHMID_LEFT_WING", "HIGHMID_TOP",
    "HIGHMID_RIGHT_WING", "HIGHMID_RIGHT_CORNER",
    "3PT_LEFT_CORNER", "3PT_LEFT_WING", "3PT_TOP",
    "3PT_RIGHT_WING", "3PT_RIGHT_CORNER",
]

ZONE_FILL_ORDER = [
    "LOWMID_LEFT", "LOWMID_CENTER", "LOWMID_RIGHT",
    "HIGHMID_LEFT_CORNER", "HIGHMID_LEFT_WING", "HIGHMID_TOP",
    "HIGHMID_RIGHT_WING", "HIGHMID_RIGHT_CORNER",
    "3PT_LEFT_WING", "3PT_TOP", "3PT_RIGHT_WING",
    "3PT_LEFT_CORNER", "3PT_RIGHT_CORNER",
]

# ---------------------------------------------------------
# Used by: ball_track.py
# ---------------------------------------------------------
NEAR_RIM_DIST = 90
BALL_HISTORY_LEN = 20

BALL_MAX_MATCH_DIST_NORMAL = 100
BALL_MAX_MATCH_DIST_NEAR_RIM = 220

BALL_MAX_MISSED_NORMAL = 12
BALL_MAX_MISSED_NEAR_RIM = 12

BALL_DEDUP_MIN_DIST = 80

# ---------------------------------------------------------
# Used by: make_miss_classifier.py
# Net-motion thresholds are recalibrated per install (camera
# distance/angle and net material change the pixel-diff
# response); the values below are a working default, not a
# universal constant.
# ---------------------------------------------------------
RIM_ABOVE_MARGIN_PX = 2.0
RIM_BELOW_MARGIN_PX = 2.0

MAKE_SUM_DIST_MAX_PX = 40.0

USE_NET_MOTION_FOR_MAKE = True
NET_CHECK_AFTER_BELOW_FRAMES = 5
NET_MOTION_THRESHOLD = 0.07
NET_DIFF_THRESHOLD = 25

# ---------------------------------------------------------
# Used by: bib_color.py
# ---------------------------------------------------------
MIN_SAT = 70
MIN_VAL = 60
