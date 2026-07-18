"""
Ball position tracking across frames: nearest-neighbour matching against a
constant-velocity prediction (with a wider matching radius near the rim,
where occlusion and fast motion are more common), missed-frame aging, and
spawning of new tracks for detections that don't match an existing one.

This module takes a plain list of per-frame ball detections (box,
confidence, centre) as input and has no dependency on how those detections
were produced. The detection model itself, along with the shot-attempt
detection and possession/player-attribution logic, lives in separate
private modules and is not part of this file. The make/miss outcome logic
is published separately in make_miss_classifier.py.
"""

import math
from collections import deque

from config import (
    NEAR_RIM_DIST, BALL_HISTORY_LEN,
    BALL_MAX_MATCH_DIST_NORMAL, BALL_MAX_MATCH_DIST_NEAR_RIM,
    BALL_MAX_MISSED_NORMAL, BALL_MAX_MISSED_NEAR_RIM,
    BALL_DEDUP_MIN_DIST,
)


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class BallTrack:
    def __init__(self, track_id, box, conf, frame_idx):
        self.track_id = track_id
        self.box = box[:]
        self.center = bbox_center(box)
        self.prev_center = self.center
        self.conf = conf

        self.missed = 0
        self.active = True

        self.history = deque(maxlen=BALL_HISTORY_LEN)
        self.history.append(self.center)

        self.created_frame = frame_idx

    def is_near_rim(self, rim_center):
        return euclidean(self.center, rim_center) <= NEAR_RIM_DIST

    def update(self, box, conf):
        self.prev_center = self.center
        self.box = box[:]
        self.center = bbox_center(box)
        self.conf = conf
        self.missed = 0
        self.active = True
        self.history.append(self.center)

    def mark_missed(self, rim_center):
        self.missed += 1
        limit = BALL_MAX_MISSED_NEAR_RIM if self.is_near_rim(rim_center) else BALL_MAX_MISSED_NORMAL
        if self.missed > limit:
            self.active = False

    def can_keep(self, rim_center):
        limit = BALL_MAX_MISSED_NEAR_RIM if self.is_near_rim(rim_center) else BALL_MAX_MISSED_NORMAL
        return self.missed <= limit


def update_ball_tracks(ball_tracks, detections, frame_idx, rim_center):
    """
    detections: list of {"box": [x1, y1, x2, y2], "conf": float,
    "center": (x, y)} for the current frame, already confidence-filtered
    upstream.
    """
    assigned_track_ids = set()
    assigned_det_ids = set()
    candidates = []

    active_tracks = [bt for bt in ball_tracks if bt.can_keep(rim_center)]

    for bt in active_tracks:
        prev = bt.center
        prev_prev = bt.prev_center

        vx = prev[0] - prev_prev[0]
        vy = prev[1] - prev_prev[1]
        pred = (prev[0] + vx, prev[1] + vy)

        near_rim = bt.is_near_rim(rim_center)
        match_dist = BALL_MAX_MATCH_DIST_NEAR_RIM if near_rim else BALL_MAX_MATCH_DIST_NORMAL

        for di, det in enumerate(detections):
            dc = det["center"]

            d_prev = euclidean(prev, dc)
            d_pred = euclidean(pred, dc)

            if d_prev > match_dist and d_pred > match_dist:
                continue

            score = 0.6 * d_pred + 0.4 * d_prev
            score += 8.0 * (1.0 - det["conf"])

            candidates.append((score, bt, di))

    candidates.sort(key=lambda x: x[0])

    for score, bt, di in candidates:
        if bt.track_id in assigned_track_ids:
            continue
        if di in assigned_det_ids:
            continue

        bt.update(detections[di]["box"], detections[di]["conf"])
        assigned_track_ids.add(bt.track_id)
        assigned_det_ids.add(di)

    for bt in active_tracks:
        if bt.track_id not in assigned_track_ids:
            bt.mark_missed(rim_center)

    next_id = 1 if len(ball_tracks) == 0 else max(bt.track_id for bt in ball_tracks) + 1

    for di, det in enumerate(detections):
        if di in assigned_det_ids:
            continue

        too_close = False
        for bt in ball_tracks:
            if not bt.can_keep(rim_center):
                continue

            if euclidean(det["center"], bt.center) < BALL_DEDUP_MIN_DIST:
                too_close = True
                break

        if too_close:
            continue

        ball_tracks.append(BallTrack(next_id, det["box"], det["conf"], frame_idx))
        next_id += 1

    ball_tracks = [bt for bt in ball_tracks if bt.can_keep(rim_center)]
    return ball_tracks
