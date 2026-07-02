"""
Camera-to-court coordinate mapping via a 4-point planar homography.

Given four image-space reference points (two free-throw-lane elbows plus the
two baseline sideline corners, the latter derived from three user-drawn
lines) and their known real-world court coordinates, this module solves for
the homography that maps any pixel in the camera view onto court-space
metres. It only covers geometry: the interactive point/line collection UI,
the homography solve, and the point transform. It does not include the
broader video pipeline (frame capture, detection model inference, tracking)
that supplies frames to this calibration step in the private system.
"""

import cv2
import numpy as np

from config import (
    MAX_DISPLAY_W, MAX_DISPLAY_H,
    POINT_CLICK_NAMES, LINE_CLICK_NAMES, COURT_PTS_4,
)


def fit_to_screen(frame_bgr, max_w=MAX_DISPLAY_W, max_h=MAX_DISPLAY_H):
    h, w = frame_bgr.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        disp = cv2.resize(frame_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    else:
        disp = frame_bgr.copy()
    return disp, s


def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(den) < 1e-9:
        raise RuntimeError("Selected lines are nearly parallel. Please recalibrate.")

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den

    return float(px), float(py)


def line_points_to_border_points(p1, p2, w, h):
    x1, y1 = p1
    x2, y2 = p2

    candidates = []

    if abs(x2 - x1) > 1e-9:
        t = (0 - x1) / (x2 - x1)
        y = y1 + t * (y2 - y1)
        if 0 <= y <= h - 1:
            candidates.append((0, y))

        t = ((w - 1) - x1) / (x2 - x1)
        y = y1 + t * (y2 - y1)
        if 0 <= y <= h - 1:
            candidates.append((w - 1, y))

    if abs(y2 - y1) > 1e-9:
        t = (0 - y1) / (y2 - y1)
        x = x1 + t * (x2 - x1)
        if 0 <= x <= w - 1:
            candidates.append((x, 0))

        t = ((h - 1) - y1) / (y2 - y1)
        x = x1 + t * (x2 - x1)
        if 0 <= x <= w - 1:
            candidates.append((x, h - 1))

    uniq = []
    for pt in candidates:
        if not any(abs(pt[0] - q[0]) < 1e-6 and abs(pt[1] - q[1]) < 1e-6 for q in uniq):
            uniq.append(pt)

    if len(uniq) < 2:
        return p1, p2

    best_pair = (uniq[0], uniq[1])
    best_d = -1

    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            d = (uniq[i][0] - uniq[j][0]) ** 2 + (uniq[i][1] - uniq[j][1]) ** 2
            if d > best_d:
                best_d = d
                best_pair = (uniq[i], uniq[j])

    return best_pair


_clicked_points = []
_clicked_lines = []
_calib_scale = 1.0
_current_line_points = []


def _mouse_cb_calib(event, x, y, flags, param):
    global _clicked_points, _clicked_lines, _calib_scale, _current_line_points

    if event == cv2.EVENT_LBUTTONDOWN:
        ox = float(x) / _calib_scale
        oy = float(y) / _calib_scale

        if len(_clicked_points) < 2:
            _clicked_points.append((ox, oy))
            return

        _current_line_points.append((ox, oy))

        if len(_current_line_points) == 2:
            _clicked_lines.append((_current_line_points[0], _current_line_points[1]))
            _current_line_points = []


def calibrate_homography(first_frame_bgr):
    global _clicked_points, _clicked_lines, _calib_scale, _current_line_points

    _clicked_points = []
    _clicked_lines = []
    _current_line_points = []

    win = "COURT HOMOGRAPHY CALIBRATION"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _mouse_cb_calib)

    while True:
        disp, s = fit_to_screen(first_frame_bgr)
        _calib_scale = s
        vis = disp.copy()

        h0, w0 = first_frame_bgr.shape[:2]

        if len(_clicked_points) < 2:
            msg = f"Click point: {POINT_CLICK_NAMES[len(_clicked_points)]} ({len(_clicked_points)}/2)"
        elif len(_clicked_lines) < 3:
            line_idx = len(_clicked_lines)
            if len(_current_line_points) == 0:
                msg = f"Draw line: {LINE_CLICK_NAMES[line_idx]} | click point 1"
            else:
                msg = f"Draw line: {LINE_CLICK_NAMES[line_idx]} | click point 2"
        else:
            msg = "ENTER=confirm | R=reset"

        cv2.putText(vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        for i, (cx, cy) in enumerate(_clicked_points):
            dx = int(cx * _calib_scale)
            dy = int(cy * _calib_scale)
            cv2.circle(vis, (dx, dy), 6, (0, 255, 255), -1)
            cv2.putText(vis, f"P{i+1}", (dx + 8, dy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        for i, (p1, p2) in enumerate(_clicked_lines):
            q1, q2 = line_points_to_border_points(p1, p2, w0, h0)
            a = (int(q1[0] * _calib_scale), int(q1[1] * _calib_scale))
            b = (int(q2[0] * _calib_scale), int(q2[1] * _calib_scale))
            cv2.line(vis, a, b, (0, 200, 255), 2)
            cv2.putText(vis, f"L{i+1}", (int((a[0] + b[0]) / 2), int((a[1] + b[1]) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        for i, (cx, cy) in enumerate(_current_line_points):
            dx = int(cx * _calib_scale)
            dy = int(cy * _calib_scale)
            cv2.circle(vis, (dx, dy), 5, (255, 180, 0), -1)
            cv2.putText(vis, f"{i+1}", (dx + 8, dy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

        if len(_clicked_lines) == 3:
            try:
                left_corner_img = line_intersection(
                    _clicked_lines[0][0], _clicked_lines[0][1],
                    _clicked_lines[2][0], _clicked_lines[2][1]
                )
                right_corner_img = line_intersection(
                    _clicked_lines[1][0], _clicked_lines[1][1],
                    _clicked_lines[2][0], _clicked_lines[2][1]
                )

                for name, pt in [("LC", left_corner_img), ("RC", right_corner_img)]:
                    dx = int(pt[0] * _calib_scale)
                    dy = int(pt[1] * _calib_scale)
                    cv2.circle(vis, (dx, dy), 7, (0, 0, 255), -1)
                    cv2.putText(vis, name, (dx + 8, dy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception:
                pass

        cv2.imshow(win, vis)
        k = cv2.waitKey(10) & 0xFF

        if k in (ord("r"), ord("R")):
            _clicked_points = []
            _clicked_lines = []
            _current_line_points = []
        elif k in (13, 10):
            if len(_clicked_points) == 2 and len(_clicked_lines) == 3 and len(_current_line_points) == 0:
                break
        elif k == 27:
            cv2.destroyWindow(win)
            raise RuntimeError("Calibration cancelled.")

    cv2.destroyWindow(win)

    left_corner_img = line_intersection(
        _clicked_lines[0][0], _clicked_lines[0][1],
        _clicked_lines[2][0], _clicked_lines[2][1]
    )
    right_corner_img = line_intersection(
        _clicked_lines[1][0], _clicked_lines[1][1],
        _clicked_lines[2][0], _clicked_lines[2][1]
    )

    img_pts_4 = np.array([
        _clicked_points[0],
        _clicked_points[1],
        left_corner_img,
        right_corner_img,
    ], dtype=np.float32)

    H_img2court, _ = cv2.findHomography(img_pts_4, COURT_PTS_4, method=0)

    if H_img2court is None:
        raise RuntimeError("Homography failed.")

    return H_img2court


def img_to_court(pt_xy, H_img2court):
    arr = np.array([[pt_xy]], dtype=np.float32)
    out = cv2.perspectiveTransform(arr, H_img2court).reshape(2,)
    return float(out[0]), float(out[1])
