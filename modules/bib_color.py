"""
Jersey/bib colour detection and matching, HSV-based.

Builds an HSV colour reference from a manually selected bib region (a
saturation/value mask plus a circular hue mean and hue histogram) and scores
how well a new region of interest matches that reference.

This is a standalone signal used by a private multi-player tracking system.
The tracking/re-tracking state machine that consumes these scores
(predicting positions across missed detections, promoting/demoting tracks,
cross-frame ID assignment) is not included here — only the colour
detection/matching functions themselves.
"""

import cv2
import numpy as np

from config import MIN_SAT, MIN_VAL


def circular_hue_mean(h_values):
    if h_values.size == 0:
        return None

    angles = h_values.astype(np.float32) * (2.0 * np.pi / 180.0)
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))

    angle = np.arctan2(mean_sin, mean_cos)
    if angle < 0:
        angle += 2.0 * np.pi

    return float(angle * 180.0 / (2.0 * np.pi))


def hue_distance_score(hue, ref_hue, good_tol=10, bad_tol=28):
    if hue is None or ref_hue is None:
        return 0.0

    diff = abs(float(hue) - float(ref_hue))
    diff = min(diff, 180.0 - diff)

    if diff <= good_tol:
        return 1.0
    if diff >= bad_tol:
        return 0.0

    score = 1.0 - ((diff - good_tol) / (bad_tol - good_tol)) ** 2
    return max(0.0, min(1.0, score))


def get_color_mask(hsv):
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    mask = (s >= MIN_SAT) & (v >= MIN_VAL)
    mask_u8 = (mask.astype(np.uint8) * 255)

    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    return mask_u8


def compute_hsv_reference(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask_u8 = get_color_mask(hsv)
    valid_mask = mask_u8 > 0

    h = hsv[:, :, 0]
    valid_h = h[valid_mask]

    if valid_h.size < 25:
        valid_mask = np.ones(h.shape, dtype=bool)
        mask_u8 = np.full(h.shape, 255, dtype=np.uint8)
        valid_h = h[valid_mask]

    ref_hue = circular_hue_mean(valid_h)
    if ref_hue is None:
        return None

    hist = cv2.calcHist([hsv], [0], mask_u8, [180], [0, 180])
    hist = cv2.normalize(hist, hist).flatten()

    valid_ratio = float(np.count_nonzero(valid_mask) / valid_mask.size)

    return {
        "ref_hue": ref_hue,
        "hist": hist,
        "valid_ratio": valid_ratio
    }


def compute_color_score(roi_bgr, ref_info):
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask_u8 = get_color_mask(hsv)
    valid_mask = mask_u8 > 0
    valid_ratio = float(np.count_nonzero(valid_mask) / valid_mask.size)

    if np.count_nonzero(valid_mask) < 20:
        return 0.0, 0.0, 0.0, valid_ratio

    h = hsv[:, :, 0]

    hist = cv2.calcHist([hsv], [0], mask_u8, [180], [0, 180])
    hist = cv2.normalize(hist, hist).flatten()

    hist_corr = cv2.compareHist(
        ref_info["hist"].astype(np.float32),
        hist.astype(np.float32),
        cv2.HISTCMP_CORREL
    )

    hist_score = float((hist_corr + 1.0) / 2.0)
    hist_score = max(0.0, min(1.0, hist_score))

    valid_h = h[valid_mask]

    hue_scores = [
        hue_distance_score(hv, ref_info["ref_hue"], good_tol=10, bad_tol=28)
        for hv in valid_h
    ]

    hue_score = float(np.mean(hue_scores)) if len(hue_scores) > 0 else 0.0
    valid_score = min(1.0, valid_ratio * 2.0)

    combined = 0.20 * hist_score + 0.70 * hue_score + 0.10 * valid_score
    combined = max(0.0, min(1.0, combined))

    return combined, hist_score, hue_score, valid_ratio
