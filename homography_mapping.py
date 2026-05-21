"""
homography_mapping.py
=====================
Court homography computation and image-to-court coordinate mapping.

How homography works
--------------------
A homography is a 3×3 matrix H that encodes a perspective transform between
two planes — here, the camera image plane and the flat court surface. It is
computed from four corresponding point pairs: image pixels whose real-world
court positions (in metres) are known in advance.

Once H is computed, any image pixel (x, y) can be mapped to court metres via
a homogeneous multiply followed by a perspective divide:

    [X, Y, w]^T = H · [x, y, 1]^T
    court_xy = (X / w,  Y / w)

cv2.perspectiveTransform handles the divide automatically.

Calibration anchors
-------------------
Four court landmarks with fixed real-world positions are used as anchors:

    Anchor                  Court position (metres)
    ----------------------  -----------------------
    Left FT-lane elbow      (-2.45,  4.225)
    Right FT-lane elbow     ( 2.45,  4.225)
    Left baseline corner    (-7.5,  -1.575)
    Right baseline corner   ( 7.5,  -1.575)

Coordinate system: origin at the hoop centre, X increases left→right,
Y increases toward half-court.

The user clicks these four points in the video frame once per session;
their image-pixel coordinates are passed to compute_homography() to produce H.
All subsequent lookups (e.g. shooter position at shot release) call
img_to_court() with the same H.

Usage
-----
    img_pts = np.array([
        [341, 512],   # left FT elbow  (image pixels, clicked by user)
        [698, 511],   # right FT elbow
        [ 87, 731],   # left baseline corner
        [952, 729],   # right baseline corner
    ], dtype=np.float32)

    H = compute_homography(img_pts)
    court_xy = img_to_court((520, 640), H)
    # → e.g. (0.31, 1.18) metres from the hoop
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Real-world court anchor positions (metres, hoop at origin)
# ---------------------------------------------------------------------------

_LANE_HALF  = 4.9 / 2.0        # 2.45 m — half the paint width
_Y_FT_LINE  = -1.575 + 5.8     # 4.225 m — FT line distance from hoop
_HALF_W     = 15.0 / 2.0       # 7.5 m  — half the court width
_Y_BASELINE = -1.575           # metres — baseline distance from hoop

COURT_PTS_4 = np.array([
    [-_LANE_HALF, _Y_FT_LINE],   # left FT-lane elbow
    [ _LANE_HALF, _Y_FT_LINE],   # right FT-lane elbow
    [-_HALF_W,    _Y_BASELINE],  # left baseline corner
    [ _HALF_W,    _Y_BASELINE],  # right baseline corner
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Homography computation
# ---------------------------------------------------------------------------

def compute_homography(img_pts: np.ndarray) -> np.ndarray:
    """Compute a 3×3 homography matrix mapping image pixels to court metres.

    Calls cv2.findHomography with an exact 4-point solve (method=0, no RANSAC)
    to find H such that for every anchor i:

        court_pts[i]  ≈  perspective_divide(H · img_pts[i])

    Args:
        img_pts: (4, 2) float32 array of image-pixel coordinates for the four
                 calibration anchors, in this exact order:
                   [0] left FT-lane elbow
                   [1] right FT-lane elbow
                   [2] left baseline corner
                   [3] right baseline corner

    Returns:
        H: (3, 3) float64 homography matrix.

    Raises:
        ValueError:   If img_pts does not have shape (4, 2).
        RuntimeError: If cv2.findHomography fails (collinear or degenerate
                      point configuration).

    Example::

        img_pts = np.array([
            [341, 512],
            [698, 511],
            [ 87, 731],
            [952, 729],
        ], dtype=np.float32)

        H = compute_homography(img_pts)
    """
    img_pts = np.asarray(img_pts, dtype=np.float32)
    if img_pts.shape != (4, 2):
        raise ValueError(
            f"img_pts must have shape (4, 2), got {img_pts.shape}. "
            "Provide exactly four (x, y) image-pixel coordinates."
        )

    H, _ = cv2.findHomography(img_pts, COURT_PTS_4, method=0)

    if H is None:
        raise RuntimeError(
            "cv2.findHomography returned None. Ensure the four anchor points "
            "are not collinear and correspond to the correct court landmarks."
        )

    return H


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------

def img_to_court(pt_xy: tuple, H: np.ndarray) -> tuple:
    """Map a single image-space point to court coordinates in metres.

    Wraps cv2.perspectiveTransform, which applies H and handles the
    homogeneous divide in one call.

    Args:
        pt_xy: (x, y) point in image pixels.
        H:     (3, 3) homography matrix from compute_homography().

    Returns:
        (x_court, y_court) in metres, with the hoop at the origin.

    Example::

        court_xy = img_to_court((520, 640), H)
        # → (0.31, 1.18)
    """
    arr = np.array([[pt_xy]], dtype=np.float32)
    out = cv2.perspectiveTransform(arr, H).reshape(2)
    return float(out[0]), float(out[1])
