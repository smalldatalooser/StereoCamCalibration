#!/usr/bin/env python3
"""
debug_corners.py
================
Visualize detected chessboard corners for each pair.
Marks corner[0] (GREEN) and corner[-1] (RED) in both left and right images.

If corner[0] is at the top-left in the left image but bottom-right in the
right image, the ordering is flipped — this will ruin stereo calibration.

Saves annotated images to debug/corners/
"""

import os
import glob
import cv2
import numpy as np

CHESS_COLS = 7
CHESS_ROWS = 4
BOARD_SIZE = (CHESS_COLS, CHESS_ROWS)

LEFT_DIR  = "calibration_images/left"
RIGHT_DIR = "calibration_images/right"
DEBUG_DIR = "debug/corners"

os.makedirs(DEBUG_DIR, exist_ok=True)

left_paths  = sorted(glob.glob(os.path.join(LEFT_DIR, "*.png")))
right_paths = sorted(glob.glob(os.path.join(RIGHT_DIR, "*.png")))

print(f"{'Pair':>12s}   {'c0_L (x,y)':>16s}   {'c0_R (x,y)':>16s}   {'y_diff':>8s}   Status")
print("-" * 80)

for i, (lp, rp) in enumerate(zip(left_paths, right_paths)):
    img_l = cv2.imread(lp)
    img_r = cv2.imread(rp)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    found_l, corners_l = cv2.findChessboardCornersSB(gray_l, BOARD_SIZE)
    found_r, corners_r = cv2.findChessboardCornersSB(gray_r, BOARD_SIZE)

    name = os.path.basename(lp)

    if not found_l or not found_r:
        print(f"  {name:>12s}   {'---':>16s}   {'---':>16s}   {'---':>8s}   SKIP (not found)")
        continue

    c0_l = corners_l[0][0]
    c0_r = corners_r[0][0]
    cn_l = corners_l[-1][0]
    cn_r = corners_r[-1][0]

    y_diff = abs(c0_l[1] - c0_r[1])

    # Check: if corner[0] in right is closer to corner[-1] in left, it's flipped
    y_diff_flipped = abs(c0_l[1] - cn_r[1])
    flipped = y_diff_flipped < y_diff and y_diff > 100

    status = "FLIPPED!" if flipped else "ok"
    print(f"  {name:>12s}   ({c0_l[0]:7.1f},{c0_l[1]:7.1f})   ({c0_r[0]:7.1f},{c0_r[1]:7.1f})   {y_diff:7.1f}   {status}")

    # Save debug visualization
    vis_l = img_l.copy()
    vis_r = img_r.copy()

    cv2.drawChessboardCorners(vis_l, BOARD_SIZE, corners_l, True)
    cv2.drawChessboardCorners(vis_r, BOARD_SIZE, corners_r, True)

    # Corner[0] = GREEN circle, Corner[-1] = RED circle
    cv2.circle(vis_l, tuple(c0_l.astype(int)), 25, (0, 255, 0), 5)
    cv2.circle(vis_l, tuple(cn_l.astype(int)), 25, (0, 0, 255), 5)
    cv2.putText(vis_l, "0", tuple((c0_l + [30, 0]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.circle(vis_r, tuple(c0_r.astype(int)), 25, (0, 255, 0), 5)
    cv2.circle(vis_r, tuple(cn_r.astype(int)), 25, (0, 0, 255), 5)
    cv2.putText(vis_r, "0", tuple((c0_r + [30, 0]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    combined = np.hstack([vis_l, vis_r])
    scale = 0.35
    combined = cv2.resize(combined, None, fx=scale, fy=scale)
    cv2.imwrite(os.path.join(DEBUG_DIR, name), combined)

print(f"\nDebug images saved to {DEBUG_DIR}/")
print("Check: GREEN circle (corner 0) should be at the SAME physical corner in both L and R.")
