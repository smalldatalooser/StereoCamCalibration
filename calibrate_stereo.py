#!/usr/bin/env python3
"""
calibrate_stereo.py
===================
Straightforward stereo calibration using OpenCV.

Pipeline:
  1. Load left/right image pairs from calibration_images/
  2. Detect chessboard corners in each pair (both must detect to keep)
  3. Calibrate each camera individually (→ K, D)
  4. Stereo calibrate with FIXED intrinsics (→ R, T)
  5. Rectify (→ R1, R2, P1, P2, Q)
  6. Compute undistortion/rectification maps
  7. Save everything to calibration_data/stereo_params.npz
  8. Show rectified sample with epipolar lines for visual check

Reads from:  calibration_images/left/  and  calibration_images/right/
Writes to:   calibration_data/stereo_params.npz
"""

import os
import sys
import glob
import cv2
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
CHESS_COLS = 7          # Inner corners horizontally
CHESS_ROWS = 4          # Inner corners vertically
BOARD_SIZE = (CHESS_COLS, CHESS_ROWS)
SQUARE_SIZE_MM = 50.0   # Real-world size of one chessboard square in mm

LEFT_DIR  = "calibration_images/left"
RIGHT_DIR = "calibration_images/right"
OUT_DIR   = "calibration_data"
OUT_FILE  = os.path.join(OUT_DIR, "stereo_params.npz")

# Sub-pixel refinement criteria
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load Image Pairs ──────────────────────────────────────────────────
    left_paths  = sorted(glob.glob(os.path.join(LEFT_DIR, "*.png")))
    right_paths = sorted(glob.glob(os.path.join(RIGHT_DIR, "*.png")))

    if len(left_paths) != len(right_paths):
        print(f"[ERROR] Mismatched count: {len(left_paths)} left vs {len(right_paths)} right")
        sys.exit(1)
    if len(left_paths) < 5:
        print(f"[ERROR] Need at least 5 image pairs, found {len(left_paths)}")
        sys.exit(1)

    print("=" * 60)
    print("  Stereo Camera Calibration")
    print(f"  Board: {CHESS_COLS}×{CHESS_ROWS}, Square: {SQUARE_SIZE_MM} mm")
    print(f"  Image pairs found: {len(left_paths)}")
    print("=" * 60)

    # ── 2. Detect Chessboard Corners ─────────────────────────────────────────
    objp = np.zeros((CHESS_ROWS * CHESS_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESS_COLS, 0:CHESS_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_points = []
    img_points_l = []
    img_points_r = []
    used_pairs = []
    image_size = None

    print("\n[Step 1] Detecting chessboard corners...")
    for i, (lp, rp) in enumerate(zip(left_paths, right_paths)):
        img_l = cv2.imread(lp)
        img_r = cv2.imread(rp)
        if img_l is None or img_r is None:
            print(f"  [SKIP] Cannot read pair {i+1}")
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray_l.shape[1], gray_l.shape[0])

        found_l, corners_l = cv2.findChessboardCornersSB(gray_l, BOARD_SIZE)
        found_r, corners_r = cv2.findChessboardCornersSB(gray_r, BOARD_SIZE)

        if not found_l or not found_r:
            side = "LEFT" if not found_l else "RIGHT"
            if not found_l and not found_r:
                side = "BOTH"
            print(f"  [SKIP] Pair {i+1:02d} — no chessboard in {side}")
            continue

        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), SUBPIX_CRITERIA)

        obj_points.append(objp)
        img_points_l.append(corners_l)
        img_points_r.append(corners_r)
        used_pairs.append(os.path.basename(lp))
        print(f"  [OK]   Pair {i+1:02d} — {os.path.basename(lp)}")

    n_valid = len(obj_points)
    print(f"\n  Valid pairs: {n_valid} / {len(left_paths)}")
    if n_valid < 5:
        print("[ERROR] Need at least 5 valid pairs for calibration.")
        sys.exit(1)

    # ── 3. Individual Camera Calibration ─────────────────────────────────────
    print("\n[Step 2] Calibrating individual cameras...")

    ret_l, K1, D1, rvecs_l, tvecs_l = cv2.calibrateCamera(
        obj_points, img_points_l, image_size, None, None
    )
    ret_r, K2, D2, rvecs_r, tvecs_r = cv2.calibrateCamera(
        obj_points, img_points_r, image_size, None, None
    )

    print(f"  Left camera  RMS error: {ret_l:.4f} px")
    print(f"  Right camera RMS error: {ret_r:.4f} px")

    # Per-image reprojection errors
    print("\n  Per-image reprojection errors:")
    print(f"  {'Pair':>12s}   {'Left':>8s}   {'Right':>8s}")
    for i in range(n_valid):
        proj_l, _ = cv2.projectPoints(obj_points[i], rvecs_l[i], tvecs_l[i], K1, D1)
        err_l = cv2.norm(img_points_l[i], proj_l, cv2.NORM_L2) / np.sqrt(len(obj_points[i]))
        proj_r, _ = cv2.projectPoints(obj_points[i], rvecs_r[i], tvecs_r[i], K2, D2)
        err_r = cv2.norm(img_points_r[i], proj_r, cv2.NORM_L2) / np.sqrt(len(obj_points[i]))
        print(f"  {used_pairs[i]:>12s}   {err_l:.4f}   {err_r:.4f}")

    # Info: check if focal lengths are similar (same camera model → similar expected)
    fl_ratio = K1[0, 0] / K2[0, 0]
    print(f"\n  [NOTE] Focal lengths differ by {abs(fl_ratio-1)*100:.0f}%")
    print(f"         Left:  fx={K1[0,0]:.1f}  fy={K1[1,1]:.1f}")
    print(f"         Right: fx={K2[0,0]:.1f}  fy={K2[1,1]:.1f}")
    print(f"         If same camera model, check FOV mode settings.")

    # ── 4. Stereo Calibration ────────────────────────────────────────────────
    print("\n[Step 3] Stereo calibration (fixed intrinsics)...")

    ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        K1, D1, K2, D2, image_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    baseline_mm = np.linalg.norm(T)
    print(f"  Stereo RMS error: {ret_stereo:.4f} px")
    print(f"  Baseline: {baseline_mm:.1f} mm ({baseline_mm/10:.1f} cm)")

    if ret_stereo > 1.0:
        print(f"\n  *** PROBLEM: Stereo error {ret_stereo:.2f}px is too high (want < 1.0) ***")
        print(f"      The calibration data is unreliable. Recapture with:")
        print(f"      - Both cameras in the same FOV mode")
        print(f"      - Board clearly visible and stationary in BOTH views")
        print(f"      - Diverse angles and positions")

    # ── 5. Stereo Rectification ──────────────────────────────────────────────
    print("\n[Step 4] Computing rectification...")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    print(f"  ROI left:  {roi1}")
    print(f"  ROI right: {roi2}")

    # ── 6. Undistortion + Rectification Maps ─────────────────────────────────
    map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)

    # ── 7. Save ──────────────────────────────────────────────────────────────
    print(f"\n[Step 5] Saving to '{OUT_FILE}'...")
    np.savez(
        OUT_FILE,
        K1=K1, D1=D1, K2=K2, D2=D2,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        roi1=roi1, roi2=roi2,
        map1_l=map1_l, map2_l=map2_l,
        map1_r=map1_r, map2_r=map2_r,
        image_size=np.array(image_size),
        reprojection_error_left=ret_l,
        reprojection_error_right=ret_r,
        reprojection_error_stereo=ret_stereo,
    )
    print("  ✓ Saved.")

    # ── 8. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"  Image pairs used:        {n_valid}")
    print(f"  Image size:              {image_size[0]}×{image_size[1]}")
    print(f"  LEFT  reproj. error:     {ret_l:.4f} px")
    print(f"  RIGHT reproj. error:     {ret_r:.4f} px")
    print(f"  STEREO reproj. error:    {ret_stereo:.4f} px")
    print(f"  Baseline:                {baseline_mm:.1f} mm ({baseline_mm/10:.1f} cm)")
    print()
    print("  Left K:")
    print(f"    fx={K1[0,0]:.1f}  fy={K1[1,1]:.1f}  cx={K1[0,2]:.1f}  cy={K1[1,2]:.1f}")
    print("  Right K:")
    print(f"    fx={K2[0,0]:.1f}  fy={K2[1,1]:.1f}  cx={K2[0,2]:.1f}  cy={K2[1,2]:.1f}")
    print("=" * 60)

    # ── 9. Show Rectified Sample ─────────────────────────────────────────────
    mid_name = used_pairs[len(used_pairs) // 2]
    lp = os.path.join(LEFT_DIR, mid_name)
    rp = os.path.join(RIGHT_DIR, mid_name)

    img_l = cv2.imread(lp)
    img_r = cv2.imread(rp)
    rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

    combined = np.hstack([rect_l, rect_r])
    for y in range(0, combined.shape[0], 40):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

    # Save to disk (always works, regardless of display issues)
    os.makedirs("debug", exist_ok=True)
    cv2.imwrite("debug/rectified_pair.png", combined)
    print(f"\n  Rectified pair saved to: debug/rectified_pair.png")

    # Show on screen (WINDOW_NORMAL + explicit resize to work around Wayland/Qt bugs)
    scale = 0.4
    disp = cv2.resize(combined, None, fx=scale, fy=scale)
    win_name = "Rectified Stereo Pair"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, disp.shape[1], disp.shape[0])
    cv2.imshow(win_name, disp)
    print("[INFO] Green lines should run straight across both images.")
    print("       If the window is black, open debug/rectified_pair.png instead.")
    print("       Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
