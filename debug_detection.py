#!/usr/bin/env python3
"""Debug chessboard detection: capture one frame, try multiple board sizes,
save annotated images so we can see exactly what's happening."""

import cv2
import numpy as np
import os

CAM_LEFT_ID = 2
CAM_RIGHT_ID = 4

# Board sizes to try (inner corners)
SIZES_TO_TRY = [
    (5, 7), (7, 5),
    (6, 8), (8, 6),
    (6, 7), (7, 6),
    (5, 8), (8, 5),
    (5, 6), (6, 5),
    (9, 6), (6, 9),
    (7, 7),
    (4, 6), (6, 4),
    (4, 7), (7, 4),
]

os.makedirs("debug", exist_ok=True)

# Capture one frame from each camera
cap_l = cv2.VideoCapture(CAM_LEFT_ID, cv2.CAP_V4L2)
cap_r = cv2.VideoCapture(CAM_RIGHT_ID, cv2.CAP_V4L2)

for cap in [cap_l, cap_r]:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

# Grab a few frames to let auto-exposure settle
for _ in range(10):
    cap_l.read()
    cap_r.read()

ret_l, frame_l = cap_l.read()
ret_r, frame_r = cap_r.read()
cap_l.release()
cap_r.release()

if not ret_l or not ret_r:
    print("Failed to capture frames!")
    exit(1)

# Save raw frames
cv2.imwrite("debug/raw_left.png", frame_l)
cv2.imwrite("debug/raw_right.png", frame_r)
print(f"Saved raw frames: debug/raw_left.png, debug/raw_right.png")
print(f"Frame size: {frame_l.shape[1]}x{frame_l.shape[0]}")

gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

print(f"\n{'Size':>8}  {'Left':>8}  {'Right':>8}")
print("-" * 30)

for cols, rows in SIZES_TO_TRY:
    found_l, corners_l = cv2.findChessboardCornersSB(gray_l, (cols, rows))
    found_r, corners_r = cv2.findChessboardCornersSB(gray_r, (cols, rows))

    status_l = "✓ FOUND" if found_l else "  --"
    status_r = "✓ FOUND" if found_r else "  --"
    print(f"  {cols}×{rows}    {status_l}    {status_r}")

    # Save annotated image for any successful detection
    if found_l:
        img = frame_l.copy()
        cv2.drawChessboardCorners(img, (cols, rows), corners_l, found_l)
        cv2.putText(img, f"FOUND: {cols}x{rows}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imwrite(f"debug/left_{cols}x{rows}_found.png", img)

    if found_r:
        img = frame_r.copy()
        cv2.drawChessboardCorners(img, (cols, rows), corners_r, found_r)
        cv2.putText(img, f"FOUND: {cols}x{rows}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imwrite(f"debug/right_{cols}x{rows}_found.png", img)

print("\nDone. Check debug/ folder for annotated images.")
