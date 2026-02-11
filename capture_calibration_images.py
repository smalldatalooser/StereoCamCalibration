#!/usr/bin/env python3
"""
capture_calibration_images.py
=============================
Interactive stereo chessboard capture for camera calibration.

Uses cv2.findChessboardCornersSB for robust corner detection.
Auto-saves when both cameras detect the board, with a cooldown between saves.

Controls
--------
  q  – Quit
"""

import os
import sys
import threading
import time
import cv2
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
CAM_LEFT_ID = 2          # /dev/video2
CAM_RIGHT_ID = 4         # /dev/video4
RESOLUTION = (1920, 1080)
FPS = 30

# Chessboard inner corners  (columns × rows)
CHESS_COLS = 7
CHESS_ROWS = 4
BOARD_SIZE = (CHESS_COLS, CHESS_ROWS)

# Output directories
OUT_DIR = "calibration_images"
LEFT_DIR = os.path.join(OUT_DIR, "left")
RIGHT_DIR = os.path.join(OUT_DIR, "right")

PREVIEW_SCALE = 0.5
COOLDOWN = 3.0  # seconds between auto-captures


def open_camera(dev_id: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open /dev/video{dev_id}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, FPS)
    return cap


class Detector(threading.Thread):
    """Runs findChessboardCornersSB in a background thread."""

    def __init__(self, board_size):
        super().__init__(daemon=True)
        self.board_size = board_size
        self._lock = threading.Lock()
        self._gray_l = None
        self._gray_r = None
        self._frame_l = None
        self._frame_r = None
        self._new = False
        self.found_l = False
        self.found_r = False
        self.corners_l = None
        self.corners_r = None
        self.frame_l = None
        self.frame_r = None
        self._running = True
        self.start()

    def submit(self, gray_l, gray_r, frame_l, frame_r):
        with self._lock:
            self._gray_l = gray_l
            self._gray_r = gray_r
            self._frame_l = frame_l.copy()
            self._frame_r = frame_r.copy()
            self._new = True

    def run(self):
        while self._running:
            gl = gr = fl_frame = fr_frame = None
            with self._lock:
                if self._new:
                    gl, gr = self._gray_l.copy(), self._gray_r.copy()
                    fl_frame = self._frame_l.copy()
                    fr_frame = self._frame_r.copy()
                    self._new = False
            if gl is None:
                time.sleep(0.005)
                continue

            # Full-res detection — same as calibration script
            fl, cl = cv2.findChessboardCornersSB(gl, self.board_size)
            fr, cr = cv2.findChessboardCornersSB(gr, self.board_size)

            with self._lock:
                self.found_l, self.corners_l = fl, cl
                self.found_r, self.corners_r = fr, cr
                self.frame_l = fl_frame
                self.frame_r = fr_frame

    def stop(self):
        self._running = False


def main() -> None:
    os.makedirs(LEFT_DIR, exist_ok=True)
    os.makedirs(RIGHT_DIR, exist_ok=True)

    cap_l = open_camera(CAM_LEFT_ID)
    cap_r = open_camera(CAM_RIGHT_ID)

    pair_count = 0
    existing = [f for f in os.listdir(LEFT_DIR) if f.endswith(".png")]
    if existing:
        pair_count = max(int(f.split("_")[1].split(".")[0]) for f in existing)
        print(f"[INFO] Continuing from pair {pair_count + 1}.")

    det = Detector(BOARD_SIZE)

    print("=" * 60)
    print("  Stereo Chessboard Capture")
    print(f"  Board: {CHESS_COLS}×{CHESS_ROWS} inner corners")
    print(f"  Auto-saves every {COOLDOWN:.0f}s when both cameras detect.")
    print("  Move board to different angles/positions between saves!")
    print("  [q] quit")
    print("=" * 60)

    last_save_time = 0

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            continue

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        det.submit(gray_l, gray_r, frame_l, frame_r)

        with det._lock:
            fl, cl = det.found_l, det.corners_l
            fr, cr = det.found_r, det.corners_r

        # Auto-save when both detected
        if fl and fr and (time.time() - last_save_time) > COOLDOWN:
            with det._lock:
                save_l = det.frame_l.copy()
                save_r = det.frame_r.copy()
            pair_count += 1
            fname = f"img_{pair_count:02d}.png"
            cv2.imwrite(os.path.join(LEFT_DIR, fname), save_l)
            cv2.imwrite(os.path.join(RIGHT_DIR, fname), save_r)
            print(f"  ✓ Auto-saved pair #{pair_count}: {fname}")
            last_save_time = time.time()

        # Draw
        disp_l, disp_r = frame_l.copy(), frame_r.copy()
        if fl and cl is not None:
            cv2.drawChessboardCorners(disp_l, BOARD_SIZE, cl, fl)
        if fr and cr is not None:
            cv2.drawChessboardCorners(disp_r, BOARD_SIZE, cr, fr)

        cv2.putText(disp_l, "L: OK" if fl else "L: --", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0) if fl else (0,0,255), 3)
        cv2.putText(disp_r, "R: OK" if fr else "R: --", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0) if fr else (0,0,255), 3)
        cv2.putText(disp_l, f"Pairs: {pair_count}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)

        h, w = disp_l.shape[:2]
        nw, nh = int(w * PREVIEW_SCALE), int(h * PREVIEW_SCALE)
        combined = np.hstack([cv2.resize(disp_l, (nw, nh)),
                              cv2.resize(disp_r, (nw, nh))])
        cv2.imshow("Stereo Capture  [q]=quit", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    det.stop()
    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {pair_count} pairs in '{OUT_DIR}/'.")


if __name__ == "__main__":
    main()
