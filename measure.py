#!/usr/bin/env python3
"""
measure.py
==========
Live stereo measurement tool with YOLO26 object detection.

Shows both rectified cameras side by side. YOLO26 detects objects on the left
image, template-matches each bbox corner to the right image, triangulates
the 3D positions, and displays width/height/diagonal measurements.

Also supports manual click-to-measure (click two points on left image).

Controls
--------
  Left-click on LEFT image  – Set a manual measurement point (need exactly 2)
  r                         – Reset manual points
  s                         – Save screenshot
  d                         – Toggle YOLO detection on/off
  q                         – Quit
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────────────────────
CAM_LEFT_ID = 2
CAM_RIGHT_ID = 4
RESOLUTION = (1920, 1080)
FPS = 30

CALIB_FILE = "calibration_data/stereo_params.npz"
YOLO_MODEL = "Models/yolo26l.pt"

# Template matching
TEMPLATE_HALF = 30
SEARCH_MARGIN_Y = 5
MIN_MATCH_SCORE = 0.4

# YOLO
YOLO_CONF = 0.4            # Minimum confidence for detections
YOLO_IMGSZ = 640           # Inference resolution (smaller = faster)

PREVIEW_SCALE = 0.5
SCREENSHOT_DIR = "screenshots"
LABELS2DETECT = ["bottle"]
# available COCO labels: # https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt  
##########################
# 1: person, 2: bicycle, 3: car, 4: motorcycle, 5: airplane, 6: bus, 7: train, 8: truck, 
# 9: boat, 10: traffic light, 11: fire hydrant, 13: stop sign, 14: parking meter, 16: bench, 
# 17: bird, 18: cat, 19: dog, 20: horse, 21: sheep, 22: cow, 23: elephant, 24: bear, 25: zebra, 
# 26: giraffe, 27: backpack, 28: umbrella, 31: handbag, 32: tie, 33: suitcase, 34: frisbee, 
# 35: skis, 36: snowboard, 37: sports ball, 38: kite, 39: baseball bat, 40: baseball glove, 
# 41: skateboard, 42: surfboard, 43: tennis racket, 44: bottle, 46: wine glass, 47: cup, 
# 48: fork, 49: knife, 50: spoon, 51: bowl, 52: banana, 53: apple, 54: sandwich, 55: orange, 
# 56: broccoli, 57: carrot, 58: hot dog, 59: pizza, 60: donut, 61: cake, 62: chair, 63: couch, 
# 64: potted plant, 65: bed, 67: dining table, 70: toilet, 72: TV, 73: laptop, 74: mouse, 
# 75: remote, 76: keyboard, 77: cell phone, 78: microwave, 79: oven, 80: toaster, 81: sink, 
# 82: refrigerator, 84: book, 85: clock, 86: vase, 87: scissors, 88: teddy bear, 89: hair drier, 
# 90: toothbrush
##########################

# ── Load Calibration Data ────────────────────────────────────────────────────
def load_calibration(path):
    """Load stereo calibration parameters. Returns a dict."""
    if not os.path.exists(path):
        print(f"[ERROR] Calibration file not found: {path}")
        print("        Run calibrate_stereo.py first.")
        sys.exit(1)

    data = np.load(path)
    calib = {
        "P1": data["P1"],
        "P2": data["P2"],
        "map1_l": data["map1_l"],
        "map2_l": data["map2_l"],
        "map1_r": data["map1_r"],
        "map2_r": data["map2_r"],
        "image_size": tuple(data["image_size"]),
    }

    stereo_err = float(data.get("reprojection_error_stereo", 0))
    T = data["T"]
    baseline = np.linalg.norm(T)

    print(f"[INFO] Calibration loaded from '{path}'")
    print(f"       Stereo reproj. error: {stereo_err:.4f} px")
    print(f"       Baseline: {baseline:.1f} mm ({baseline/10:.1f} cm)")

    return calib


# ── Camera ───────────────────────────────────────────────────────────────────
def open_camera(dev_id):
    """Open a V4L2 camera with MJPG at the configured resolution."""
    cap = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open /dev/video{dev_id}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, FPS)
    return cap


# ── Core Stereo Functions ────────────────────────────────────────────────────
def rectify(frame_l, frame_r, calib):
    """Apply undistortion + rectification to a stereo pair."""
    rect_l = cv2.remap(frame_l, calib["map1_l"], calib["map2_l"], cv2.INTER_LINEAR)
    rect_r = cv2.remap(frame_r, calib["map1_r"], calib["map2_r"], cv2.INTER_LINEAR)
    return rect_l, rect_r


def find_correspondence(rect_l, rect_r, x_l, y_l):
    """Find matching point in right image using template matching.

    Returns (x_r, y_r, score) or (None, None, 0.0) if no good match.
    """
    h, w = rect_l.shape[:2]
    t = TEMPLATE_HALF

    y_top = max(0, y_l - t)
    y_bot = min(h, y_l + t + 1)
    x_left = max(0, x_l - t)
    x_right = min(w, x_l + t + 1)

    template = rect_l[y_top:y_bot, x_left:x_right]
    if template.size == 0:
        return None, None, 0.0

    search_y_top = max(0, y_l - t - SEARCH_MARGIN_Y)
    search_y_bot = min(h, y_l + t + 1 + SEARCH_MARGIN_Y)
    search_x_right = min(w, x_l + t + 1)

    search_strip = rect_r[search_y_top:search_y_bot, 0:search_x_right]

    if search_strip.shape[0] < template.shape[0] or search_strip.shape[1] < template.shape[1]:
        return None, None, 0.0

    result = cv2.matchTemplate(search_strip, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < MIN_MATCH_SCORE:
        return None, None, max_val

    x_r = max_loc[0] + (x_l - x_left)
    y_r = search_y_top + max_loc[1] + (y_l - y_top)

    return x_r, y_r, max_val


def triangulate_point(P1, P2, pt_l, pt_r):
    """Triangulate a single 3D point from a pair of 2D correspondences.

    Returns np.array([X, Y, Z]) in mm.
    """
    pts_l = np.array([[pt_l[0], pt_l[1]]], dtype=np.float64).T
    pts_r = np.array([[pt_r[0], pt_r[1]]], dtype=np.float64).T

    pts_4d = cv2.triangulatePoints(P1, P2, pts_l, pts_r)
    pts_3d = (pts_4d[:3] / pts_4d[3]).flatten()
    return pts_3d


def measure_point(rect_l, rect_r, x_l, y_l, P1, P2):
    """Find correspondence + triangulate for a single point.

    Returns (p3d, score) or (None, 0.0) on failure.
    """
    x_r, y_r, score = find_correspondence(rect_l, rect_r, x_l, y_l)
    if x_r is None:
        return None, score

    disparity = x_l - x_r
    if disparity <= 0:
        return None, score

    p3d = triangulate_point(P1, P2, (x_l, y_l), (x_r, y_r))
    if p3d[2] <= 0:
        return None, score

    return p3d, score


# ── YOLO Bbox Measurement ───────────────────────────────────────────────────
def measure_bbox(rect_l, rect_r, x1, y1, x2, y2, P1, P2):
    """Measure the real-world dimensions of a bounding box.

    Triangulates the four corners of the bbox and computes:
      - width (horizontal distance between left and right edges)
      - height (vertical distance between top and bottom edges)
      - diagonal (hypotenuse)

    Returns dict with 'width_mm', 'height_mm', 'diag_mm' or None on failure.
    """
    # Use midpoints of each edge for more robust matching (edges have more texture
    # than corners which might be on the object boundary / background)
    mid_top    = (int((x1 + x2) / 2), int(y1))
    mid_bottom = (int((x1 + x2) / 2), int(y2))
    mid_left   = (int(x1), int((y1 + y2) / 2))
    mid_right  = (int(x2), int((y1 + y2) / 2))

    # Also measure corners for diagonal
    top_left     = (int(x1), int(y1))
    bottom_right = (int(x2), int(y2))

    # Triangulate edge midpoints
    p3d_top, _    = measure_point(rect_l, rect_r, mid_top[0], mid_top[1], P1, P2)
    p3d_bottom, _ = measure_point(rect_l, rect_r, mid_bottom[0], mid_bottom[1], P1, P2)
    p3d_left, _   = measure_point(rect_l, rect_r, mid_left[0], mid_left[1], P1, P2)
    p3d_right, _  = measure_point(rect_l, rect_r, mid_right[0], mid_right[1], P1, P2)

    # Triangulate corners for diagonal
    p3d_tl, _ = measure_point(rect_l, rect_r, top_left[0], top_left[1], P1, P2)
    p3d_br, _ = measure_point(rect_l, rect_r, bottom_right[0], bottom_right[1], P1, P2)

    result = {}

    if p3d_left is not None and p3d_right is not None:
        result["width_mm"] = float(np.linalg.norm(p3d_right - p3d_left))
    if p3d_top is not None and p3d_bottom is not None:
        result["height_mm"] = float(np.linalg.norm(p3d_bottom - p3d_top))
    if p3d_tl is not None and p3d_br is not None:
        result["diag_mm"] = float(np.linalg.norm(p3d_br - p3d_tl))

    # Need at least width or height to be useful
    if "width_mm" not in result and "height_mm" not in result:
        return None

    return result


def format_mm(val_mm):
    """Format a distance nicely."""
    if val_mm >= 1000:
        return f"{val_mm/1000:.2f}m"
    elif val_mm >= 100:
        return f"{val_mm/10:.1f}cm"
    else:
        return f"{val_mm:.0f}mm"


# ── Drawing ──────────────────────────────────────────────────────────────────
def draw_detections(disp_l, detections, scale):
    """Draw YOLO bounding boxes with 3D measurements on the left display image."""
    for det in detections:
        x1 = int(det["x1"] * scale)
        y1 = int(det["y1"] * scale)
        x2 = int(det["x2"] * scale)
        y2 = int(det["y2"] * scale)
        label = det["label"]
        conf = det["conf"]
        meas = det.get("measurements")

        # Bbox
        cv2.rectangle(disp_l, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Class label + confidence
        class_text = f"{label} {conf:.0%}"
        cv2.putText(disp_l, class_text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if meas is None:
            cv2.putText(disp_l, "no match", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            continue

        # Measurement labels
        texts = []
        if "width_mm" in meas:
            texts.append(f"W: {format_mm(meas['width_mm'])}")
        if "height_mm" in meas:
            texts.append(f"H: {format_mm(meas['height_mm'])}")
        if "diag_mm" in meas:
            texts.append(f"D: {format_mm(meas['diag_mm'])}")

        # Draw measurement text below bbox
        for j, txt in enumerate(texts):
            ty = y2 + 18 + j * 20
            # Background
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(disp_l, (x1 - 1, ty - th - 2), (x1 + tw + 4, ty + 4), (0, 0, 0), -1)
            cv2.putText(disp_l, txt, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Draw measurement lines on the bbox
        mid_y = (y1 + y2) // 2
        mid_x = (x1 + x2) // 2

        # Width line (horizontal, at mid-height)
        if "width_mm" in meas:
            cv2.line(disp_l, (x1, mid_y), (x2, mid_y), (255, 200, 0), 1, cv2.LINE_AA)
        # Height line (vertical, at mid-width)
        if "height_mm" in meas:
            cv2.line(disp_l, (mid_x, y1), (mid_x, y2), (200, 255, 0), 1, cv2.LINE_AA)
        # Diagonal line
        if "diag_mm" in meas:
            cv2.line(disp_l, (x1, y1), (x2, y2), (0, 200, 255), 1, cv2.LINE_AA)


def draw_manual_points(disp_l, disp_r, points, distance, scale, left_width):
    """Draw manual measurement points on both images."""
    colors = [(0, 0, 255), (255, 0, 0)]

    for i, pt_data in enumerate(points):
        pt_l = pt_data["pt_l"]
        pt_r = pt_data["pt_r"]

        cx_l = int(pt_l[0] * scale)
        cy_l = int(pt_l[1] * scale)
        cv2.circle(disp_l, (cx_l, cy_l), 8, colors[i], -1)
        cv2.circle(disp_l, (cx_l, cy_l), 10, (255, 255, 255), 2)
        cv2.putText(disp_l, f"P{i+1}", (cx_l + 15, cy_l - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cx_r = int(pt_r[0] * scale)
        cy_r = int(pt_r[1] * scale)
        cv2.circle(disp_r, (cx_r, cy_r), 8, colors[i], -1)
        cv2.circle(disp_r, (cx_r, cy_r), 10, (255, 255, 255), 2)
        cv2.putText(disp_r, f"P{i+1}'", (cx_r + 15, cy_r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if len(points) == 2 and distance is not None:
        p1 = (int(points[0]["pt_l"][0] * scale), int(points[0]["pt_l"][1] * scale))
        p2 = (int(points[1]["pt_l"][0] * scale), int(points[1]["pt_l"][1] * scale))
        cv2.line(disp_l, p1, p2, (0, 255, 255), 2)

        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        text = f"{distance/10:.1f} cm" if distance >= 100 else f"{distance:.1f} mm"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        cv2.rectangle(disp_l, (mid_x - 5, mid_y - th - 10),
                      (mid_x + tw + 10, mid_y + 10), (0, 0, 0), -1)
        cv2.putText(disp_l, text, (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)


# ── Main Loop ────────────────────────────────────────────────────────────────
def main():
    calib = load_calibration(CALIB_FILE)

    # Load YOLO model on GPU
    print(f"[INFO] Loading YOLO model from '{YOLO_MODEL}'...")
    model = YOLO(YOLO_MODEL)
    model.to("cuda")
    print(f"  ✓ Model loaded on ({model.device})")

    cap_l = open_camera(CAM_LEFT_ID)
    cap_r = open_camera(CAM_RIGHT_ID)
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # State
    manual_points = []
    manual_distance = None
    screenshot_count = 0
    yolo_enabled = True

    current_rect_l = None
    current_rect_r = None

    left_disp_w = int(RESOLUTION[0] * PREVIEW_SCALE)

    def on_mouse(event, x, y, flags, param):
        nonlocal manual_points, manual_distance, current_rect_l, current_rect_r

        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(manual_points) >= 2:
            return
        if current_rect_l is None or current_rect_r is None:
            return
        if x >= left_disp_w:
            print("  ✗ Click on the LEFT image, not the right.")
            return

        x_full = int(x / PREVIEW_SCALE)
        y_full = int(y / PREVIEW_SCALE)

        x_r, y_r, score = find_correspondence(current_rect_l, current_rect_r, x_full, y_full)

        if x_r is None:
            print(f"  ✗ No match for ({x_full}, {y_full}). Score: {score:.3f}")
            return

        disparity = x_full - x_r
        if disparity <= 0:
            print(f"  ✗ Invalid disparity ({disparity:.1f})")
            return

        pt_l = (x_full, y_full)
        pt_r = (x_r, y_r)
        p3d = triangulate_point(calib["P1"], calib["P2"], pt_l, pt_r)

        if p3d[2] <= 0:
            print(f"  ✗ Negative depth ({p3d[2]:.1f} mm)")
            return

        manual_points.append({"pt_l": pt_l, "pt_r": pt_r, "p3d": p3d, "score": score})
        print(f"  Point {len(manual_points)}: L=({pt_l[0]},{pt_l[1]})  R=({x_r},{y_r})  "
              f"disp={disparity:.1f}  depth={p3d[2]:.0f}mm  score={score:.3f}")

        if len(manual_points) == 2:
            manual_distance = float(np.linalg.norm(manual_points[1]["p3d"] - manual_points[0]["p3d"]))
            print(f"\n  ═══════════════════════════════════════")
            print(f"  ✓  Distance: {manual_distance:.1f} mm  ({manual_distance/10:.2f} cm)")
            print(f"  ═══════════════════════════════════════\n")

    win_name = "Stereo Measurement + YOLO26"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, left_disp_w * 2, int(RESOLUTION[1] * PREVIEW_SCALE))
    cv2.setMouseCallback(win_name, on_mouse)

    print("\n" + "=" * 60)
    print("  Stereo Measurement Tool + YOLO26 Detection")
    print("  Click two points on LEFT image for manual measurement.")
    print("  YOLO bboxes show W/H/Diagonal in real-world units.")
    print("  [r] reset  [s] save  [d] toggle detection  [q] quit")
    print("=" * 60)

    frame_count = 0

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            continue

        current_rect_l, current_rect_r = rectify(frame_l, frame_r, calib)
        frame_count += 1

        # Run YOLO detection on the rectified left image (every frame)
        detections = []
        if yolo_enabled:
            results = model(current_rect_l, conf=YOLO_CONF, imgsz=YOLO_IMGSZ,
                            verbose=False, device="cuda")
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    label = model.names[cls_id]

                    # Filter: only bottles
                    if label not in LABELS2DETECT:
                        continue

                    # Measure this bbox
                    meas = measure_bbox(
                        current_rect_l, current_rect_r,
                        x1, y1, x2, y2,
                        calib["P1"], calib["P2"]
                    )

                    detections.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": conf, "label": label,
                        "measurements": meas,
                    })

        # Scale both images for display
        disp_l = cv2.resize(current_rect_l, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
        disp_r = cv2.resize(current_rect_r, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)

        # Draw YOLO detections with measurements
        draw_detections(disp_l, detections, PREVIEW_SCALE)

        # Draw manual points
        draw_manual_points(disp_l, disp_r, manual_points, manual_distance,
                           PREVIEW_SCALE, left_disp_w)

        # Labels
        cv2.putText(disp_l, "LEFT (click here)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        det_status = "ON" if yolo_enabled else "OFF"
        cv2.putText(disp_r, f"RIGHT | YOLO: {det_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # Combine side by side
        combined = np.hstack([disp_l, disp_r])

        cv2.putText(combined,
                    "[r]eset  [s]ave  [d]etection  [q]uit",
                    (10, combined.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(win_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            manual_points = []
            manual_distance = None
            print("[RESET] Manual points cleared.")
        elif key == ord("d"):
            yolo_enabled = not yolo_enabled
            print(f"[YOLO] Detection {'ON' if yolo_enabled else 'OFF'}")
        elif key == ord("s"):
            screenshot_count += 1
            fname = os.path.join(SCREENSHOT_DIR, f"measurement_{screenshot_count:03d}.png")
            cv2.imwrite(fname, combined)
            print(f"  ✓ Screenshot saved: {fname}")

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
