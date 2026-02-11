# Stereo Camera Calibration & Distance Measurement

A complete pipeline for calibrating two **DJI Osmo Action 5 Pro** cameras as a stereo pair and measuring real-world distances between two user-selected points using OpenCV's triangulation.

```
┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│  1. CAPTURE          │ ───► │  2. CALIBRATE        │ ───► │  3. MEASURE          │
│  capture_calibration │      │  calibrate_stereo.py │      │  measure.py          │
│  _images.py          │      │                      │      │                      │
│                      │      │  Individual calib    │      │  Live stereo feed    │
│  Side-by-side live   │      │  Stereo calib        │      │  Click 2 points      │
│  preview + chessboard│      │  Rectification       │      │  Template matching   │
│  corner detection    │      │  Undistortion maps   │      │  Triangulation       │
│                      │      │  → stereo_params.npz │      │  → Distance in mm    │
└──────────────────────┘      └──────────────────────┘      └──────────────────────┘
```

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Background: Why Camera Calibration?](#background-why-camera-calibration)
3. [Script 1: Capturing Calibration Images](#script-1-capturing-calibration-images)
4. [Script 2: Calibration Pipeline](#script-2-calibration-pipeline)
5. [Script 3: Measurement Tool](#script-3-measurement-tool)
6. [Configuration Reference](#configuration-reference)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install dependencies
.venv/bin/pip install -r requirements.txt

# 2. Capture ≥15 chessboard image pairs
.venv/bin/python capture_calibration_images.py

# 3. Run calibration
.venv/bin/python calibrate_stereo.py

# 4. Measure things
.venv/bin/python measure.py
```

---

## Background: Why Camera Calibration?

### The Pinhole Camera Model

Every real camera is an imperfect version of an idealized **pinhole camera**. In the pinhole model, a 3D point in the world `P = (X, Y, Z)` is projected onto a 2D image plane as:

```
s · [u]   [fx  0  cx] [r11 r12 r13 tx] [X]
    [v] = [ 0 fy  cy] [r21 r22 r23 ty] [Y]
    [1]   [ 0  0   1] [r31 r32 r33 tz] [Z]
                                        [1]
       =      K      ·    [R | t]      · Pw
```

Where:
- **K** is the **intrinsic matrix** (camera matrix) — encodes the internal properties of the camera:
  - `fx, fy` — focal length in pixels (how strongly the lens converges light)
  - `cx, cy` — principal point (where the optical axis hits the sensor, ideally the image center)
- **[R | t]** is the **extrinsic matrix** — describes the camera's position and orientation in the world

**Why this matters:** Without knowing `K`, you cannot convert pixel coordinates into real-world angles or distances. It's the bridge between "pixels on screen" and "geometry in reality."

### Lens Distortion

Real lenses are not perfect. They introduce two types of distortion:

1. **Radial distortion** — straight lines appear curved (barrel or pincushion effect). Worse near image edges.
   ```
   x_distorted = x · (1 + k₁r² + k₂r⁴ + k₃r⁶)
   y_distorted = y · (1 + k₁r² + k₂r⁴ + k₃r⁶)
   ```

2. **Tangential distortion** — caused by the lens not being perfectly parallel to the sensor.
   ```
   x_distorted = x + [2p₁xy + p₂(r² + 2x²)]
   y_distorted = y + [p₁(r² + 2y²) + 2p₂xy]
   ```

This gives us **5 distortion coefficients**: `(k₁, k₂, p₁, p₂, k₃)`.

**Why this matters:** If you don't correct for distortion, your pixel measurements will be systematically wrong — especially near the image edges. A straight ruler would appear curved, and triangulation would produce incorrect 3D coordinates.

### Stereo Vision and Triangulation

With a **single camera**, you lose all depth information — a point at 1m and a point at 10m can produce the same pixel. With **two cameras** at a known separation (the **baseline**), you can recover depth through **triangulation**:

```
         ←── baseline (b) ──→
    [Camera L]            [Camera R]
         \                  /
          \    3D point P  /
           \      ●       /
            \    / \     /
             \  /   \   /
              \/     \ /
         image L    image R
         (xL, yL)   (xR, yR)

    Depth Z = f · b / d
    where d = xL - xR  (disparity)
```

The disparity `d` is the horizontal pixel difference of the same physical point between left and right images. Larger disparity = closer object. This is why you need both cameras running simultaneously.

---

## Script 1: Capturing Calibration Images

**File:** `capture_calibration_images.py`

### What it does

Opens both DJI Osmo cameras, shows a live side-by-side preview, and lets you capture synchronized chessboard image pairs.

### Step-by-step explanation

#### 1. Camera Initialization

```python
cap = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
```

**Why:** The cameras are opened via Video4Linux2 (V4L2) in MJPEG mode. MJPEG allows the cameras to deliver full 1920×1080 at 30fps over USB — raw (YUYV) mode is typically too bandwidth-heavy for USB 2.0 and would limit you to lower resolutions or framerates.

#### 2. Chessboard Corner Detection

```python
found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
```

**What OpenCV does internally:**
1. **Adaptive thresholding** — converts the grayscale image into a binary image using multiple threshold levels to handle varying lighting.
2. **Contour extraction** — finds all closed contours (potential squares).
3. **Quad filtering** — selects only contours that are roughly quadrilateral (4 corners).
4. **Grid grouping** — looks for quads that form a regular grid pattern matching the expected `BOARD_SIZE` (9×6 inner corners).
5. **Corner ordering** — sorts corners in a consistent left-to-right, top-to-bottom order so that corner[0] in the left image corresponds to corner[0] in the right image.

**Why a chessboard?** The chessboard has several properties that make it ideal:
- Sharp black-white transitions enable sub-pixel corner localization.
- The grid structure provides many points (54 for 9×6) in a single image.
- The known geometry (all squares same size, all coplanar) provides strong geometric constraints.

#### 3. Sub-pixel Refinement

```python
corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
```

**What OpenCV does internally:**
- For each detected corner, it examines an 11×11 pixel neighborhood.
- It fits a mathematical model to the intensity gradient around the corner.
- Through iterative refinement (gradient descent), it finds the precise sub-pixel location where gradients from all four edges converge.
- Convergence is controlled by `criteria` (max 30 iterations or epsilon < 0.001).

**Why this matters:** Raw corner detection is only accurate to ±0.5 pixels. Sub-pixel refinement brings this down to ±0.05 pixels or better. Since your measurement accuracy is directly proportional to corner accuracy, this step can improve your final results by ~10×.

#### 4. Why You Need 15–20 Pairs

The calibration solver needs to estimate:
- 4 intrinsic parameters per camera (fx, fy, cx, cy) = **8 total**
- 5 distortion coefficients per camera = **10 total**
- 3 rotation + 3 translation for stereo = **6 total**
- **= 24 unknowns minimum**

Each chessboard image provides 54 corner correspondences (2D constraints each), but they must span diverse positions and angles to avoid degenerate solutions. In practice, 15–20 well-distributed pairs give robust estimates.

**Tips for good calibration images:**
- Cover the entire field of view — corners, edges, center.
- Vary the chessboard angle (tilt left, right, toward, away).
- Vary the distance (close-up filling the frame and further away).
- Keep the board flat and still when capturing.

---

## Script 2: Calibration Pipeline

**File:** `calibrate_stereo.py`

### Step-by-step explanation

#### 1. Object Points Definition

```python
objp = np.zeros((CHESS_ROWS * CHESS_COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESS_COLS, 0:CHESS_ROWS].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM
```

**What this does:** Creates the 3D coordinates of each chessboard corner in the chessboard's own coordinate system. We assume the board lies flat on the Z=0 plane, so the corners are at positions like `(0,0,0)`, `(25,0,0)`, `(50,0,0)`, etc. (in mm).

**Why this matters:** These are the "ground truth" 3D positions. The calibration algorithm compares where these points _should_ appear in the image (via the projection model) with where they _actually_ appear. The difference drives the optimization.

#### 2. Individual Camera Calibration (`cv2.calibrateCamera`)

```python
ret, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
```

**What OpenCV does internally:**
1. **Initial estimate** — uses a closed-form solution (Zhang's method) to get an initial guess for K.
2. **Levenberg-Marquardt optimization** — iteratively adjusts K and D to minimize the total **reprojection error**: the sum of squared distances between detected corners and where the model predicts they should be.
3. **Output:**
   - `K` — 3×3 intrinsic matrix
   - `D` — distortion coefficients (k₁, k₂, p₁, p₂, k₃)
   - `rvecs, tvecs` — rotation and translation for each image (where the chessboard was relative to the camera)

**Why individual calibration first?** Stereo calibration has many parameters to optimize simultaneously. By first solving each camera independently, we get good initial intrinsics. The stereo step then only needs to find the relative pose (R, T) between cameras, which is a much more constrained problem.

#### 3. Reprojection Error

```python
projected, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, D)
error = cv2.norm(img_pts[i], projected, cv2.NORM_L2)
```

**What this measures:** For each calibration image, the code:
1. Takes the known 3D chessboard points.
2. Projects them into the image using the estimated K, D, and the pose (rvec, tvec).
3. Compares the projected positions with the actually detected corners.
4. The RMS distance (in pixels) is the reprojection error.

**Interpretation:**
- **< 0.5 px** — Excellent. Your measurements will be very accurate.
- **0.5–1.0 px** — Good. Suitable for most practical measurements.
- **> 1.0 px** — Fair. Consider recapturing with better images.

#### 4. Stereo Calibration (`cv2.stereoCalibrate`)

```python
ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_l, img_points_r,
    K1, D1, K2, D2, image_size,
    flags=cv2.CALIB_FIX_INTRINSIC,
)
```

**What OpenCV does internally:**
1. Uses the pre-calibrated intrinsics (K1, D1, K2, D2) as fixed inputs (`CALIB_FIX_INTRINSIC`).
2. For each image pair, it knows where the same 3D points appear in both cameras.
3. It solves for the **rotation matrix R** and **translation vector T** that describe how Camera 2 is positioned and oriented relative to Camera 1.
4. Also computes the **Essential matrix E** (encodes R and T) and **Fundamental matrix F** (relates corresponding points between uncalibrated views).

**The key outputs:**
- **R** (3×3) — rotation from Camera 1 to Camera 2
- **T** (3×1) — translation vector. Its magnitude is your **baseline** (in mm, since your square size was in mm)

**Why `CALIB_FIX_INTRINSIC`?** Since we already have good intrinsics from Step 2, fixing them during stereo calibration prevents the optimizer from "trading" intrinsic accuracy for extrinsic fit. This produces more stable results.

#### 5. Stereo Rectification (`cv2.stereoRectify`)

```python
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, D1, K2, D2, image_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
)
```

**What OpenCV does internally:**
1. Computes two rotation matrices (R1, R2) that, when applied to each camera, make their image planes **coplanar** and their rows perfectly aligned horizontally.
2. After rectification, the same physical point will appear on **exactly the same row** in both images (epipolar lines become horizontal).
3. Computes **projection matrices** P1 (3×4) and P2 (3×4) for the rectified coordinate system.
4. Computes the **disparity-to-depth matrix Q** (4×4) for converting disparity maps to 3D.

**Why rectification?**
- Without it, corresponding points could be anywhere in the other image (2D search).
- With it, for any point at row `y` in the left image, its correspondence is guaranteed to be at the same row `y` in the right image (1D search).
- This makes template matching faster and more reliable.

**What `alpha=0` means:** Only pixels visible in both rectified images are kept (no black borders). This ensures every pixel in the output has valid data.

#### 6. Undistortion + Rectification Maps

```python
map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
```

**What OpenCV does internally:**
- Pre-computes a pixel-level lookup table. For each pixel `(u, v)` in the output (rectified) image, the map stores the corresponding `(x, y)` position in the original (distorted) image.
- At runtime, `cv2.remap()` simply looks up each pixel — no complex math needed per frame.

**Why pre-compute?** The mapping involves inverse distortion and rotation calculations that would be expensive to repeat 30× per second for 2 million pixels. By doing it once and storing the result, live rectification runs at full framerate.

---

## Script 3: Measurement Tool

**File:** `measure.py`

### Step-by-step explanation

#### 1. Live Rectification

```python
rect_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
rect_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)
```

Every frame from both cameras is remapped using the pre-computed lookup tables. This simultaneously:
- **Removes lens distortion** (undoes the barrel/pincushion effect).
- **Applies rectification** (rotates the image so epipolar lines are horizontal).

After this step, corresponding points between left and right images lie on the same pixel row.

#### 2. User Clicks a Point on the Left Image

When you click on the left image, the system records the pixel coordinates `(x_L, y_L)` in the rectified image.

#### 3. Finding the Correspondence in the Right Image (Template Matching)

```python
template = rect_l[y-25:y+26, x-25:x+26]   # 51×51 patch
search_strip = rect_r[y-28:y+29, 0:x+26]   # Same rows, left of x
result = cv2.matchTemplate(search_strip, template, cv2.TM_CCOEFF_NORMED)
```

**What OpenCV does internally:**
1. Extracts a small patch (51×51 pixels) centered on the clicked point from the left image.
2. Slides this patch across the **same horizontal strip** in the right image (because after rectification, the match _must_ be on the same row).
3. At each position, computes the **Normalized Cross-Correlation (NCC)** — a similarity score from -1 to +1:
   ```
   NCC = Σ[(L - L̄)(R - R̄)] / √[Σ(L-L̄)² · Σ(R-R̄)²]
   ```
4. The position with the highest NCC is the best match.

**Why template matching instead of feature matching?**
- Feature detectors (SIFT, ORB) find _their own_ interesting points — but you want to match _your specific_ clicked point.
- Template matching directly searches for "this exact patch of pixels" in the other image.
- The epipolar constraint (same# Stereo Camera Calibration & Measurement

A Python toolkit for calibrating a stereo camera pair and performing real-world 3D measurements. Now integrates **YOLO26** for automatic object detection and sizing (specifically bottles).

## Features

- **Robust Stereo Calibration**:
  - Uses OpenCV's standard pipeline with fixed intrinsics for reliability.
  - Automatic corner ordering checks and consistency validation.
  - Per-image quality reports to identify bad captures.
  - Full-resolution rectification and disparity maps.

- **Live Measurement Tool (`measure.py`)**:
  - **Dual-View Interface**: Shows rectified left/right views side-by-side.
  - **Automatic Bottle Measurement**: Detects bottles using YOLO26 (running on GPU) and measures their real-world Width, Height, and Diagonal.
  - **Manual Measurement**: Click two points to measure distance between arbitrary features.
  - **Visual Verification**: Shows exactly where the stereo matcher found correspondence in the right image.

- **Utilities**:
  - `capture_calibration_images.py`: Helper to capture synchronized stereo pairs.
  - `debug_corners.py`: Visual tool to debug board detection/ordering issues.

---

## 1. Setup

### Prerequisites
- Python 3.8+
- NVIDIA GPU (RTX 30xx/40xx recommended) with CUDA support (for YOLO26)
- OpenCV, NumPy, Ultralytics

### Installation
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install ultralytics

# Ensure you have the YOLO model
# Place 'yolo26l.pt' in the Models/ directory
mkdir -p Models
# (Download or copy your model here)
```

## 2. Usage Workflow

### Step 1: Capture Calibration Images
Print a checkerboard pattern (e.g., 7x4 inner corners) and mount it on a flat surface.
```bash
python capture_calibration_images.py
```
- Press **SPACE** to capture a pair.
- Capture 30-50 pairs from diverse angles and distances.
- Images are saved to `calibration_images/left/` and `calibration_images/right/`.

**Crucial:** Ensure both cameras are in the **exact same FOV mode** (e.g., both "Wide") and resolution.

### Step 2: Calibrate
Run the calibration script to compute intrinsics and stereo geometry.
```bash
python calibrate_stereo.py
```
- This script detects corners, validates left/right consistency, and computes `stereo_params.npz`.
- It will report reprojection errors. **Stereo RMS should be < 1.0 px** for high accuracy.
- A rectified sample image is saved to `debug/rectified_pair.png`. **Check this image!** The green lines must be perfectly horizontal across both views.

### Step 3: Measure
Run the live measurement tool.
```bash
python measure.py
```

#### Controls
| Key | Action |
|---|---|
| **Left Click** | (On Left Image) Set manual measurement points |
| **d** | Toggle YOLO detection ON/OFF |
| **r** | Reset manual points |
| **s** | Save screenshot to `screenshots/` |
| **q** | Quit |

#### How it Works
- **Automatic**: The script runs YOLO26 on the left frame. For every detected "bottle", it finds the 3D positions of the bbox edges and calculates the object's physical dimensions.
- **Manual**: Click a feature on the left image. The script searches the corresponding row in the right image (epipolar constraint). It draws the match on the right side so you can verify if it locked onto the correct feature. If the match is wrong, the measurement is wrong.

---

## Troubleshooting

- **"Stereo error is high (>1.0 px)"**: Your cameras likely moved, have different settings, or the board detection was messy. Run `debug_corners.py` to check for flipped corners.
- **Measurements are wildly off**:
  - Check `SQUARE_SIZE_MM` in `calibrate_stereo.py` matches your physical board.
  - Ensure the matched point on the right image (in `measure.py`) is actually correct.
  - Avoid measuring featureless areas (blank walls) where template matching fails.
- **YOLO not using GPU**: Ensure `torch.cuda.is_available()` is True and you installed the CUDA version of PyTorch.

## Configuration
Key settings are top-level constants in the scripts:

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `CAM_LEFT_ID` | `measure.py` | `2` | Device ID for left camera |
| `CAM_RIGHT_ID` | `measure.py` | `4` | Device ID for right camera |
| `SQUARE_SIZE_MM` | `calibrate_stereo.py` | `50.0` | Size of one square on your board |
| `CHESS_COLS` | `calibrate_stereo.py` | `7` | Inner corners horizontal |
| `CHESS_ROWS` | `calibrate_stereo.py` | `4` | Inner corners vertical |
| `YOLO_CONF` | `measure.py` | `0.4` | Confidence threshold for detection |
| `YOLO_MODEL` | `measure.py` | `Models/yolo26l.pt` | Path to model weights |

---

## Troubleshooting

### "Cannot open /dev/videoX"
- Check camera connections: `ls /dev/video*`
- The indices may change after reconnecting. Update `CAM_LEFT_ID` / `CAM_RIGHT_ID`.

### "No chessboard found in both images"
- Ensure the entire chessboard is visible in **both** cameras simultaneously.
- Improve lighting — avoid shadows across the board.
- Hold the board steady (motion blur defeats corner detection).
- Try rotating the board so the 9-column side aligns with the wider camera dimension.

### Reprojection error > 1.0
- Recapture with more diverse angles and positions.
- Ensure the chessboard is perfectly flat (no warping).
- Verify `SQUARE_SIZE_MM` is correct.
- Use at least 15 pairs.

### Template matching gives wrong correspondences
- Increase `TEMPLATE_HALF` for more context (but slower).
- Ensure calibration quality is good (low reprojection error).
- Click on **textured** areas — matching fails on uniform surfaces.

### Measured distance seems wrong
- Verify with a known object (ruler, credit card = 85.6mm).
- Double-check `SQUARE_SIZE_MM`.
- Objects very close (< 30cm) or very far (> 3m) will have reduced accuracy.

---

## Project Structure

```
StereoCamCalibration/
├── capture_calibration_images.py   # Step 1: Capture chessboard pairs
├── calibrate_stereo.py             # Step 2: Run calibration
├── measure.py                      # Step 3: Live measurement
├── record_dji.sh                   # Video recording utility
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
├── calibration_images/             # Created by Step 1
│   ├── left/
│   │   ├── img_01.png
│   │   └── ...
│   └── right/
│       ├── img_01.png
│       └── ...
├── calibration_data/               # Created by Step 2
│   └── stereo_params.npz
├── screenshots/                    # Created by Step 3
│   └── measurement_001.png
└── Videos/                         # Raw stereo recordings
```
