"""
================================================================================
 sem_pipeline.py  --  End-to-end SEM -> parametric-shape fitting pipeline
================================================================================

WHAT THIS SCRIPT DOES (the full pipeline, in order)
---------------------------------------------------
This single file consolidates what used to live across many scripts in `old/`
(`utils.py`, `sam_test.py`, `find_params_exclude_wave.py`, `generate_gds.py`,
`show_overlay.py`, `visual_creator.py`, `process_sem.py`). It implements the
four stages the pipeline is built around:

    STAGE 1  SEM -> scale          Read an SEM image and figure out how many
                                   nanometers each pixel represents. This is
                                   done automatically by OCR-ing the scale-bar
                                   label ("1 um", "500 nm", ...) and measuring
                                   the scale bar's pixel length.
                                   -> from utils._find_pixel_to_nm_auto

    STAGE 2  SEM -> cutout (SAM)   Use Meta's Segment Anything Model (SAM) to
                                   segment the structure of interest. You give
                                   SAM one or more "positive" click points; it
                                   returns a mask. We take the largest contour
                                   of that mask and convert it from pixels to
                                   nanometers using the Stage-1 scale.
                                   -> from sam_test.py / visual_creator.lock_shape

    STAGE 3  optimize fit          Fit an idealized *parametric* waveguide shape
                                   to the measured contour. This is a NESTED
                                   optimization:
                                     (3a) OUTER: search the shape parameters
                                          [W1,W2,W3,W4,L1,L2,length_waveguide]
                                          (differential evolution, +/-20% box).
                                     (3b) INNER (per outer candidate): find the
                                          best rigid placement -- translation
                                          (dx,dy) and ROTATION (theta) -- that
                                          maximizes overlap of the parametric
                                          shape with the measured contour.
                                   i.e. "optimize params of shape, then optimize
                                   angle for overlap for each one."
                                   -> from find_params_exclude_wave.py

    STAGE 4  compare vs existing   Take the fitted/measured outline and register
                                   it SPATIALLY onto an existing design layout
                                   (a reference GDS), by maximizing intersection
                                   area over angle + (dx,dy). Writes a combined
                                   GDS with the design (layer 1) and the aligned
                                   outline (layer 2) so you can see how the
                                   fabricated shape compares to what was drawn.
                                   -> from utils.superimpose_gds

HOW AN AGENT / SCRIPT SHOULD USE THIS
-------------------------------------
Everything is exposed as plain functions with no GUI dependency, plus one
convenience orchestrator `run_full_pipeline(...)`. The SAM click-point
selection can be supplied programmatically (`sam_points=[[x,y], ...]`) so the
whole thing runs headless, OR left as None to open an interactive matplotlib
window for manual clicking.

    from sem_pipeline import run_full_pipeline

    result = run_full_pipeline(
        sem_path="old/SEM_results.tif",
        nominal_params={"W1":300,"W2":400,"W3":500,"W4":600,"L1":200,"L2":300},
        sam_points=[[760, 420]],            # None -> interactive clicking
        design_gds="old/my_layout.gds",     # None -> skip Stage 4
        out_prefix="results/run1",
    )
    print(result["complete_params"], result["best_hausdorff"])

DATA CONVENTIONS (important, they are easy to get wrong)
--------------------------------------------------------
* Shape parameter vector order is ALWAYS:
      [W1, W2, W3, W4, L1, L2, length_waveguide]
  (7 values). generate_waveguide_shape() expects exactly this order.
* Contours are (N, 2) float arrays in NANOMETERS after Stage 2.
* Angles: internal shape fitting uses radians; the GDS spatial compare
  (Stage 4) works in degrees. Both are noted at each call site.

DEPENDENCIES
------------
    numpy, scipy, shapely, gdspy, opencv-python (cv2), pillow (PIL),
    pytesseract (+ system `tesseract` binary), matplotlib,
    torch + segment-anything (with a SAM checkpoint .pth on disk).

SAM checkpoint (default vit_b):
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
================================================================================
"""

from __future__ import annotations

import os
import re
import json
import time
import multiprocessing
from functools import partial

import numpy as np

# Optimization / geometry
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from shapely.ops import unary_union

# GDS layout I/O
import gdspy


# =============================================================================
# STAGE 1 -- SEM -> SCALE (nm per pixel)
# =============================================================================
# We locate two small regions of interest (ROIs) baked into the SEM image
# chrome: the text label of the scale bar, and the scale bar itself.
#   - OCR the text ("1 um", "500nm", ...) -> physical length in nm.
#   - Measure the bar's length in pixels (find the two bright end-caps).
#   - nm_per_px = length_nm / length_px.
#
# NOTE: The default ROIs below are hard-coded for the microscope/magnification
# that produced `SEM_results.tif`. For a different instrument or resolution you
# MUST update text_box / scale_box (they are (x1, y1, x2, y2) pixel rectangles).
# If OCR/scale detection is unreliable for your images, pass a known
# `nm_per_px` straight into the later stages and skip this entirely.
# =============================================================================

# Default ROIs (x1, y1, x2, y2) in pixels -- tuned for SEM_results.tif chrome.
DEFAULT_TEXT_BOX = (767, 701, 827, 719)    # scale-bar text label
DEFAULT_SCALE_BOX = (765, 715, 890, 742)   # scale-bar graphic


def _parse_scale_text_to_nm(meas_str: str) -> float:
    """
    Turn an OCR'd scale label (e.g. "1 um", "0.5um", "500 nm") into nanometers.

    OCR frequently misreads the leading "1" as T / I / l and mangles the micro
    sign, so we apply a small substitution table first, then try a few regexes,
    then fall back to "assume the number is in microns".
    """
    # Common OCR misreadings -> corrected token.
    substitutions = {
        'Tym': '1 µm', 'Tum': '1 µm', 'Tnm': '1 nm',
        'Iym': '1 µm', 'Ium': '1 µm', 'Inm': '1 nm',
        'lym': '1 µm', 'lum': '1 µm', 'lnm': '1 nm',
        'T': '1', 'I': '1', 'l': '1',
    }
    cleaned = meas_str
    for wrong, right in substitutions.items():
        cleaned = cleaned.replace(wrong, right)

    # Number + unit patterns, from most to least specific.
    patterns = [
        r'([\d\.]+)\s*([µumn]+)m?',   # "1 µm", "0.5um"
        r'([\d\.]+)\s*([µumn])',      # "1 µ", "0.5u"
        r'([\d\.]+)\s*([mn])',        # "1 m", "0.5n"
    ]
    unit_to_nm = {'nm': 1, 'um': 1e3, 'mm': 1e6, 'n': 1, 'm': 1e3}

    for pattern in patterns:
        m = re.match(pattern, cleaned, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            unit = m.group(2).lower().replace('µ', 'u')
            if unit in unit_to_nm:
                return val * unit_to_nm[unit]

    # Last resort: grab any number and assume microns.
    num_match = re.search(r'([\d\.]+)', cleaned)
    if num_match:
        val = float(num_match.group(1))
        print(f"  [scale] WARNING: couldn't parse unit in '{meas_str}', assuming um -> {val} um")
        return val * 1e3

    raise ValueError(f"Could not parse scale measurement: '{meas_str}' (cleaned: '{cleaned}')")


def detect_scale_nm_per_px(sem_path: str,
                           text_box=DEFAULT_TEXT_BOX,
                           scale_box=DEFAULT_SCALE_BOX,
                           verbose: bool = True) -> dict:
    """
    STAGE 1. Automatically compute nm-per-pixel from an SEM image's scale bar.

    Returns a dict:
        { 'nm_per_px', 'length_nm', 'pixel_distance', 'raw_ocr' }

    (Original: utils._find_pixel_to_nm_auto, but returns the intermediate
    values too so downstream code / visualization can show its work.)
    """
    import cv2
    import pytesseract
    from scipy.signal import find_peaks

    img = cv2.imread(sem_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image '{sem_path}'")

    # --- 1a. OCR the text label -> physical length in nm ---------------------
    x1, y1, x2, y2 = text_box
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # White-on-dark text: invert-threshold so tesseract sees dark text on white.
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    raw_ocr = pytesseract.image_to_string(th, config='--psm 7').strip()  # psm 7 = single line
    length_nm = _parse_scale_text_to_nm(raw_ocr)

    # --- 1b. Measure the scale bar's length in pixels ------------------------
    x1, y1, x2, y2 = scale_box
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Normalize to 0..255 so the peak threshold is contrast-independent.
    norm = (gray.astype(np.float32) / gray.max() * 255) if gray.max() > 0 else gray.astype(np.float32)
    norm = norm.astype(np.uint8)
    # Collapse vertically -> 1D intensity profile across the bar's width.
    projection = norm.sum(axis=0)
    projection_smooth = cv2.GaussianBlur(projection.reshape(-1, 1), (9, 1), 0).flatten()
    # The bar's two bright end-caps show up as the two outermost strong peaks.
    threshold = 0.8 * np.max(projection_smooth)
    peaks, _ = find_peaks(projection_smooth, height=threshold)
    if len(peaks) < 2:
        peaks = np.argsort(projection_smooth)[-2:]   # fallback: two brightest columns
    p1, p2 = int(np.min(peaks)), int(np.max(peaks))
    pixel_distance = abs(p2 - p1)

    if pixel_distance == 0:
        raise ValueError("Scale bar pixel length is zero -- check scale_box ROI or image quality.")

    nm_per_px = float(length_nm) / float(pixel_distance)

    if verbose:
        print(f"  [scale] OCR='{raw_ocr}'  ->  {length_nm:.1f} nm  over  {pixel_distance} px")
        print(f"  [scale] resolution = {nm_per_px:.3f} nm/px")

    return {
        'nm_per_px': nm_per_px,
        'length_nm': float(length_nm),
        'pixel_distance': pixel_distance,
        'raw_ocr': raw_ocr,
    }


# =============================================================================
# STAGE 2 -- SEM -> CUTOUT (Segment Anything)
# =============================================================================
# We load SAM once, hand it the image, and let it segment the structure from a
# few click points. We then extract the largest external contour of the mask
# and convert it into nanometer coordinates using the Stage-1 scale.
#
# Two modes:
#   * headless : pass explicit `points=[[x_px, y_px], ...]` (positive prompts).
#   * interactive : pass points=None to open a matplotlib window where you
#     click on the structure; the live SAM mask updates on each click; close
#     the window when satisfied.
# =============================================================================

def load_sam_predictor(checkpoint: str = "sam_vit_b_01ec64.pth",
                       model_type: str = "vit_b"):
    """
    Build a SAM predictor. Uses Apple MPS if available, else CPU.
    Kept separate so the (slow) model load can be reused across many images.
    """
    import torch
    from segment_anything import sam_model_registry, SamPredictor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    print(f"  [sam] loaded {model_type} on {device}")
    return SamPredictor(sam)


def _mask_to_contour_nm(mask: np.ndarray, nm_per_px: float) -> np.ndarray:
    """
    Convert a boolean SAM mask into an (N,2) contour in NANOMETERS.
    Takes the single largest external contour (the main object outline).
    """
    import cv2
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("SAM mask produced no contour.")
    main_contour = max(contours, key=cv2.contourArea)          # largest = the structure
    pts_px = main_contour.reshape(-1, 2).astype(np.float64)    # (N,2) in pixels
    return pts_px * nm_per_px                                   # -> nanometers


def sam_cutout(sem_path: str,
               nm_per_px: float,
               predictor=None,
               points=None,
               checkpoint: str = "sam_vit_b_01ec64.pth") -> np.ndarray:
    """
    STAGE 2. Segment the structure and return its contour in nanometers.

    Parameters
    ----------
    sem_path   : path to the SEM image.
    nm_per_px  : scale from Stage 1 (used to convert the contour to nm).
    predictor  : an already-loaded SAM predictor (optional; built if None).
    points     : list of [x_px, y_px] positive prompt points.
                 If None -> open an interactive clicking window.
    checkpoint : SAM weights file (only used if predictor is None).

    Returns
    -------
    contour_nm : (N,2) float array, the object outline in nanometers.
    """
    from PIL import Image

    img = np.array(Image.open(sem_path).convert("RGB"))
    if predictor is None:
        predictor = load_sam_predictor(checkpoint)
    predictor.set_image(img)   # SAM embeds the image once; predict() is then cheap

    if points is None:
        # ---- interactive mode: click, watch the mask, close when done -------
        points = _interactive_sam_points(img, predictor)
        if not points:
            raise ValueError("No SAM points selected.")

    input_point = np.array(points, dtype=float)
    input_label = np.ones(len(points), dtype=int)   # 1 = "this pixel is inside the object"
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,       # single best mask
    )
    mask = masks[0]
    contour_nm = _mask_to_contour_nm(mask, nm_per_px)
    print(f"  [sam] mask score={float(scores[0]):.3f}  ->  contour with {len(contour_nm)} pts")
    return contour_nm


def _interactive_sam_points(img, predictor):
    """
    Open a matplotlib window; each click adds a positive prompt point and the
    SAM mask is re-predicted and drawn live. Returns the clicked points when
    the window is closed. (Original behavior of sam_test.py.)
    """
    import matplotlib.pyplot as plt

    clicked = []
    state = {"mask_artist": None, "pts_artist": None}

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Click points on the structure; close the window when done")

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        clicked.append([int(event.xdata), int(event.ydata)])
        pts = np.array(clicked)
        masks, _, _ = predictor.predict(
            point_coords=pts, point_labels=np.ones(len(pts), int), multimask_output=False)
        # Redraw mask + points.
        if state["mask_artist"] is not None:
            state["mask_artist"].remove()
        if state["pts_artist"] is not None:
            state["pts_artist"].remove()
        state["mask_artist"] = ax.contour(masks[0], colors='r')
        state["pts_artist"] = ax.scatter(pts[:, 0], pts[:, 1], c='lime', marker='x')
        fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    return clicked


# =============================================================================
# STAGE 3 -- OPTIMIZE FIT (parametric shape + rigid placement)
# =============================================================================
# The idealized structure is a straight waveguide (a rectangle) joined to a
# tapered transition polygon. It is fully described by 7 parameters:
#
#     length_waveguide : length of the straight rectangle section
#     W1 : width of the straight waveguide (rectangle height)
#     W2 : taper width at the junction (x=0)
#     W3 : taper width at x = L1
#     W4 : taper end width at x = L1 + L2
#     L1 : length of the first taper segment
#     L2 : length of the second taper segment
#
#            W2/2  __ W3/2
#      +---------+/     \__ W4/2
#   W1 |         |          \
#   ---|  rect   |0   taper   >   (x increases to the right)
#      |         |          /
#      +---------+\ ___ ___/
#     -Lwg        0  L1   L1+L2
#
# The fit has two layers:
#   3a  optimize_shape_params : OUTER differential-evolution over the 7 shape
#       parameters (each within +/-20% of nominal; length_waveguide gets a
#       wider, contour-derived bound). For every candidate parameter vector it
#       calls fit_waveguide_to_contours to score it.
#   3b  fit_waveguide_to_contours : INNER placement search -- given fixed shape
#       parameters, find the translation+rotation that best overlaps the
#       contour. Done as: centroid pre-align -> coarse angle sweep (0/90/180/270
#       with a dx,dy DE at each) -> a final 3-D DE over (dx,dy,theta). Objective
#       is the UNION AREA of contour+shape (smaller union == better overlap for
#       equal-ish areas). The returned score is the Hausdorff distance of the
#       final placement (a worst-case boundary mismatch, in nm).
# =============================================================================

# Canonical parameter order used everywhere in Stage 3.
PARAM_NAMES = ['W1', 'W2', 'W3', 'W4', 'L1', 'L2', 'length_waveguide']


def generate_waveguide_shape(params) -> np.ndarray:
    """
    Build the parametric outline (closed polygon, (N,2)) from a 7-vector
    [W1, W2, W3, W4, L1, L2, length_waveguide].

    The rectangle spans x in [-length_waveguide, 0]; the taper spans
    x in [0, L1+L2]. When W1 != W2 there is a small vertical step at x=0 that we
    include so the outline stays a single closed ring.
    """
    W1, W2, W3, W4, L1, L2, length_waveguide = params
    pts = []

    # Left rectangle: bottom-left -> top-left.
    pts.append((-length_waveguide, -W1 / 2))
    pts.append((-length_waveguide,  W1 / 2))

    # Step from rectangle (W1) up to taper mouth (W2) at x=0, if they differ.
    if abs(W1 - W2) > 1e-6:
        pts.append((0,  W1 / 2))
        pts.append((0,  W2 / 2))
    else:
        pts.append((0,  W1 / 2))

    # Taper top edge: mouth -> W3 at L1 -> W4 at L1+L2.
    pts.append((L1,       W3 / 2))
    pts.append((L1 + L2,  W4 / 2))
    # Taper end + bottom edge back toward the mouth.
    pts.append((L1 + L2, -W4 / 2))
    pts.append((L1,      -W3 / 2))

    # Step back down from taper (W2) to rectangle (W1) at x=0, if they differ.
    if abs(W1 - W2) > 1e-6:
        pts.append((0, -W2 / 2))
        pts.append((0, -W1 / 2))
    else:
        pts.append((0, -W1 / 2))

    # Close the ring.
    pts.append((-length_waveguide, -W1 / 2))
    return np.array(pts)


def transform_shape(points: np.ndarray, dx: float, dy: float, dtheta: float) -> np.ndarray:
    """
    Rigidly place a shape: rotate by `dtheta` (radians) about its own centroid,
    then translate by (dx, dy). Rotating about the centroid keeps dx/dy and
    angle roughly decoupled, which helps the optimizer.
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    R = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                  [np.sin(dtheta),  np.cos(dtheta)]])
    rotated = centered @ R.T
    return rotated + centroid + np.array([dx, dy])


def _combined_area(contour_pts, shape_params, dx, dy, theta_rad) -> float:
    """
    Objective helper for the placement search: area of the UNION of the
    contour polygon and the transformed parametric shape. Perfect overlap of
    identical shapes minimizes this; misalignment grows the union. Invalid
    polygons are repaired with buffer(0); failures return a big penalty.
    """
    shape_pts = generate_waveguide_shape(shape_params)
    aligned = transform_shape(shape_pts, dx, dy, theta_rad)
    try:
        contour_poly = Polygon(np.asarray(contour_pts))
        shape_poly = Polygon(np.asarray(aligned))
        if not contour_poly.is_valid:
            contour_poly = contour_poly.buffer(0)
        if not shape_poly.is_valid:
            shape_poly = shape_poly.buffer(0)
        return contour_poly.union(shape_poly).area
    except Exception:
        return 1e12


def fit_waveguide_to_contours(contour_pts, init_shape_params):
    """
    STAGE 3b -- INNER placement search for ONE fixed set of shape parameters.

    Steps:
      1. Centroid pre-alignment: translate the shape so its centroid sits on
         the contour's centroid (a good dx,dy starting guess).
      2. Coarse ROTATION sweep at 0/90/180/270 deg. At each angle, run a small
         differential evolution over (dx,dy) (relative to the centroid guess)
         minimizing the union area. Keep the best (angle,dx,dy).  <-- this is
         the "optimize angle for overlap" part of the request.
      3. Final 3-D differential evolution over (dx,dy,theta) in a narrow box
         around that best, to polish the placement.

    Returns
    -------
    (hausdorff_distance, aligned_shape)
        hausdorff_distance : nm, worst-case boundary mismatch of the final fit
                             (used as the OUTER objective's score).
        aligned_shape      : (N,2) the placed parametric outline.
    """
    contour_pts = np.asarray(contour_pts)
    shape_pts = generate_waveguide_shape(init_shape_params)

    # 1) Centroid pre-alignment baseline.
    contour_centroid = Polygon(contour_pts).centroid
    shape_centroid = Polygon(shape_pts).centroid
    dx_baseline = contour_centroid.x - shape_centroid.x
    dy_baseline = contour_centroid.y - shape_centroid.y

    # Search radius derived from the largest edge in either polygon -- a scale
    # for "how far might we reasonably need to slide".
    max_edge_contour = max(np.linalg.norm(contour_pts[i] - contour_pts[i + 1])
                           for i in range(len(contour_pts) - 1))
    max_edge_shape = max(np.linalg.norm(shape_pts[i] - shape_pts[i + 1])
                         for i in range(len(shape_pts) - 1))
    max_distance = max(max_edge_contour, max_edge_shape)

    # 2) Coarse angle sweep with a (dx,dy) DE at each angle.
    best = {"angle_deg": None, "dx": None, "dy": None, "combined_area": np.inf}
    for angle_deg in range(0, 360, 90):
        theta_rad = np.deg2rad(angle_deg)

        def obj_dxdy(dxy):
            return _combined_area(contour_pts, init_shape_params,
                                  dx_baseline + dxy[0], dy_baseline + dxy[1], theta_rad)

        bounds = [(-max_distance, max_distance), (-max_distance, max_distance)]
        res = differential_evolution(obj_dxdy, bounds, strategy='best1bin',
                                     popsize=10, maxiter=50, tol=1e-3, polish=True, disp=False)
        if res.fun < best["combined_area"]:
            best.update(angle_deg=angle_deg,
                        dx=dx_baseline + res.x[0],
                        dy=dy_baseline + res.x[1],
                        combined_area=res.fun)

    dx_best, dy_best = best["dx"], best["dy"]
    theta_best = np.deg2rad(best["angle_deg"])

    # 3) Final 3-D polish over (dx,dy,theta), searching *deltas* around best.
    def obj_dxdytheta(v):
        return _combined_area(contour_pts, init_shape_params,
                              dx_best + v[0], dy_best + v[1], theta_best + v[2])

    res3 = differential_evolution(
        obj_dxdytheta,
        bounds=[(-0.2 * max_distance, 0.2 * max_distance),
                (-0.2 * max_distance, 0.2 * max_distance),
                (-np.deg2rad(90), np.deg2rad(90))],
        strategy='best1bin', popsize=10, maxiter=50, tol=1e-3, polish=True, disp=False)

    dx_f = dx_best + res3.x[0]
    dy_f = dy_best + res3.x[1]
    theta_f = theta_best + res3.x[2]

    aligned_shape = transform_shape(generate_waveguide_shape(init_shape_params), dx_f, dy_f, theta_f)

    # Score = Hausdorff distance (worst-case boundary gap) between the two rings.
    hausdorff = Polygon(aligned_shape).hausdorff_distance(Polygon(contour_pts))
    return hausdorff, aligned_shape


def _hausdorff_objective(shape_params, contour_pts):
    """
    Picklable OUTER objective (module-level so it survives multiprocessing).
    Runs the full inner placement search and returns just the Hausdorff score.
    """
    try:
        h, _ = fit_waveguide_to_contours(contour_pts, shape_params)
        return h
    except Exception:
        return 1e6   # penalize parameter vectors that blow up the geometry


def optimize_shape_params(contour_pts, nominal_shape_params, workers=None,
                          popsize=10, maxiter=50):
    """
    STAGE 3a -- OUTER search over the 7 shape parameters.

    Differential evolution over:
        W1..L2               : +/-20% of nominal
        length_waveguide     : [100, max x-extent of the contour]
    For each candidate it calls fit_waveguide_to_contours (the inner placement
    search) and minimizes the resulting Hausdorff distance. Runs in parallel
    across CPU cores.

    `nominal_shape_params` is the 7-vector in PARAM_NAMES order; the last entry
    (length_waveguide) is only used to size defaults -- its bound comes from the
    contour extent, not from the nominal value.

    Returns { 'best_shape_params': ndarray(7), 'best_hausdorff': float }.
    """
    contour_pts = np.asarray(contour_pts)
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)

    # length_waveguide upper bound = how wide the contour actually is.
    max_length = np.max(contour_pts[:, 0]) - np.min(contour_pts[:, 0])

    bounds = [(0.8 * p, 1.2 * p) for p in nominal_shape_params[:-1]]  # W1..L2 +/-20%
    bounds.append((100, max_length))                                  # length_waveguide

    obj = partial(_hausdorff_objective, contour_pts=contour_pts)
    result = differential_evolution(
        obj, bounds, strategy='best1bin', popsize=popsize, maxiter=maxiter,
        tol=1e-3, polish=False, disp=True,
        workers=workers, updating='deferred')  # 'deferred' is required for workers>1

    return {'best_shape_params': result.x, 'best_hausdorff': result.fun}


def calculate_length_waveguide(contour_pts, L1, L2, minimum=100.0) -> float:
    """
    Seed length_waveguide from geometry: (contour x-extent) - L1 - L2.
    This gives the outer search a sensible starting scale/bounds for the
    straight section. (From visual_creator.calculate_length_waveguide.)
    """
    max_distance = np.max(contour_pts[:, 0]) - np.min(contour_pts[:, 0])
    return max(max_distance - L1 - L2, minimum)


# =============================================================================
# STAGE 4 -- COMPARE AGAINST EXISTING (spatial registration onto a design GDS)
# =============================================================================
# Given the measured/fitted outline and an EXISTING reference layout (a GDS of
# what was intended), register the outline onto the design by MAXIMIZING the
# intersection area. This answers "where does this fabricated structure sit
# relative to the drawn design, and how well do they overlap?".
#
#   1. Center the outline on the design's (area-weighted) centroid.
#   2. Coarse sweep of `angle_steps` angles in [0,180]; per angle, DE over
#      (dx,dy) maximizing intersection area (note degrees here, not radians).
#   3. Final 3-D DE over (dx,dy,angle) around the best.
#   4. Write a combined GDS: design on layer 1, aligned outline on layer 2.
# =============================================================================

def _load_gds_polygons(gds_filename):
    """Load every polygon (as a list of (M,2) point arrays) from a GDS file."""
    lib = gdspy.GdsLibrary()
    lib.read_gds(gds_filename)
    cell = lib.top_level()[0]
    polys = []
    for P in cell.polygons:
        for pts in P.polygons:
            polys.append(np.asarray(pts).copy())
    return lib, polys


def _weighted_centroid(polygons):
    """Area-weighted centroid of a set of polygons (robust to holes/invalids)."""
    centroids, areas = [], []
    for pts in polygons:
        try:
            poly = Polygon(pts)
            if poly.is_valid:
                centroids.append(np.array(poly.centroid.coords[0]))
                areas.append(poly.area)
                continue
        except Exception:
            pass
        centroids.append(np.mean(pts, axis=0))   # fallback: vertex mean
        areas.append(1.0)
    if not centroids:
        return np.array([0.0, 0.0])
    total = sum(areas)
    if total > 0:
        return sum(c * a for c, a in zip(centroids, areas)) / total
    return np.mean(centroids, axis=0)


def superimpose_gds(gds_outline_filename, gds_design_filename, gds_output_filename,
                    angle_steps=37, pop_xy=15, iters_xy=50, pop_final=20, iters_final=100,
                    search_radius=2000.0):
    """
    STAGE 4. Spatially align an outline GDS onto a design GDS by maximizing
    overlap (intersection) area, and write a combined GDS.

    Parameters
    ----------
    angle_steps : number of angles tried in [0,180] (37 -> ~5 deg steps).
    pop_xy/iters_xy    : DE settings for the per-angle (dx,dy) search.
    pop_final/iters_final : DE settings for the final (dx,dy,angle) polish.
    search_radius : +/- box (nm) around the centroid guess for (dx,dy).

    Returns a dict with the final angle/dx/dy, best_overlap area, output path,
    and the centroid alignment info.
    (Original: utils.superimpose_gds.)
    """
    lib_o, outline_polys = _load_gds_polygons(gds_outline_filename)
    lib_d, design_polys = _load_gds_polygons(gds_design_filename)
    if not outline_polys or not design_polys:
        raise RuntimeError("No polygons found in one of the GDS files.")

    # Precompute the design as one geometry for fast intersection tests.
    design_union = unary_union([Polygon(p) for p in design_polys if Polygon(p).is_valid])

    # 1) Centroid alignment gives the initial (dx0, dy0).
    outline_centroid = _weighted_centroid(outline_polys)
    design_centroid = _weighted_centroid(design_polys)
    dx0, dy0 = design_centroid - outline_centroid
    print(f"  [compare] centroid offset dx0={dx0:.1f}, dy0={dy0:.1f}")

    def overlap_of(dx, dy, angle_deg):
        """Intersection area of the rotated+translated outline with the design."""
        theta = np.deg2rad(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        transformed = [Polygon(pts.dot(R.T) + [dx, dy]) for pts in outline_polys]
        transformed = [p for p in transformed if p.is_valid]
        if not transformed:
            return 0.0
        return unary_union(transformed).intersection(design_union).area

    # 2) Coarse angle grid; per angle, DE over (dx,dy) maximizing overlap.
    bounds_xy = [(dx0 - search_radius, dx0 + search_radius),
                 (dy0 - search_radius, dy0 + search_radius)]
    best = {"angle": None, "dx": None, "dy": None, "overlap": -1}
    for angle in np.linspace(0, 180, angle_steps):
        res = differential_evolution(lambda xy: -overlap_of(xy[0], xy[1], angle),
                                     bounds_xy, strategy='best1bin', popsize=pop_xy,
                                     maxiter=iters_xy, tol=1e-3, polish=True, init='sobol')
        ov = -res.fun
        if ov > best["overlap"]:
            best.update(angle=angle, dx=res.x[0], dy=res.x[1], overlap=ov)
            print(f"  [compare] new best @ {angle:.1f} deg: overlap={ov:.1f}")

    # 3) Final 3-D polish in a narrow box around the best.
    dax = day = 500.0    # +/- nm for dx, dy
    daa = 10.0           # +/- deg for angle
    bounds_final = [(best["dx"] - dax, best["dx"] + dax),
                    (best["dy"] - day, best["dy"] + day),
                    (max(0, best["angle"] - daa), min(180, best["angle"] + daa))]
    res3 = differential_evolution(lambda x: -overlap_of(x[0], x[1], x[2]),
                                  bounds_final, strategy='best1bin', popsize=pop_final,
                                  maxiter=iters_final, tol=1e-4, polish=True)
    dx_f, dy_f, ang_f = res3.x
    best_overlap = -res3.fun

    # 4) Write combined GDS: design (layer 1) + aligned outline (layer 2).
    lib_out = gdspy.GdsLibrary()
    rc = lib_out.new_cell("COMBINED")
    for poly in lib_d.top_level()[0].polygons:
        rc.add(poly)
    theta = np.deg2rad(ang_f)
    Rf = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for pts in outline_polys:
        rc.add(gdspy.Polygon(pts.dot(Rf.T) + [dx_f, dy_f], layer=2))
    lib_out.write_gds(gds_output_filename)

    print(f"  [compare] final angle={ang_f:.2f} deg, dx={dx_f:.1f}, dy={dy_f:.1f}, "
          f"overlap={best_overlap:.1f}  ->  {gds_output_filename}")
    return {"angle": ang_f, "dx": dx_f, "dy": dy_f, "best_overlap": best_overlap,
            "output_file": gds_output_filename,
            "centroid_offset": (float(dx0), float(dy0))}


# =============================================================================
# HELPERS -- save the fitted result as GDS + JSON
# =============================================================================

def save_fit_outline_gds(contour_nm, aligned_shape, out_gds):
    """
    Write the fit result as a GDS: measured contour on layer 1, fitted
    parametric outline on layer 2. This layer-1 outline is what Stage 4
    (superimpose_gds) expects as its `gds_outline_filename`.
    """
    lib = gdspy.GdsLibrary()
    cell = lib.new_cell("FIT_RESULT")
    cell.add(gdspy.Polygon(np.asarray(contour_nm), layer=1))
    cell.add(gdspy.Polygon(np.asarray(aligned_shape), layer=2))
    lib.write_gds(out_gds)
    return out_gds


def save_fit_json(out_json, sem_path, scale_info, nominal_params,
                  best_params, best_hausdorff, contour_nm, extra=None):
    """Persist the fitted parameters + provenance to JSON."""
    results = {
        'sem_image': sem_path,
        'scale_nm_per_px': scale_info['nm_per_px'] if scale_info else None,
        'scale_detection_info': scale_info,
        'nominal_params': dict(nominal_params),
        'best_hausdorff_distance': float(best_hausdorff),
        'parameter_names': PARAM_NAMES,
        'best_shape_params': [float(x) for x in best_params],
        'complete_params': {n: float(v) for n, v in zip(PARAM_NAMES, best_params)},
        'num_contour_points': int(len(contour_nm)),
        'timestamp': int(time.time()),
    }
    if extra:
        results.update(extra)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    return results


# =============================================================================
# ORCHESTRATOR -- run every stage end to end
# =============================================================================

DEFAULT_NOMINAL = {'W1': 300.0, 'W2': 400.0, 'W3': 500.0,
                   'W4': 600.0, 'L1': 200.0, 'L2': 300.0}


def run_full_pipeline(sem_path,
                      nominal_params=None,
                      sam_points=None,
                      sam_checkpoint="sam_vit_b_01ec64.pth",
                      nm_per_px=None,
                      text_box=DEFAULT_TEXT_BOX,
                      scale_box=DEFAULT_SCALE_BOX,
                      design_gds=None,
                      out_prefix=None,
                      outer_popsize=10,
                      outer_maxiter=50):
    """
    Run the whole pipeline:  SEM -> scale -> SAM cutout -> fit -> (compare).

    Parameters
    ----------
    sem_path       : path to the SEM image.
    nominal_params : dict with W1..L2 starting guesses (length_waveguide is
                     derived automatically). Defaults to DEFAULT_NOMINAL.
    sam_points     : list of [x_px,y_px] positive prompts. None -> interactive.
    sam_checkpoint : SAM weights file.
    nm_per_px      : skip Stage 1 by supplying a known scale (float). If None,
                     scale is auto-detected from text_box/scale_box.
    design_gds     : reference layout to compare against (Stage 4). None -> skip.
    out_prefix     : path prefix for outputs ("<prefix>.gds", ".json",
                     "_combined.gds"). None -> "sem_pipeline_<timestamp>".

    Returns a dict with contour, best params, Hausdorff, file paths, and
    (if design_gds given) the spatial comparison result.
    """
    nominal_params = dict(nominal_params or DEFAULT_NOMINAL)
    if out_prefix is None:
        out_prefix = f"sem_pipeline_{int(time.time())}"
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    # ---- STAGE 1: scale --------------------------------------------------
    if nm_per_px is None:
        print("[1/4] Detecting scale...")
        scale_info = detect_scale_nm_per_px(sem_path, text_box, scale_box)
        nm_per_px = scale_info['nm_per_px']
    else:
        print(f"[1/4] Using provided scale: {nm_per_px:.3f} nm/px")
        scale_info = {'nm_per_px': nm_per_px, 'length_nm': None,
                      'pixel_distance': None, 'raw_ocr': None}

    # ---- STAGE 2: SAM cutout --------------------------------------------
    print("[2/4] Segmenting structure with SAM...")
    contour_nm = sam_cutout(sem_path, nm_per_px, points=sam_points,
                            checkpoint=sam_checkpoint)

    # ---- STAGE 3: optimize fit ------------------------------------------
    print("[3/4] Optimizing shape parameters + placement...")
    length_wg = calculate_length_waveguide(contour_nm,
                                            nominal_params['L1'], nominal_params['L2'])
    nominal_vec = [nominal_params[n] for n in PARAM_NAMES[:-1]] + [length_wg]
    fit = optimize_shape_params(contour_nm, nominal_vec,
                                popsize=outer_popsize, maxiter=outer_maxiter)
    best_params = fit['best_shape_params']
    best_hausdorff = fit['best_hausdorff']
    # Recompute the placed outline for the winning parameters (for GDS output).
    _, aligned_shape = fit_waveguide_to_contours(contour_nm, best_params)
    print(f"      best Hausdorff = {best_hausdorff:.2f} nm")
    print("      fitted params: " +
          ", ".join(f"{n}={v:.1f}" for n, v in zip(PARAM_NAMES, best_params)))

    # ---- persist fit -----------------------------------------------------
    out_gds = f"{out_prefix}.gds"
    out_json = f"{out_prefix}.json"
    save_fit_outline_gds(contour_nm, aligned_shape, out_gds)
    save_fit_json(out_json, sem_path, scale_info, nominal_params,
                  best_params, best_hausdorff, contour_nm)

    result = {
        'contour_nm': contour_nm,
        'best_shape_params': best_params,
        'complete_params': {n: float(v) for n, v in zip(PARAM_NAMES, best_params)},
        'best_hausdorff': best_hausdorff,
        'aligned_shape': aligned_shape,
        'scale_info': scale_info,
        'fit_gds': out_gds,
        'fit_json': out_json,
    }

    # ---- STAGE 4: compare against existing design -----------------------
    if design_gds is not None:
        print("[4/4] Spatially comparing fit outline against existing design...")
        combined_gds = f"{out_prefix}_combined.gds"
        # superimpose_gds reads layer-1 polygons from the outline file (our fit
        # GDS stores the measured contour on layer 1) and registers onto design.
        compare = superimpose_gds(out_gds, design_gds, combined_gds)
        result['comparison'] = compare
        result['combined_gds'] = combined_gds
    else:
        print("[4/4] No design_gds provided -- skipping spatial comparison.")

    print("Done.")
    return result


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="SEM -> SAM cutout -> parametric fit -> spatial compare pipeline.")
    p.add_argument("sem_path", help="Path to the SEM image (.tif/.jpg/.png).")
    p.add_argument("--points", default=None,
                   help="SAM prompt points as 'x1,y1;x2,y2'. Omit for interactive clicking.")
    p.add_argument("--checkpoint", default="sam_vit_b_01ec64.pth", help="SAM weights .pth.")
    p.add_argument("--nm-per-px", type=float, default=None,
                   help="Known scale (nm/px); skips scale-bar OCR if given.")
    p.add_argument("--design-gds", default=None,
                   help="Existing design GDS to spatially compare against (Stage 4).")
    p.add_argument("--out-prefix", default=None, help="Output path prefix.")
    # Nominal shape parameters.
    for name, default in DEFAULT_NOMINAL.items():
        p.add_argument(f"--{name}", type=float, default=default, help=f"Nominal {name}.")
    return p


def main():
    args = _build_arg_parser().parse_args()
    points = None
    if args.points:
        points = [[float(v) for v in pair.split(",")] for pair in args.points.split(";")]
    nominal = {name: getattr(args, name) for name in DEFAULT_NOMINAL}
    run_full_pipeline(
        sem_path=args.sem_path,
        nominal_params=nominal,
        sam_points=points,
        sam_checkpoint=args.checkpoint,
        nm_per_px=args.nm_per_px,
        design_gds=args.design_gds,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()
