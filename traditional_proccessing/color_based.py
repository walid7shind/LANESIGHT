"""
lane_color_weighting.py
-----------------------

Soft color-based weighting for traditional lane detection.
Uses CIE Lab space to bias edge detection toward lane-like colors
(white / yellow) while suppressing road-asphalt edges.

This module DOES NOT perform detection.
It provides confidence weights to be used as priors.
"""

from __future__ import annotations
import numpy as np
import cv2


# ============================================================
# Color anchors in Lab space (empirical, camera-agnostic)
# ============================================================

# OpenCV Lab ranges:
# L in [0,255], a,b in [0,255] with 128 as neutral

LANE_WHITE_LAB = np.array([200, 128, 128], dtype=np.float32)
LANE_YELLOW_LAB = np.array([180, 135, 160], dtype=np.float32)


# ============================================================
# Utilities
# ============================================================

def bgr_to_lab(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to Lab (float32).
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    return lab.astype(np.float32)


def estimate_road_lab(
    lab_img: np.ndarray,
    h_frac: tuple[float, float] = (0.75, 1.0),
    w_frac: tuple[float, float] = (0.30, 0.70),
) -> np.ndarray:
    """
    Estimate average road color from a bottom-center region.

    Parameters
    ----------
    lab_img : (H,W,3) Lab image
    h_frac  : vertical sampling fraction (start, end)
    w_frac  : horizontal sampling fraction (start, end)

    Returns
    -------
    mu_road : (3,) mean Lab color
    """
    H, W = lab_img.shape[:2]

    y0 = int(h_frac[0] * H)
    y1 = int(h_frac[1] * H)
    x0 = int(w_frac[0] * W)
    x1 = int(w_frac[1] * W)

    roi = lab_img[y0:y1, x0:x1]
    if roi.size == 0:
        return np.array([128, 128, 128], dtype=np.float32)

    return np.mean(roi.reshape(-1, 3), axis=0)


# ============================================================
# Core weighting logic
# ============================================================

def _gaussian_distance(d: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


def lane_color_weight_map(
    lab_img: np.ndarray,
    mu_road: np.ndarray,
    sigma_lane: float = 25.0,
    sigma_road: float = 40.0,
) -> np.ndarray:
    """
    Compute a soft per-pixel lane confidence weight ∈ [0,1].

    Weight increases when:
    - pixel is close to white OR yellow lane color
    - pixel is far from road color

    Parameters
    ----------
    lab_img    : (H,W,3) Lab image
    mu_road    : (3,) mean road Lab color
    sigma_lane : controls tolerance to lane color variation
    sigma_road : controls suppression of road-like colors

    Returns
    -------
    weight_map : (H,W) float32 in [0,1]
    """

    # Distances to lane anchors
    d_white = np.linalg.norm(lab_img - LANE_WHITE_LAB, axis=2)
    d_yellow = np.linalg.norm(lab_img - LANE_YELLOW_LAB, axis=2)
    d_lane = np.minimum(d_white, d_yellow)

    # Distance to road color
    d_road = np.linalg.norm(lab_img - mu_road, axis=2)

    w_lane = _gaussian_distance(d_lane, sigma_lane)
    w_road = _gaussian_distance(d_road, sigma_road)

    weight = w_lane * (1.0 - w_road)
    return np.clip(weight, 0.0, 1.0)


# ============================================================
# Integration helpers
# ============================================================

def apply_weight_to_edges(
    edges: np.ndarray,
    weight_map: np.ndarray,
    min_weight: float = 0.2,
) -> np.ndarray:
    """
    Apply color weighting to an edge map.

    Parameters
    ----------
    edges       : (H,W) uint8 edge image (Canny/Sobel)
    weight_map  : (H,W) float32 ∈ [0,1]
    min_weight  : lower clamp to avoid total suppression

    Returns
    -------
    weighted_edges : (H,W) uint8
    """
    w = np.clip(weight_map, min_weight, 1.0)
    out = edges.astype(np.float32) * w
    return np.clip(out, 0, 255).astype(np.uint8)


def compute_lane_weighted_edges(
    image_bgr: np.ndarray,
    edges: np.ndarray,
    sigma_lane: float = 25.0,
    sigma_road: float = 40.0,
) -> np.ndarray:
    """
    One-call helper:
    image → Lab → road estimation → weight → weighted edges
    """
    lab = bgr_to_lab(image_bgr)
    mu_road = estimate_road_lab(lab)
    weight = lane_color_weight_map(
        lab,
        mu_road,
        sigma_lane=sigma_lane,
        sigma_road=sigma_road,
    )
    return apply_weight_to_edges(edges, weight)


# ============================================================
# Debug visualization
# ============================================================

def visualize_weight_map(weight_map: np.ndarray) -> np.ndarray:
    """
    Convert weight map to a heatmap for visualization.
    """
    w = (255.0 * weight_map).astype(np.uint8)
    return cv2.applyColorMap(w, cv2.COLORMAP_JET)


def split_and_score_lines(lines, min_slope=0.5):
    left = []
    right = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue

        slope = dy / dx
        if abs(slope) < min_slope:
            continue

        length = np.hypot(dx, dy)
        intercept = y1 - slope * x1

        if slope < 0:
            left.append((slope, intercept, length))
        else:
            right.append((slope, intercept, length))

    return left, right
def weighted_fit(lines):
    if not lines:
        return None

    slopes = np.array([l[0] for l in lines])
    intercepts = np.array([l[1] for l in lines])
    weights = np.array([l[2] for l in lines])  # lengths

    m = np.average(slopes, weights=weights)
    c = np.average(intercepts, weights=weights)
    return m, c
