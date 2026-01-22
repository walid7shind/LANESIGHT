from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

Array = np.ndarray


def _line_kernel(length: int, angle_deg: float) -> Array:
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    t = np.deg2rad(angle_deg)
    dx, dy = np.cos(t), np.sin(t)
    cv2.line(
        k,
        (int(c - dx * c), int(c - dy * c)),
        (int(c + dx * c), int(c + dy * c)),
        1,
        1,
    )
    return k


def _skeletonize(bin_img: Array) -> Array:
    img = (bin_img > 0).astype(np.uint8) * 255
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(img)
    skel = np.zeros_like(img)
    el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        er = cv2.erode(img, el)
        tmp = cv2.dilate(er, el)
        tmp = cv2.subtract(img, tmp)
        skel |= tmp
        img = er
        if cv2.countNonZero(img) == 0:
            break
    return skel

def filter_components_by_geometry(
    bin_img: np.ndarray,
    *,
    min_area: int = 150,
    min_elongation: float = 4.0,
    angle_ranges: tuple = ((20, 85), (-85, -20)),  # degrees
) -> np.ndarray:
    """
    Keep only elongated, non-horizontal connected components
    with orientation inside given angle ranges.
    """
    bin01 = (bin_img > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)

    out = np.zeros_like(bin01, dtype=np.uint8)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x, y, w, h, _ = stats[i]
        elong = max(h / max(w, 1), w / max(h, 1))
        if elong < min_elongation:
            continue

        ys, xs = np.where(labels == i)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)

        if pts.shape[0] < 10:
            continue

        # PCA for orientation
        mean = pts.mean(axis=0)
        pts0 = pts - mean
        cov = pts0.T @ pts0
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argmax(eigvals)]

        angle = np.degrees(np.arctan2(direction[1], direction[0]))

        keep = False
        for a_min, a_max in angle_ranges:
            if a_min <= angle <= a_max:
                keep = True
                break

        if not keep:
            continue

        out[labels == i] = 1

    return out.astype(np.uint8) * 255

def _connected_components_filter(bin_img: Array, min_area: int, min_elong: float) -> Array:
    b = (bin_img > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    out = np.zeros_like(b)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        e = max(h / max(w, 1), w / max(h, 1))
        if e < min_elong:
            continue
        out[lab == i] = 1
    return out * 255


def directional_closing(bin_img: np.ndarray, k: int = 17) -> np.ndarray:
    angles = [75, 85, 95, 105]
    out = bin_img.copy()

    for a in angles:
        kernel = np.zeros((k, k), np.uint8)
        c = k // 2
        rad = np.deg2rad(a)
        dx, dy = np.cos(rad), np.sin(rad)
        cv2.line(
            kernel,
            (int(c - dx * c), int(c - dy * c)),
            (int(c + dx * c), int(c + dy * c)),
            1,
            1,
        )
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)

    return out



def _second_diff_smooth_1d(x: Array, lam: float) -> Array:
    n = x.size
    if n < 5:
        return x
    D2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        D2[i, i:i + 3] = [1, -2, 1]
    A = np.eye(n) + lam * (D2.T @ D2)
    return np.linalg.solve(A, x)


def _extract_components(skel: Array, min_pts: int) -> List[Array]:
    b = (skel > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_pts:
            continue
        ys, xs = np.where(lab == i)
        out.append(np.stack([xs, ys], 1))
    return out


def _render_polyline(pts: Array, H: int, W: int, y_step: int, lam: float, th: int, y_min_frac: float) -> Array:
    out = np.zeros((H, W), np.uint8)
    y_min = int(H * y_min_frac)
    pts = pts[pts[:, 1] >= y_min]
    if pts.shape[0] < 20:
        return out
    xs, ys = pts[:, 0], pts[:, 1]
    bins = np.arange(y_min, H, y_step)
    xm, yu = [], []
    for y0 in bins:
        m = (ys >= y0) & (ys < y0 + y_step)
        if not np.any(m):
            continue
        xm.append(np.median(xs[m]))
        yu.append(y0 + y_step / 2)
    if len(xm) < 10:
        return out
    xm = _second_diff_smooth_1d(np.array(xm, np.float32), lam)
    poly = np.stack([xm, np.array(yu)], 1).astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
    cv2.polylines(out, [poly], False, 255, th, cv2.LINE_AA)
    return out


@dataclass
class PostProcessConfig:
    min_area_frac: float = 0.0005
    min_elongation: float = 1.0
    closing_k: int = 13
    closing_angles: Tuple[float, ...] = (90.0, 80.0, 100.0)
    skel_min_points: int = 20
    y_step: int = 4
    curvature_lam: float = 60.0
    thickness: int = 10
    y_min_frac: float = 0.14


class LanePostProcessor:
    def __init__(self, cfg: PostProcessConfig):
        self.cfg = cfg

    def process(self, bin_img: Array) -> Array:
        """Post-process a binary lane mask.

        Accepts masks in {0,1} or {0,255} (or bool), returns uint8 {0,255}.
        """
        if bin_img.dtype != np.uint8:
            bin_img = bin_img.astype(np.uint8)
        if bin_img.max() <= 1:
            bin_img = bin_img * 255

        H, W = bin_img.shape
        min_area = int(self.cfg.min_area_frac * H * W)

        # 1) Geometry filter first (remove blobs / wrong directions)
        b1 = filter_components_by_geometry(
            (bin_img > 0).astype(np.uint8) * 255,
            min_area=max(10, min_area),
            min_elongation=float(self.cfg.min_elongation),
        )

        # 2) Skeletonize + polyline reconstruction
        sk = _skeletonize(b1)
        comps = _extract_components(sk, self.cfg.skel_min_points)
        out = np.zeros((H, W), np.uint8)
        for c in comps:
            out |= _render_polyline(
                c, H, W,
                self.cfg.y_step,
                self.cfg.curvature_lam,
                self.cfg.thickness,
                self.cfg.y_min_frac,
            )

        # 3) Directional closing last (connect final rendered lanes)
        out = directional_closing(out, k=int(self.cfg.closing_k))
        return out

