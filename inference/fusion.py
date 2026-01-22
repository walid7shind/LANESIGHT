import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ============================================================
# Data structures
# ============================================================

@dataclass
class Lane:
    """
    Unified lane representation.
    """
    points: np.ndarray        # (N,2) int image coordinates
    confidence: float         # [0,1]
    source: str               # "ufld", "vit", "fused"


@dataclass
class LaneSet:
    lanes: List[Lane]


# ============================================================
# Utility functions
# ============================================================

def sample_prob_along_polyline(
    prob_map: np.ndarray,
    polyline: np.ndarray,
    radius: int = 2,
) -> float:
    """
    Sample probability values along a polyline from a probability map.

    Returns mean probability in a small band around the polyline.
    """
    h, w = prob_map.shape
    values = []

    for (x, y) in polyline:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                xx = int(np.clip(x + dx, 0, w - 1))
                yy = int(np.clip(y + dy, 0, h - 1))
                values.append(prob_map[yy, xx])

    if not values:
        return 0.0

    return float(np.mean(values))


def intersects_boxes(
    polyline: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    margin: int = 5,
) -> bool:
    """
    Check if a polyline intersects any bounding box.
    """
    for (x1, y1, x2, y2) in boxes:
        x1 -= margin
        y1 -= margin
        x2 += margin
        y2 += margin

        for (x, y) in polyline:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True

    return False


def resample_polyline(polyline: np.ndarray, num: int = 50) -> np.ndarray:
    """
    Resample a polyline to a fixed number of points (arc-length based).
    """
    if len(polyline) < 2:
        return polyline

    d = np.sqrt(np.sum(np.diff(polyline, axis=0) ** 2, axis=1))
    s = np.insert(np.cumsum(d), 0, 0)
    s /= s[-1]

    t = np.linspace(0, 1, num)
    x = np.interp(t, s, polyline[:, 0])
    y = np.interp(t, s, polyline[:, 1])

    return np.stack([x, y], axis=1).astype(np.int32)


# ============================================================
# Fusion core
# ============================================================

def fuse_lanes(
    ufld_lanes: List[np.ndarray],
    vit_prob_map: Optional[np.ndarray] = None,
    yolo_boxes: Optional[List[dict]] = None,
    *,
    min_vit_support: float = 0.15,
    reject_on_yolo: bool = True,
) -> LaneSet:
    """
    Fuse UFLD geometry with ViT semantics and YOLO obstacles.

    Args:
        ufld_lanes: list of polylines from UFLD [(N,2), ...]
        vit_prob_map: (H,W) lane probability map from ViT
        yolo_boxes: list of YOLO detections (dicts with 'xyxy')
        min_vit_support: minimum mean ViT probability to keep a lane
        reject_on_yolo: reject lanes crossing vehicles

    Returns:
        LaneSet
    """

    fused: List[Lane] = []

    # Extract YOLO boxes if present
    boxes = []
    if yolo_boxes:
        for d in yolo_boxes:
            boxes.append(tuple(d["xyxy"]))

    for poly in ufld_lanes:
        poly = resample_polyline(poly, num=60)

        # ----------------------------
        # ViT semantic support
        # ----------------------------
        if vit_prob_map is not None:
            vit_conf = sample_prob_along_polyline(vit_prob_map, poly)
        else:
            vit_conf = 1.0  # geometry-only fallback

        if vit_conf < min_vit_support:
            continue

        # ----------------------------
        # YOLO rejection
        # ----------------------------
        if reject_on_yolo and boxes:
            if intersects_boxes(poly, boxes):
                continue

        # ----------------------------
        # Final confidence
        # ----------------------------
        confidence = vit_conf

        fused.append(
            Lane(
                points=poly,
                confidence=confidence,
                source="fused",
            )
        )

    return LaneSet(fused)


# ============================================================
# Optional helper: build mask from fused lanes
# ============================================================

def lanes_to_mask(
    lanes: LaneSet,
    shape: Tuple[int, int],
    thickness: int = 6,
) -> np.ndarray:
    """
    Rasterize fused lanes into a binary mask.

    Args:
        lanes: LaneSet
        shape: (H,W)
        thickness: line thickness

    Returns:
        mask: (H,W) uint8 {0,1}
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for lane in lanes.lanes:
        pts = lane.points.reshape(-1, 1, 2)
        cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=thickness)

    return mask
