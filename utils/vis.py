import cv2
import numpy as np


# -------------------------------------------------
# Color palette (BGR for OpenCV)
# class 0 = background (not drawn)
# -------------------------------------------------
DEFAULT_PALETTE = np.array([
    [0,   0,   0],     # background (ignored)
    [0,   0, 255],     # lane 1 - red
    [0, 255,   0],     # lane 2 - green
    [255, 0,   0],     # lane 3 - blue
    [0, 255, 255],    # lane 4 - yellow
    [255, 0, 255],    # lane 5 - magenta
    [255, 255, 0],    # lane 6 - cyan
], dtype=np.uint8)


# -------------------------------------------------
# Safe colorization
# -------------------------------------------------
def colorize_mask(
    mask: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    palette: np.ndarray = DEFAULT_PALETTE,
):
    """
    mask: (H,W) int
    returns: (H,W,3) BGR
    """
    H, W = mask.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)

    pal = palette[:num_classes]

    for cls in range(1, num_classes):  # skip background
        colored[mask == cls] = pal[cls]

    # ignore_index → leave black
    return colored


# -------------------------------------------------
# Overlay segmentation on image
# -------------------------------------------------
def overlay_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    num_classes: int,
    alpha: float = 0.6,
    ignore_index: int = 255,
):
    """
    frame_bgr: (H,W,3) uint8
    mask: (H,W) int
    """
    assert frame_bgr.dtype == np.uint8
    assert mask.ndim == 2

    color = colorize_mask(
        mask,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    # only blend where lane pixels exist
    lane_pixels = (mask > 0) & (mask != ignore_index)

    out = frame_bgr.copy()

    out[lane_pixels] = cv2.addWeighted(
        frame_bgr[lane_pixels],
        1.0,
        color[lane_pixels],
        alpha,
        0
    )

    return out


# -------------------------------------------------
# Quick standalone debug
# -------------------------------------------------
def debug_visual(frame_bgr, logits, num_classes):
    """
    logits: (C,H,W) torch or numpy
    """
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()

    mask = logits.argmax(0).astype(np.uint8)
    return overlay_mask(frame_bgr, mask, num_classes)
