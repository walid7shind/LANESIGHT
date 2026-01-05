import cv2
import numpy as np

def colorize_mask(mask, num_classes):
    # mask: (H,W) int
    # simple palette
    palette = np.array([
        [0,0,0],
        [255,0,0],
        [0,255,0],
        [0,0,255],
        [255,255,0],
        [255,0,255],
        [0,255,255],
    ], dtype=np.uint8)
    pal = palette[:max(num_classes, 2)]
    return pal[mask]

def overlay(frame_bgr, mask, alpha=0.45, num_classes=5):
    # frame_bgr: (H,W,3), mask: (H,W)
    colored = colorize_mask(mask, num_classes=num_classes)   # RGB
    colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    out = cv2.addWeighted(frame_bgr, 1.0, colored, alpha, 0)
    return out
