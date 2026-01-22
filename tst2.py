import cv2
import numpy as np
from pathlib import Path

# =================================================
# PATHS
# =================================================
BASE_DIR = Path(__file__).resolve().parent
TEST_IMAGES = BASE_DIR / "test_images"
OUT_DIR = BASE_DIR / "test_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# pick first video
videos = sorted(TEST_IMAGES.glob("*.mp4"))
if not videos:
    raise FileNotFoundError("No mp4 found in test_images/")
VIDEO_PATH = videos[0]

OUT_VIDEO = OUT_DIR / "fusion_10fps.mp4"

# =================================================
# IMPORT WRAPPERS
# =================================================
from inference.ufld_wrapper import UFLDInferencer
from inference.vit_hybrid_wrapper import ViTLaneInferencer
from inference.yolo_wrapper import YOLOInferencer
from inference.fusion import fuse_lanes, lanes_to_mask
from traditional_proccessing.taditionalBS import traditional_polygon_mask

# =================================================
# INIT MODELS
# =================================================
# ---- UFLD ----
VISION_ROOT = BASE_DIR.parent
UFLD_ROOT = VISION_ROOT / "UFLD" / "Ultra-Fast-Lane-Detection"

ufld = UFLDInferencer(
    weights_path=UFLD_ROOT / "weights" / "culane_18.pth",
    config_path=UFLD_ROOT / "configs" / "culane.py",
)

# ---- ViT ----
vit = ViTLaneInferencer(
    ckpt_path=BASE_DIR / "checkpoints" / "best.pt"
)

# ---- YOLO ----
yolo = YOLOInferencer(
    weights="yolov8n.pt",
    classes=[2, 3],  # car, motorcycle
    conf_thres=0.3,
)

# =================================================
# VIDEO IO
# =================================================
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError("Failed to open video")

fps_in = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps_out = 10.0

writer = cv2.VideoWriter(
    str(OUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_out,
    (w, h),
)

frame_idx = 0
written = 0

# =================================================
# PROCESS LOOP
# =================================================
while True:
    ok, frame = cap.read()
    if not ok:
        break

    t_in = frame_idx / fps_in
    t_out = written / fps_out

    if t_in >= t_out:
        # ----------------------------
        # Run models
        # ----------------------------
        ufld_lanes = ufld.infer_frame(frame)
        vit_prob, vit_mask = vit.infer_frame(frame)
        yolo_boxes = yolo.infer_frame(frame)

        # ----------------------------
        # Fuse
        # ----------------------------
        fused = fuse_lanes(
            ufld_lanes,
            vit_prob_map=vit_prob,
            yolo_boxes=yolo_boxes,
        )

        fused_mask = lanes_to_mask(fused, frame.shape[:2])

        # Traditional polygon mask (uint8 {0,1})
        roi_mask = traditional_polygon_mask(frame)

        # ----------------------------
        # Combine masks
        # ----------------------------
        combined_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # ViT mask → blue
        combined_mask[vit_mask == 1] = (255, 0, 0)

        # UFLD fused lanes → green
        combined_mask[fused_mask == 1] = (0, 255, 0)

        # ----------------------------
        # Overlay
        # ----------------------------
        overlay = cv2.addWeighted(frame, 0.7, combined_mask, 0.6, 0.0)

        # Traditional polygon overlay (green, low alpha)
        roi_color = np.zeros_like(frame)
        roi_color[roi_mask == 1] = (0, 255, 0)
        overlay = cv2.addWeighted(overlay, 1.0, roi_color, 0.20, 0.0)

        writer.write(overlay)
        written += 1

    frame_idx += 1

cap.release()
writer.release()

print(f"✔ Output written to {OUT_VIDEO}")
