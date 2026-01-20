import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from models.hybrid_vit_unet import HybridViTUNet

# =================================================
# PATHS
# =================================================
BASE_DIR = Path(__file__).resolve().parent

CKPT_PATH = BASE_DIR / "checkpoints" / "best.pt"
FRAMES_DIR = BASE_DIR / "test_output"
OUT_DIR = BASE_DIR / "test_images"

# If FRAMES_DIR is a video file, process every Nth frame.
FRAME_STRIDE = 5

USE_CUDA = torch.cuda.is_available()
TORCH_DEVICE = "cuda" if USE_CUDA else "cpu"
# Ultralytics accepts device as int GPU index or a string like 'cpu'.
YOLO_DEVICE = 0 if USE_CUDA else "cpu"

# Back-compat variable (used by some calls)
DEVICE = TORCH_DEVICE

# =================================================
# LOAD LANE MODEL
# =================================================
ckpt = torch.load(CKPT_PATH, map_location=TORCH_DEVICE)
img_size = ckpt["img_size"]
num_classes = ckpt["num_classes"]

lane_model = HybridViTUNet(
    num_classes=num_classes,
    img_size_hw=img_size,
    base=48
).to(TORCH_DEVICE)

lane_model.load_state_dict(ckpt["model"])
lane_model.eval()

# =================================================
# LOAD YOLO
# =================================================
yolo = YOLO("yolov8n.pt")  # car + motorcycle from COCO

# =================================================
# IO
# =================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)

img_paths = []
video_path = None

if FRAMES_DIR.is_dir():
    img_paths = sorted(
        p
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for p in FRAMES_DIR.glob(ext)
    )
    assert img_paths, f"No images found in {FRAMES_DIR}"
elif FRAMES_DIR.is_file():
    video_path = FRAMES_DIR
else:
    raise FileNotFoundError(f"Input path does not exist: {FRAMES_DIR}")

# =================================================
# PALETTE
# =================================================
palette = np.array([
    [0,   0,   0],    # background
    [255, 0,   0],    # lane
], dtype=np.uint8)

def _process_bgr(frame_bgr: np.ndarray, index: int) -> None:
    orig_h, orig_w = frame_bgr.shape[:2]

    # ---------- LANE SEG ----------
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rs = cv2.resize(frame_rgb, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)

    x = frame_rs.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(TORCH_DEVICE)

    with torch.no_grad():
        logits = lane_model(x)
        mask = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

    color_mask = palette[mask]
    color_mask = cv2.resize(color_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    lane_overlay = cv2.addWeighted(frame_bgr, 0.7, color_mask, 0.8, 0)

    # ---------- YOLO (cars only) ----------
    results = yolo(
        frame_bgr,
        conf=0.3,
        classes=[2, 3],  # car, motorcycle
        device=YOLO_DEVICE,
        verbose=False,
    )
    final_overlay = results[0].plot(img=lane_overlay)

    out_path = OUT_DIR / f"overlay_.png"
    cv2.imwrite(str(out_path), final_overlay)


# =================================================
# PROCESS (images or video)
# =================================================
if video_path is not None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    i = 0
    out_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if FRAME_STRIDE > 1 and (i % FRAME_STRIDE != 0):
            i += 1
            continue
        _process_bgr(frame, out_idx)
        out_idx += 1
        i += 1

    cap.release()
else:
    for i, img_path in enumerate(img_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        _process_bgr(img_bgr, i)

print(f"✔ Done. Wrote overlays to: {OUT_DIR}")
