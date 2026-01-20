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
FRAMES_DIR = BASE_DIR / "test_images"
OUT_DIR = BASE_DIR / "test_output"

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
    if not img_paths:
        video_candidates = sorted(
            p
            for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv")
            for p in FRAMES_DIR.glob(ext)
        )
        if len(video_candidates) == 1:
            video_path = video_candidates[0]
            print(f"No images found; using video: {video_path}")
        elif len(video_candidates) > 1:
            video_path = video_candidates[0]
            print(
                f"No images found; multiple videos found, using first: {video_path}"
            )
        else:
            raise AssertionError(f"No images or videos found in {FRAMES_DIR}")
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

def _process_bgr(frame_bgr: np.ndarray, index: int) -> np.ndarray:
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

    return final_overlay



# =================================================
# PROCESS (images or video)
# =================================================
# =================================================
# PROCESS VIDEO → VIDEO (10 FPS)
# =================================================
if video_path is not None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_out = 10.0

    out_path = OUT_DIR / "output_10fps.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path),
        fourcc,
        fps_out,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open")

    frame_idx = 0
    written_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_in = frame_idx / fps_in
        t_out = written_frames / fps_out

        if t_in >= t_out:
            processed = _process_bgr(frame, written_frames)
            writer.write(processed)
            written_frames += 1

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"✔ Video written: {out_path}")

else:
    # PROCESS IMAGES → IMAGES
    for i, img_path in enumerate(img_paths):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Skipping unreadable image: {img_path}")
            continue
        processed = _process_bgr(frame, i)
        out_path = OUT_DIR / f"{img_path.stem}_out.png"
        cv2.imwrite(str(out_path), processed)

    print(f"✔ Images written to: {OUT_DIR}")
