import cv2
import torch
import numpy as np

from models.hybrid_vit_unet import HybridViTUNet

# -------------------------
# CONFIG
# -------------------------
CKPT_PATH = "checkpoints/best.pt"
IMG_PATH  = r"C:\Users\walid\dev\vision\LANESIGHT\data\driver_23_30frame\05161653_0627.MP4\02155.jpg"
OUT_PATH  = r"C:\Users\walid\dev\vision\LANESIGHT\test_output\overlay.png"

DEVICE = "cpu"   # IMPORTANT (RTX 5060 not supported yet)

# -------------------------
# LOAD CHECKPOINT
# -------------------------
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
img_size = ckpt["img_size"]
num_classes = ckpt["num_classes"]

model = HybridViTUNet(
    num_classes=num_classes,
    img_size_hw=img_size,
    base=48
)
model.load_state_dict(ckpt["model"])
model.eval()

# -------------------------
# LOAD IMAGE
# -------------------------
img_bgr = cv2.imread(IMG_PATH)
assert img_bgr is not None, "Image not found"

orig_h, orig_w = img_bgr.shape[:2]

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_rgb = cv2.resize(img_rgb, (img_size[1], img_size[0]))

img = img_rgb.astype(np.float32) / 255.0
img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

# -------------------------
# INFERENCE
# -------------------------
with torch.no_grad():
    logits = model(img)
    pred = torch.argmax(logits, dim=1)[0].numpy().astype(np.uint8)

# -------------------------
# COLORIZE MASK
# -------------------------
palette = np.array([
    [0,   0,   0],    # background
    [255, 0,   0],    # lane 1
    [0,   255, 0],    # lane 2
    [0,   0,   255],  # lane 3
    [255, 255, 0],    # lane 4
], dtype=np.uint8)

color_mask = palette[pred]
color_mask = cv2.resize(color_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

# -------------------------
# OVERLAY
# -------------------------
overlay = cv2.addWeighted(img_bgr, 0.7, color_mask, 0.8, 0)
cv2.imwrite(OUT_PATH, overlay)

print("Saved:", OUT_PATH)
print("Predicted classes:", np.unique(pred))
