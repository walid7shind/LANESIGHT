import cv2
import torch
import torchvision.transforms.functional as TF
import numpy as np
from models.hybrid_vit_unet import HybridViTUNet
from utils.vis import overlay

@torch.no_grad()
def run_video(input_path, output_path, ckpt="checkpoints/best.pt", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(ckpt, map_location="cpu")
    img_h, img_w = ck["img_size"]
    num_classes = ck["num_classes"]

    model = HybridViTUNet(num_classes=num_classes, img_size_hw=(img_h, img_w), base=48)
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W0, H0))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rs = cv2.resize(frame_rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        x = TF.to_tensor(frame_rs).unsqueeze(0).to(device)

        logits = model(x)                 # (1,K,H,W)
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # (H,W)

        # resize mask back to original frame size for overlay
        pred_big = cv2.resize(pred, (W0, H0), interpolation=cv2.INTER_NEAREST)
        vis = overlay(frame, pred_big, alpha=0.45, num_classes=num_classes)

        out.write(vis)

    cap.release()
    out.release()
    print("Saved:", output_path)

if __name__ == "__main__":
    run_video("input.mp4", "output_lanes.mp4")
