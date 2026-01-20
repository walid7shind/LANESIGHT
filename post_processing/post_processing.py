from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

Array = np.ndarray
TorchTensor = Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CKPT_PATH = ROOT / "checkpoints" / "best.pt"
BASE_DIR = ROOT / "test_output"  # base frames (no predicted lanes)
OUT_DIR = ROOT / "test_images"

# Outputs (keep both spellings; user asked for overlay_proccessed.png)
OUT_OVERLAY = OUT_DIR / "overlay_proccessed.png"
OUT_OVERLAY_ALT = OUT_DIR / "overlay_processed.png"
OUT_MASK_RAW = OUT_DIR / "lane_mask_raw.png"
OUT_MASK_PP = OUT_DIR / "lane_mask_pp.png"


def _pick_base_image_path() -> Path:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    imgs = [p for p in BASE_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not imgs:
        raise FileNotFoundError(f"No base images found in: {BASE_DIR}")
    imgs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return imgs[0]


def _load_lane_model(*, device: str):
    import torch
    from models.hybrid_vit_unet import HybridViTUNet

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=device)
    img_size = ckpt["img_size"]  # (H,W)
    num_classes = int(ckpt["num_classes"])

    model = HybridViTUNet(num_classes=num_classes, img_size_hw=img_size, base=48).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, img_size, num_classes


def _infer_lane_mask_argmax(model, *, img_bgr: Array, img_size_hw: Tuple[int, int], device: str) -> Array:
    """Return raw lane mask (uint8 0/255) at original image resolution."""
    import torch

    H0, W0 = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rs = cv2.resize(img_rgb, (int(img_size_hw[1]), int(img_size_hw[0])), interpolation=cv2.INTER_LINEAR)

    x = (img_rs.astype(np.float32) / 255.0)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1)[0].detach().cpu().numpy().astype(np.uint8)

    # lane class assumed to be 1 (as in tst.py)
    mask = (pred == 1).astype(np.uint8) * 255
    mask = cv2.resize(mask, (W0, H0), interpolation=cv2.INTER_NEAREST)
    return mask


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=None, help="Path to base image (no predicted lanes). Defaults to latest in test_output/.")
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'. Default: auto")
    ap.add_argument("--show", action="store_true", help="Show raw/pp masks in OpenCV windows.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base) if args.base else _pick_base_image_path()
    base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
    if base is None:
        raise FileNotFoundError(base_path)

    # Device selection
    device: str
    if args.device:
        device = args.device
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    model, img_size_hw, _ = _load_lane_model(device=device)

    # 1) Model raw mask
    mask_raw = _infer_lane_mask_argmax(model, img_bgr=base, img_size_hw=img_size_hw, device=device)
    cv2.imwrite(str(OUT_MASK_RAW), mask_raw)

    # 2) Post-process the mask
    pp = LanePostProcessor(PostProcessConfig())
    mask_pp = pp.process(mask_raw)
    cv2.imwrite(str(OUT_MASK_PP), mask_pp)

    # 3) Overlay onto base
    overlay = base.copy()
    overlay[mask_pp > 0] = (0, 0, 255)
    out = cv2.addWeighted(base, 1.0, overlay, 0.60, 0)
    cv2.imwrite(str(OUT_OVERLAY), out)
    cv2.imwrite(str(OUT_OVERLAY_ALT), out)

    if args.show:
        cv2.imshow("mask_raw", mask_raw)
        cv2.imshow("mask_pp", mask_pp)
        cv2.imshow("overlay", out)
        cv2.waitKey(0)

    print("Base:", base_path)
    print("Saved:")
    print(" -", OUT_MASK_RAW)
    print(" -", OUT_MASK_PP)
    print(" -", OUT_OVERLAY)
    print(" -", OUT_OVERLAY_ALT)


if __name__ == "__main__":
    main()
