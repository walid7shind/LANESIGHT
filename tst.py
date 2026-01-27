"""Evaluate the lane segmentation model checkpoint against CULane-style labels.

This script is tailored to your workspace layout:
  - frames: LANESIGHT/data/driver_37_30frame/<VIDEO>.MP4/*.jpg
  - masks : LANESIGHT/data/laneseg_label_w16_test/driver_37_30frame/<VIDEO>.MP4/*.png

It runs the lane model (best.pt) on two sequences and outputs:
  - per-frame MSE graph (probability map vs GT mask)
  - mean-metric bar chart (IoU, Dice, Precision, Recall)
  - a CSV with per-frame metrics
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


def _get_plt(show: bool):
    """Import pyplot with a suitable backend.

    If show=False, force a non-interactive backend so this works headlessly.
    If show=True, keep matplotlib's default backend so figures can display.
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

from inference.vit_hybrid_wrapper import ViTLaneInferencer


@dataclass(frozen=True)
class FrameMetrics:
    frame_id: str
    mse: float
    iou: float
    dice: float
    precision: float
    recall: float


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def _compute_metrics(
    prob: np.ndarray,
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    prob_f = prob.astype(np.float32)
    gt_f = gt_bin.astype(np.float32)
    mse = float(np.mean((prob_f - gt_f) ** 2))

    pred = (pred_bin > 0).astype(np.uint8)
    gt = (gt_bin > 0).astype(np.uint8)

    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))

    inter = tp
    union = tp + fp + fn

    iou = _safe_div(inter, union)
    dice = _safe_div(2 * inter, (2 * inter + fp + fn))
    precision = _safe_div(tp, (tp + fp))
    recall = _safe_div(tp, (tp + fn))

    return mse, iou, dice, precision, recall


def _iter_frames(frames_dir: Path) -> List[Path]:
    return sorted(frames_dir.glob("*.jpg"))


def evaluate_sequence(
    inferencer: ViTLaneInferencer,
    frames_dir: Path,
    masks_dir: Path,
    threshold: float,
    max_frames: int | None = None,
) -> List[FrameMetrics]:
    frame_paths = _iter_frames(frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in: {frames_dir}")

    metrics: List[FrameMetrics] = []

    for i, img_path in enumerate(frame_paths):
        if max_frames is not None and i >= max_frames:
            break

        mask_path = masks_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing GT mask for frame {img_path.name}: {mask_path}")

        frame_bgr = cv2.imread(str(img_path))
        if frame_bgr is None:
            raise RuntimeError(f"Failed to read frame: {img_path}")

        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        gt_bin = (gt > 0).astype(np.uint8)

        prob, pred_bin = inferencer.infer_frame(frame_bgr, threshold=threshold)

        mse, iou, dice, precision, recall = _compute_metrics(prob, pred_bin, gt_bin)
        metrics.append(
            FrameMetrics(
                frame_id=img_path.stem,
                mse=mse,
                iou=iou,
                dice=dice,
                precision=precision,
                recall=recall,
            )
        )

    return metrics


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def write_csv(out_csv: Path, per_video: Dict[str, List[FrameMetrics]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_id", "mse", "iou", "dice", "precision", "recall"])
        for video_name, rows in per_video.items():
            for r in rows:
                w.writerow([video_name, r.frame_id, r.mse, r.iou, r.dice, r.precision, r.recall])


def plot_mse_over_frames(out_path: Path, per_video: Dict[str, List[FrameMetrics]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt = _get_plt(show=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    for video_name, rows in per_video.items():
        xs = np.arange(len(rows))
        ys = np.array([r.mse for r in rows], dtype=np.float32)
        ax.plot(xs, ys, linewidth=1.2, label=video_name)

    ax.set_title("Lane Segmentation MSE over frames")
    ax.set_xlabel("Frame index (in extracted sequence)")
    ax.set_ylabel("MSE(probability map vs GT mask)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)


def plot_mean_metrics(out_path: Path, per_video: Dict[str, List[FrameMetrics]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt = _get_plt(show=False)

    video_names = list(per_video.keys())
    metrics = {
        "IoU": [_mean(r.iou for r in per_video[v]) for v in video_names],
        "Dice": [_mean(r.dice for r in per_video[v]) for v in video_names],
        "Precision": [_mean(r.precision for r in per_video[v]) for v in video_names],
        "Recall": [_mean(r.recall for r in per_video[v]) for v in video_names],
    }

    x = np.arange(len(video_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))

    for j, (metric_name, values) in enumerate(metrics.items()):
        ax.bar(x + (j - 1.5) * width, values, width, label=metric_name)

    ax.set_title("Mean lane segmentation metrics (thresholded mask)")
    ax.set_xticks(x)
    ax.set_xticklabels(video_names, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)


def show_plots(per_video: Dict[str, List[FrameMetrics]], out_dir: Path) -> None:
    """Display plots interactively (Windows / VS Code) and save PNGs."""
    plt = _get_plt(show=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) MSE curve
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    for video_name, rows in per_video.items():
        xs = np.arange(len(rows))
        ys = np.array([r.mse for r in rows], dtype=np.float32)
        ax1.plot(xs, ys, linewidth=1.2, label=video_name)
    ax1.set_title("Lane Segmentation MSE over frames")
    ax1.set_xlabel("Frame index (in extracted sequence)")
    ax1.set_ylabel("MSE(probability map vs GT mask)")
    ax1.grid(True, alpha=0.25)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(str(out_dir / "mse_over_frames.png"), dpi=180)

    # 2) Mean metrics bar chart
    video_names = list(per_video.keys())
    metrics = {
        "IoU": [_mean(r.iou for r in per_video[v]) for v in video_names],
        "Dice": [_mean(r.dice for r in per_video[v]) for v in video_names],
        "Precision": [_mean(r.precision for r in per_video[v]) for v in video_names],
        "Recall": [_mean(r.recall for r in per_video[v]) for v in video_names],
    }
    x = np.arange(len(video_names))
    width = 0.2

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for j, (metric_name, values) in enumerate(metrics.items()):
        ax2.bar(x + (j - 1.5) * width, values, width, label=metric_name)
    ax2.set_title("Mean lane segmentation metrics (thresholded mask)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(video_names, rotation=15, ha="right")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(ncol=4)
    fig2.tight_layout()
    fig2.savefig(str(out_dir / "mean_metrics_bar.png"), dpi=180)

    plt.show()


def main() -> int:
    base_dir = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(description="Evaluate best.pt vs laneseg_label_w16_test")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=base_dir / "checkpoints" / "best.pt",
        help="Path to best.pt checkpoint (HybridViTUNet)",
    )
    p.add_argument(
        "--frames-root",
        type=Path,
        default=base_dir / "data" / "driver_37_30frame",
        help="Root containing extracted frame folders (driver_37_30frame)",
    )
    p.add_argument(
        "--masks-root",
        type=Path,
        default=base_dir / "data" / "laneseg_label_w16_test" / "driver_37_30frame",
        help="Root containing GT mask folders (laneseg_label_w16_test)",
    )
    p.add_argument(
        "--videos",
        nargs="+",
        default=["05181432_0203.MP4", "05181520_0219.MP4","05181608_0235.MP4","05181656_0251.MP4"],
        help="Sequence folder names to evaluate",
    )
    p.add_argument("--threshold", type=float, default=0.5, help="Mask threshold")
    p.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display graphs in a window (also saves PNGs)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for quicker runs (debug)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=base_dir / "test_output" / "eval_driver_37",
        help="Output directory",
    )

    args = p.parse_args()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    inferencer = ViTLaneInferencer(args.ckpt)

    per_video: Dict[str, List[FrameMetrics]] = {}

    for video_name in args.videos:
        frames_dir = args.frames_root / video_name
        masks_dir = args.masks_root / video_name

        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        print(f"Evaluating {video_name} ...")
        rows = evaluate_sequence(
            inferencer,
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            threshold=float(args.threshold),
            max_frames=args.max_frames,
        )
        per_video[video_name] = rows

        print(
            "  mean: "
            f"MSE={_mean(r.mse for r in rows):.6f}, "
            f"IoU={_mean(r.iou for r in rows):.4f}, "
            f"Dice={_mean(r.dice for r in rows):.4f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "metrics_per_frame.csv", per_video)

    if args.show:
        # IMPORTANT: do not import pyplot with Agg before showing.
        try:
            show_plots(per_video, out_dir=args.out_dir)
        except Exception as e:
            print(
                "\n⚠ Could not display interactive plots (still saving PNGs headlessly). "
                f"Reason: {e}"
            )
            plot_mse_over_frames(args.out_dir / "mse_over_frames.png", per_video)
            plot_mean_metrics(args.out_dir / "mean_metrics_bar.png", per_video)
    else:
        plot_mse_over_frames(args.out_dir / "mse_over_frames.png", per_video)
        plot_mean_metrics(args.out_dir / "mean_metrics_bar.png", per_video)

    print(f"\n✔ Wrote: {args.out_dir / 'metrics_per_frame.csv'}")
    print(f"✔ Wrote: {args.out_dir / 'mse_over_frames.png'}")
    print(f"✔ Wrote: {args.out_dir / 'mean_metrics_bar.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
