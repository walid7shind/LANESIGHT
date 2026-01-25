from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from inference.ufld_wrapper import UFLDInferencer
from inference.vit_hybrid_wrapper import ViTLaneInferencer
from inference.yolo_wrapper import YOLOInferencer
from inference.fusion import Lane, LaneSet, lanes_to_mask
from traditional_proccessing.taditionalBS import traditional_polygon_mask


def _resolve_default_paths() -> Dict[str, Path]:
	"""Resolve default model/config paths relative to the LANESIGHT repo."""
	lanesight_root = Path(__file__).resolve().parents[1]
	vision_root = lanesight_root.parent
	ufld_root = vision_root / "UFLD" / "Ultra-Fast-Lane-Detection"

	return {
		"lanesight_root": lanesight_root,
		"vision_root": vision_root,
		"ufld_root": ufld_root,
		"ufld_weights": ufld_root / "weights" / "culane_18.pth",
		"ufld_config": ufld_root / "configs" / "culane.py",
		"vit_ckpt": lanesight_root / "checkpoints" / "best.pt",
		"yolo_weights": lanesight_root / "yolov8n.pt",
	}


def yolo_detections_to_mask(
	detections: List[dict],
	shape_hw: Tuple[int, int],
) -> np.ndarray:
	"""Rasterize YOLO xyxy boxes into a uint8 {0,1} mask."""
	h, w = shape_hw
	mask = np.zeros((h, w), dtype=np.uint8)

	if not detections:
		return mask

	for det in detections:
		x1, y1, x2, y2 = det.get("xyxy", (0, 0, 0, 0))
		x1 = int(np.clip(x1, 0, w - 1))
		x2 = int(np.clip(x2, 0, w - 1))
		y1 = int(np.clip(y1, 0, h - 1))
		y2 = int(np.clip(y2, 0, h - 1))
		if x2 <= x1 or y2 <= y1:
			continue
		mask[y1:y2, x1:x2] = 1

	return mask


def ufld_lanes_to_mask(
	ufld_lanes: List[np.ndarray],
	shape_hw: Tuple[int, int],
	*,
	thickness: int = 6,
) -> np.ndarray:
	"""Rasterize UFLD lane polylines into a uint8 {0,1} mask."""
	lane_set = LaneSet(
		[
			Lane(points=poly.astype(np.int32), confidence=1.0, source="ufld")
			for poly in ufld_lanes
			if poly is not None and len(poly) >= 2
		]
	)
	return lanes_to_mask(lane_set, shape_hw, thickness=thickness)


@dataclass
class FourMasks:
	"""Container for the four masks you requested."""

	vit_mask: np.ndarray
	ufld_mask: np.ndarray
	traditional_mask: np.ndarray
	yolo_mask: np.ndarray


class VisionMaskGenerator:
	"""Generates 4 masks per frame (ViT, UFLD, Traditional polygon, YOLO)."""

	def __init__(
		self,
		*,
		ufld_weights: Optional[Path] = None,
		ufld_config: Optional[Path] = None,
		vit_ckpt: Optional[Path] = None,
		yolo_weights: str | Path | None = None,
		yolo_classes: Optional[List[int]] = None,
		yolo_conf_thres: float = 0.3,
		device: str | None = None,
	):
		paths = _resolve_default_paths()

		ufld_weights = Path(ufld_weights) if ufld_weights is not None else paths["ufld_weights"]
		ufld_config = Path(ufld_config) if ufld_config is not None else paths["ufld_config"]
		vit_ckpt = Path(vit_ckpt) if vit_ckpt is not None else paths["vit_ckpt"]

		if yolo_weights is None:
			yolo_weights = str(paths["yolo_weights"])
		else:
			yolo_weights = str(yolo_weights)

		self.ufld = UFLDInferencer(weights_path=ufld_weights, config_path=ufld_config, device=device)
		self.vit = ViTLaneInferencer(ckpt_path=vit_ckpt, device=device)
		self.yolo = YOLOInferencer(
			weights=yolo_weights,
			device=device,
			classes=yolo_classes,
			conf_thres=yolo_conf_thres,
		)

	def infer_masks(self, frame_bgr: np.ndarray) -> FourMasks:
		"""Return the four binary masks as uint8 {0,1}, shape (H,W)."""
		h, w = frame_bgr.shape[:2]

		# 1) ViT mask (best.pt)
		_vit_prob, vit_mask = self.vit.infer_frame(frame_bgr)

		# 2) UFLD mask
		ufld_lanes = self.ufld.infer_frame(frame_bgr)
		ufld_mask = ufld_lanes_to_mask(ufld_lanes, (h, w))

		# 3) Traditional polygon mask
		trad_mask = traditional_polygon_mask(frame_bgr)

		# 4) YOLO mask
		yolo_dets = self.yolo.infer_frame(frame_bgr)
		yolo_mask = yolo_detections_to_mask(yolo_dets, (h, w))

		# Normalize dtypes
		vit_mask = vit_mask.astype(np.uint8)
		ufld_mask = ufld_mask.astype(np.uint8)
		trad_mask = trad_mask.astype(np.uint8)
		yolo_mask = yolo_mask.astype(np.uint8)

		return FourMasks(
			vit_mask=vit_mask,
			ufld_mask=ufld_mask,
			traditional_mask=trad_mask,
			yolo_mask=yolo_mask,
		)


def overlay_masks(
	frame_bgr: np.ndarray,
	masks: FourMasks,
	*,
	alpha: float = 0.25,
) -> np.ndarray:
	"""Convenience helper to visualize the 4 masks on top of the frame."""
	overlay = frame_bgr.copy()
	color = np.zeros_like(frame_bgr)

	# Treat any nonzero mask as active (some producers output 0/255).
	vit_on = masks.vit_mask != 0
	ufld_on = masks.ufld_mask != 0
	poly_on = masks.traditional_mask != 0
	yolo_on = masks.yolo_mask != 0

	# ViT -> blue
	color[vit_on] = (255, 0, 0)
	# UFLD -> green
	color[ufld_on] = (0, 255, 0)
	# Traditional polygon -> green
	color[poly_on] = (0, 255, 0)
	# YOLO -> red
	color[yolo_on] = (0, 0, 255)

	overlay = cv2.addWeighted(overlay, 1.0, color, alpha, 0.0)
	return overlay


def overlay_single_mask(
	frame_bgr: np.ndarray,
	mask01: np.ndarray,
	*,
	alpha: float = 0.25,
	color_bgr: Tuple[int, int, int] = (255, 0, 0),  # blue in BGR
	title: str | None = None,
) -> np.ndarray:
	"""Overlay a single uint8 {0,1} mask onto a frame (all masks use same color)."""
	overlay = frame_bgr.copy()
	color = np.zeros_like(frame_bgr)
	color[mask01 != 0] = color_bgr
	out = cv2.addWeighted(overlay, 1.0, color, alpha, 0.0)

	if title:
		cv2.putText(
			out,
			title,
			(12, 32),
			cv2.FONT_HERSHEY_SIMPLEX,
			1.0,
			(255, 255, 255),
			2,
			cv2.LINE_AA,
		)
		cv2.putText(
			out,
			title,
			(12, 32),
			cv2.FONT_HERSHEY_SIMPLEX,
			1.0,
			(0, 0, 0),
			5,
			cv2.LINE_AA,
		)
	return out


def make_split_2x2(
	frame_bgr: np.ndarray,
	masks: FourMasks,
	*,
	alpha: float = 0.25,
	vit_color_bgr: Tuple[int, int, int] = (255, 0, 0),
	yolo_color_bgr: Tuple[int, int, int] = (0, 0, 255),
	ufld_color_bgr: Tuple[int, int, int] = (0, 255, 0),
	poly_color_bgr: Tuple[int, int, int] = (0, 255, 0),
	with_titles: bool = True,
) -> np.ndarray:
	"""
	Return a 2x2 split-screen BGR frame:
	(1) ViT (top-left), (2) YOLO (top-right), (3) UFLD (bottom-left), (4) Polygon (bottom-right)
	Colors: ViT blue, YOLO red, UFLD green, Polygon green.
	"""
	h, w = frame_bgr.shape[:2]
	h2, w2 = max(1, h // 2), max(1, w // 2)
	out_h, out_w = h2 * 2, w2 * 2

	vit = overlay_single_mask(frame_bgr, masks.vit_mask, alpha=alpha, color_bgr=vit_color_bgr, title=("ViT-Hybrid" if with_titles else None))
	yolo = overlay_single_mask(frame_bgr, masks.yolo_mask, alpha=alpha, color_bgr=yolo_color_bgr, title=("YOLO" if with_titles else None))
	ufld = overlay_single_mask(frame_bgr, masks.ufld_mask, alpha=alpha, color_bgr=ufld_color_bgr, title=("UFLD" if with_titles else None))
	poly = overlay_single_mask(frame_bgr, masks.traditional_mask, alpha=alpha, color_bgr=poly_color_bgr, title=("Polygon" if with_titles else None))

	vit = cv2.resize(vit, (w2, h2), interpolation=cv2.INTER_AREA)
	yolo = cv2.resize(yolo, (w2, h2), interpolation=cv2.INTER_AREA)
	ufld = cv2.resize(ufld, (w2, h2), interpolation=cv2.INTER_AREA)
	poly = cv2.resize(poly, (w2, h2), interpolation=cv2.INTER_AREA)

	top = np.hstack([vit, yolo])
	bot = np.hstack([ufld, poly])
	grid = np.vstack([top, bot])

	# Ensure exact even dimensions (for odd input sizes).
	return grid[:out_h, :out_w].copy()

