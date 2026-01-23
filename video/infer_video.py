
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

	# ViT -> blue
	color[masks.vit_mask == 1] = (255, 0, 0)
	# UFLD -> red
	color[masks.ufld_mask == 1] = (0, 0, 255)
	# Traditional -> green
	color[masks.traditional_mask == 1] = (0, 255, 0)
	# YOLO -> red
	color[masks.yolo_mask == 1] = (0, 0, 255)

	overlay = cv2.addWeighted(overlay, 1.0, color, alpha, 0.0)
	return overlay

