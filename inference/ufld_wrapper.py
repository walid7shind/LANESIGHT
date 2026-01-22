import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
import numpy as np
import scipy.special

# ============================================================
# Resolve UFLD repo path dynamically (NO hardcoding)
# ============================================================
THIS_DIR = Path(__file__).resolve().parent
LANESIGHT_ROOT = THIS_DIR.parent
VISION_ROOT = LANESIGHT_ROOT.parent
UFLD_ROOT = VISION_ROOT / "UFLD" / "Ultra-Fast-Lane-Detection"

if not UFLD_ROOT.exists():
    raise FileNotFoundError(
        f"UFLD repo not found at {UFLD_ROOT}. "
        f"Clone it as a sibling of LANESIGHT."
    )

sys.path.insert(0, str(UFLD_ROOT))

# ============================================================
# UFLD internal imports
# ============================================================
from model.model import parsingNet
from utils.config import Config
from data.constant import culane_row_anchor, tusimple_row_anchor


# ============================================================
# UFLD Inferencer (PURE inference, no side effects)
# ============================================================
class UFLDInferencer:
    """
    Ultra-Fast Lane Detection wrapper.

    Responsibilities:
      - load pretrained UFLD model
      - run inference on a single frame
      - return lane geometry in image coordinates

    Does NOT:
      - draw
      - write video
      - depend on LANESIGHT internals
    """

    def __init__(
        self,
        weights_path: Path,
        config_path: Path,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ----------------------------
        # Load config
        # ----------------------------
        self.cfg = Config.fromfile(str(config_path))

        if self.cfg.dataset == "CULane":
            self.row_anchor = culane_row_anchor
            self.cls_num_per_lane = 18
        elif self.cfg.dataset == "Tusimple":
            self.row_anchor = tusimple_row_anchor
            self.cls_num_per_lane = 56
        else:
            raise NotImplementedError(f"Unsupported dataset: {self.cfg.dataset}")

        # ----------------------------
        # Build model
        # ----------------------------
        self.model = parsingNet(
            pretrained=False,
            backbone=str(self.cfg.backbone),
            cls_dim=(
                self.cfg.griding_num + 1,
                self.cls_num_per_lane,
                self.cfg.num_lanes,
            ),
            use_aux=False,
        )

        # ----------------------------
        # Load weights
        # ----------------------------
        state = torch.load(str(weights_path), map_location="cpu")
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state

        clean_state = {}
        for k, v in state_dict.items():
            clean_state[k.replace("module.", "")] = v

        self.model.load_state_dict(clean_state, strict=False)
        self.model.to(self.device).eval()

        # UFLD fixed input resolution
        self.inp_w = 800
        self.inp_h = 288

    # ========================================================
    # Public API
    # ========================================================
    def infer_frame(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        """
        Args:
            frame_bgr: np.ndarray (H,W,3) BGR image

        Returns:
            lanes: list of np.ndarray, each shape (N,2) in image coords
        """
        h, w = frame_bgr.shape[:2]

        x = self._preprocess(frame_bgr)

        with torch.no_grad():
            output = self.model(x)

        lanes = self._decode_lanes(output, w, h)
        return lanes

    # ========================================================
    # Internal helpers
    # ========================================================
    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(frame_bgr, (self.inp_w, self.inp_h))
        img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR → RGB

        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )

        img = np.transpose(img, (2, 0, 1))
        x = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return x

    def _decode_lanes(
        self,
        output: torch.Tensor,
        img_w: int,
        img_h: int,
    ) -> List[np.ndarray]:
        """Decode UFLD output into lane point sequences."""
        out = output[0].detach().cpu().numpy()
        out = out[:, ::-1, :]

        prob = scipy.special.softmax(out[:-1, :, :], axis=0)
        idx = (np.arange(self.cfg.griding_num) + 1).reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)

        out_i = np.argmax(out, axis=0)
        loc[out_i == self.cfg.griding_num] = 0

        col_sample = np.linspace(0, 800 - 1, self.cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        lanes: List[np.ndarray] = []

        for lane_id in range(loc.shape[1]):
            pts: List[Tuple[int, int]] = []

            if np.count_nonzero(loc[:, lane_id]) <= 2:
                continue

            for k in range(loc.shape[0]):
                if loc[k, lane_id] <= 0:
                    continue

                x = int(loc[k, lane_id] * col_sample_w * img_w / 800) - 1
                y = int(
                    img_h * (self.row_anchor[self.cls_num_per_lane - 1 - k] / 288)
                ) - 1

                pts.append((x, y))

            if len(pts) >= 2:
                lanes.append(np.array(pts, dtype=np.int32))

        return lanes
