import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

from models.hybrid_vit_unet import HybridViTUNet


class ViTLaneInferencer:
    """
    ViT-UNet hybrid lane segmentation wrapper.

    Outputs semantic lane information:
      - probability map
      - binary mask
    """

    def __init__(
        self,
        ckpt_path: Path,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(str(ckpt_path), map_location=self.device)

        self.img_size = ckpt["img_size"]          # (H, W)
        self.num_classes = ckpt["num_classes"]    # usually 2 (bg / lane)

        self.model = HybridViTUNet(
            num_classes=self.num_classes,
            img_size_hw=self.img_size,
            base=48,
        ).to(self.device)

        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    # ======================================================
    # Public API
    # ======================================================
    def infer_frame(
        self,
        frame_bgr: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            frame_bgr: (H,W,3) BGR image
            threshold: probability threshold for binary mask

        Returns:
            prob_map: (H,W) float32 in [0,1]
            bin_mask: (H,W) uint8 {0,1}
        """
        h, w = frame_bgr.shape[:2]

        x = self._preprocess(frame_bgr)

        with torch.no_grad():
            logits = self.model(x)  # (1,C,H,W)

        prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()  # lane class

        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        bin_mask = (prob >= threshold).astype(np.uint8)

        return prob, bin_mask

    # ======================================================
    # Internal helpers
    # ======================================================
    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(
            frame_bgr,
            (self.img_size[1], self.img_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        x = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return x
