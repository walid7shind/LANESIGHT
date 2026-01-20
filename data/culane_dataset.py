import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np

# OpenCV can spawn its own threadpool per worker process; with PyTorch DataLoader
# workers this can cause CPU oversubscription and slowdowns.
try:
    cv2.setNumThreads(0)
except Exception:
    pass

class CULaneDataset(Dataset):
    """
    Expects list lines:
      img_path mask_path exist1 exist2 exist3 exist4
    Uses only img_path and mask_path.
    """
    def __init__(self, root: str, list_file: str, img_size=(288, 800), augment=False):
        self.root = os.path.abspath(root)
        self.list_file = list_file
        self.augment = augment
        self.img_h, self.img_w = img_size

        with open(list_file, "r", encoding="utf-8") as f:
            self.lines = [ln.strip() for ln in f if ln.strip()]

    @staticmethod
    def _normalize_rel_path(p: str) -> str:
        # List files often contain paths like '/driver_xxx/...'; on Windows,
        # os.path.join(root, '/...') discards root, causing FileNotFoundError.
        p = p.strip().replace("\\", "/")
        p = p.lstrip("/")
        if p.startswith("./"):
            p = p[2:]
        return os.path.normpath(p)

    def __len__(self):
        return len(self.lines)

    def _aug(self, img, mask):
        # IMPORTANT:
        # - Photometric augmentation is applied ONLY to the RGB image (on-GPU in train.py).
        # - Geometric augmentation (warp) must be applied identically to BOTH image and mask.
        #
        # This dataset stays geometry-neutral by default to avoid label corruption.
        return img, mask

    def __getitem__(self, idx):
        items = self.lines[idx].split()
        img_rel = self._normalize_rel_path(items[0])
        mask_rel = self._normalize_rel_path(items[1])

        img_path = os.path.join(self.root, img_rel)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(
                f"Missing image file: rel='{items[0]}' -> '{img_rel}', abs='{img_path}' (root='{self.root}', list='{self.list_file}')"
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.root, mask_rel)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(
                f"Missing mask file: rel='{items[1]}' -> '{mask_rel}', abs='{mask_path}' (root='{self.root}', list='{self.list_file}')"
            )

        # Resize to fixed (H,W) (CULane is wide; prefer non-square)
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = self._aug(img, mask)

        img = TF.to_tensor(img)                      # float32 in [0,1], (3,H,W)
        mask = torch.from_numpy(mask > 0).long()        # 2 classes only (0=bg,1=lane), (H,W)
        return img, mask
