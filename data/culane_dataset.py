import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np

class CULaneDataset(Dataset):
    """
    Expects list lines:
      img_path mask_path exist1 exist2 exist3 exist4
    Uses only img_path and mask_path.
    """
    def __init__(self, root: str, list_file: str, img_size=(288, 800), augment=False):
        self.root = root
        self.augment = augment
        self.img_h, self.img_w = img_size

        with open(list_file, "r") as f:
            self.lines = [ln.strip() for ln in f if ln.strip()]

    def __len__(self):
        return len(self.lines)

    def _aug(self, img, mask):
        # simple + safe augmentations (no geometry heavy stuff unless you handle masks carefully)
        if np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        # brightness/contrast
        if np.random.rand() < 0.5:
            alpha = 0.8 + 0.4 * np.random.rand()  # contrast
            beta = (np.random.rand() - 0.5) * 20  # brightness
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
        return img, mask

    def __getitem__(self, idx):
        items = self.lines[idx].split()
        img_rel = items[0]
        mask_rel = items[1]

        img = cv2.imread(os.path.join(self.root, img_rel))
        if img is None:
            raise FileNotFoundError(img_rel)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.root, mask_rel), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_rel)

        # Resize to fixed (H,W) (CULane is wide; prefer non-square)
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = self._aug(img, mask)

        img = TF.to_tensor(img)                      # float32 in [0,1], (3,H,W)
        mask = torch.from_numpy(mask).long()         # (H,W) int64
        return img, mask
