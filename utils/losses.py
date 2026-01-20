import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Binary Lane Dice Loss (foreground only)
# -------------------------------------------------
class LaneDiceLoss(nn.Module):
    """
    Dice loss computed ONLY on lane pixels (class = 1).
    Uses sigmoid (NOT softmax).
    """
    def __init__(self, ignore_index=255, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: (B, 2, H, W)
        target: (B, H, W) ∈ {0,1} or ignore_index
        """
        # lane channel
        lane_logits = logits[:, 1, :, :]      # (B,H,W)
        probs = torch.sigmoid(lane_logits)    # (B,H,W)

        valid_mask = target != self.ignore_index
        target_lane = (target == 1).float()

        probs = probs * valid_mask
        target_lane = target_lane * valid_mask

        dims = (0, 1, 2)
        inter = torch.sum(probs * target_lane, dims)
        union = torch.sum(probs + target_lane, dims)

        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice


# -------------------------------------------------
# Combined Segmentation Loss (CE + Dice + Vertical Prior)
# -------------------------------------------------
class SegLoss(nn.Module):
    """
    Binary lane segmentation loss with:
      - Class-weighted Cross Entropy
      - Soft vertical prior (loss-level)
      - Dice loss (shape consistency)
    """
    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=0.3,
        ignore_index=255,
        class_weights=(0.3, 3.0),   # background, lane
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32),
            persistent=False,
        )

        self.dice = LaneDiceLoss(ignore_index=ignore_index)

        # Cache for vertical prior map to avoid recreating it every forward.
        self._vert_cache = None
        self._vert_cache_hw = None
        self._vert_cache_device = None

    # -------------------------
    # Vertical prior (static)
    # -------------------------
    @staticmethod
    def vertical_weight_map(
        H,
        W,
        device,
        y0=0.6,
        softness=0.1,
        strength=2.0,
    ):
        """
        Returns (1, H, W) tensor.
        Higher penalty near top of image.
        """
        y = torch.linspace(0, 1, steps=H, device=device)  # (H,)
        y = y.view(H, 1).expand(H, W)                     # (H,W)

        w = 1.0 + strength * torch.sigmoid((y0 - y) / softness)
        return w.unsqueeze(0)  # (1,H,W)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, logits, target):
        """
        logits: (B,2,H,W)
        target: (B,H,W)
        """
        weight = self.class_weights.to(logits.device)

        # pixel-wise CE
        ce = F.cross_entropy(
            logits,
            target,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction="none"
        )  # (B,H,W)

        B, H, W = target.shape

        # vertical prior (cached)
        if (
            self._vert_cache is None
            or self._vert_cache_hw != (H, W)
            or self._vert_cache_device != logits.device
        ):
            self._vert_cache = self.vertical_weight_map(
                H,
                W,
                device=logits.device,
                y0=0.6,
                softness=0.1,
                strength=2.0,
            )
            self._vert_cache_hw = (H, W)
            self._vert_cache_device = logits.device

        w_vert = self._vert_cache

        # ignore mask
        valid = target != self.ignore_index

        # OPTIONAL: penalize mainly false positives
        with torch.no_grad():
            lane_pred = logits.argmax(dim=1) == 1  # (B,H,W)

        loss_ce = (ce * w_vert * lane_pred.float())[valid].mean()

        # Dice (unchanged)
        loss_dice = self.dice(logits, target)

        return self.ce_weight * loss_ce + self.dice_weight * loss_dice
