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
        target: (B, H, W) ∈ {0,1}
        """
        # take lane channel only
        lane_logits = logits[:, 1, :, :]          # (B,H,W)
        probs = torch.sigmoid(lane_logits)        # (B,H,W)

        # valid pixels
        valid_mask = target != self.ignore_index

        target = (target == 1).float()

        probs = probs * valid_mask
        target = target * valid_mask

        dims = (0, 1, 2)

        inter = torch.sum(probs * target, dims)
        union = torch.sum(probs + target, dims)

        dice = (2.0 * inter + self.smooth) / (union + self.smooth)

        return 1.0 - dice


# -------------------------------------------------
# Combined Segmentation Loss (Weighted CE + Dice)
# -------------------------------------------------
class SegLoss(nn.Module):
    """
    Final binary segmentation loss:
      Weighted CrossEntropy (pixel accuracy)
    + Dice loss (lane shape consistency)
    """
    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=1.0,
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

        self.dice = LaneDiceLoss(
            ignore_index=ignore_index
        )

    def forward(self, logits, target):
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(device=logits.device)

        loss_ce = F.cross_entropy(
            logits,
            target,
            weight=weight,
            ignore_index=self.ignore_index,
        )
        loss_dice = self.dice(logits, target)

        return self.ce_weight * loss_ce + self.dice_weight * loss_dice
