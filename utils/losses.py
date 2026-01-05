import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target, num_classes):
        # logits: (B,K,H,W), target: (B,H,W)
        probs = F.softmax(logits, dim=1)
        target_1h = F.one_hot(target, num_classes=num_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        inter = torch.sum(probs * target_1h, dims)
        union = torch.sum(probs + target_1h, dims)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class SegLoss(nn.Module):
    def __init__(self, num_classes, ce_w=1.0, dice_w=0.5, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = SoftDiceLoss()
        self.ce_w = ce_w
        self.dice_w = dice_w

    def forward(self, logits, target):
        return self.ce_w * self.ce(logits, target) + self.dice_w * self.dice(logits, target, self.num_classes)
