from monai.losses import FocalLoss, DiceLoss
import torch.nn as nn

class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, alpha=0.8, gamma=2.0, apply_sigmoid=True):
        super().__init__()
        self.dice = DiceLoss(sigmoid=apply_sigmoid, reduction="mean")
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        dice = self.dice(preds, targets)
        focal = self.focal(preds, targets)
        return self.dice_weight * dice + (1 - self.dice_weight) * focal
