import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for handling highly imbalanced datasets.
    It adds a scaling factor to BCEWithLogitsLoss to focus on hard, misclassified examples.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, pos_weight: float = 1.0, label_smoothing: float = 0.0, reduction: str = "mean"):
        """
        Args:
            alpha: Weighting factor for the positive class
            gamma: Focusing parameter to reduce the relative loss for well-classified examples
            pos_weight: Weight applied to positive examples (equivalent to pos_weight in BCEWithLogitsLoss)
            label_smoothing: Float in [0.0, 1.0]. Smooths labels by squeezing them towards 0.5.
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        # Use a register buffer for pos_weight so it's moved to the correct device automatically
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, 1) or (N,) raw unnormalized scores
            targets: (N, 1) or (N,) binary labels (0 or 1)
        """
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        if targets.dim() > 1:
            targets = targets.squeeze(-1)
            
        orig_targets = targets.float()
        targets = orig_targets.clone()
        
        # Apply label smoothing
        if self.label_smoothing > 0.0:
            targets = orig_targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            
        # Calculate BCE loss per element
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        
        # Calculate pt (probability of the true class)
        probs = torch.sigmoid(logits)
        pt = torch.where(orig_targets == 1.0, probs, 1 - probs)
        
        # Calculate Focal Loss
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
