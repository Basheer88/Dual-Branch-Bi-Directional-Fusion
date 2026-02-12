import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMarginContextLoss(nn.Module):
    """
    Enhanced Adaptive Margin Context Loss (EAMCL) using label-dependent margins.
    When instantiated, it computes adaptive margins from the provided training labels.
    
    Parameters:
        train_labels (array-like): A list or NumPy array of training labels.
        max_m (float): Maximum margin value (default: 0.5).
        lambda_coef (float): Weighting coefficient for the margin loss term (default: 0.1).
        reduction (str): Reduction method ('mean' or 'sum', default: 'mean').
    """
    def __init__(self, train_labels, max_m=0.5, lambda_coef=0.1, reduction='mean'):
        super().__init__()
        # Compute class counts from train_labels.
        class_counts = np.bincount(train_labels)
        # Compute adaptive margin for each class; smaller classes get larger margins.
        m_list = torch.tensor([max_m / (n ** 0.25) for n in class_counts], dtype=torch.float)
        self.m_list = m_list
        self.lambda_coef = lambda_coef
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        true_logits = inputs.gather(1, targets.unsqueeze(1)).squeeze(1)
        mask = torch.ones_like(inputs, dtype=torch.bool)
        mask.scatter_(1, targets.unsqueeze(1), False)
        max_other, _ = inputs.masked_fill(~mask, -float('inf')).max(dim=1)
        # Ensure the adaptive margin tensor is on the same device as inputs.
        m_target = self.m_list.to(inputs.device)[targets]
        margin_loss = F.relu(m_target - (true_logits - max_other))
        loss = ce_loss + self.lambda_coef * margin_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()
