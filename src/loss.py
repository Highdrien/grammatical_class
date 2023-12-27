import torch
from torch import nn as nn


class CrossEntropyLossOneHot(nn.Module):
    def __init__(self,reduction='mean'):
        super(CrossEntropyLossOneHot, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, input, target):
        # Convert one-hot vectors to indices
        target_indices = torch.argmax(target, dim=2)
        #permutation des axes pour que la shape de y_pred et y_true soit la même
        input=input.permute(0,2,1)
        return self.cross_entropy_loss(input, target_indices)