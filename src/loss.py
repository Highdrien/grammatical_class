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
        return self.cross_entropy_loss(input, target_indices.float())  # Convert target_indices to float
    
class CrossEntropyLossOneHotMorph(nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLossOneHotMorph, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, input, target):
        

        # Initialize a list to store the losses for each subclass
        losses = []

        # Iterate over the subclasses dimension
        for i in range(input.shape[3]):
            # Select the input and target for the current subclass
            input_i = input[:, :, i, :]
            target_i = target[:, :, i, :]
            # Convert one-hot vectors to indices
            target_i = torch.argmax(target, dim=2)
            #permutation des axes pour que la shape de y_pred et y_true soit la même
            input_i=input_i.permute(0,2,1)
            target_i=target_i.permute(0,2,1)

            # Compute the loss for the current subclass
            loss_i = self.cross_entropy_loss(input_i, target_i.float())  # Convert target_i to float

            # Add the loss to the list of losses
            losses.append(loss_i)

        # Compute the mean of the losses
        loss = torch.stack(losses).mean()

        return loss