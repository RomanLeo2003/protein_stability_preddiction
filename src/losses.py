import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_factor=2.1):
        super(WeightedMSELoss, self).__init__()
        self.weight_factor = weight_factor

    def forward(self, output, target):
        loss = (output - target) ** 2

        weights = torch.ones_like(target)
        weights[target > 0] *= self.weight_factor

        weights[target > 3] *= self.weight_factor
        weights[target < -3] *= self.weight_factor

        weighted_loss = loss * weights
        return weighted_loss.mean()
    
