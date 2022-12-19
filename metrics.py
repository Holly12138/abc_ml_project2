import torch
import torch.nn as nn
# Jaccard Index (IoULoss)
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, pred, target):
        
        m1 = pred.view(-1)
        m2 = target.view(-1)
        
        score = Loss_prep(m1, m2)

        return 1 - score


def Loss_prep(m1, m2):
    smooth = 20
    inter = (m1 * m2).sum()
    union = (m1 + m2).sum() - inter
    Score = (inter + smooth) / (union + smooth)
    return Score


# F1-Loss
class F1_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(F1_Loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        F1_approx = 2*(intersection + smooth)/(union + intersection + smooth)
                
        return 1 - F1_approx
