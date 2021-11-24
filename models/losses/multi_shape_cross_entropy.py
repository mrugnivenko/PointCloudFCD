import torch
import torch.nn as nn
import torch.nn.functional as F
        
class DiceBCELoss(nn.Module):
    def __init__(self, weights=None):
        super(DiceBCELoss, self).__init__()
        if weights is not None:
            self.bce = torch.nn.CrossEntropyLoss(torch.tensor(weights).float().cuda())
        else:
            self.bce = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, targets, smooth=1):
        inputs_for_bce =  inputs.permute(0,2,1)
        inputs_for_bce = inputs_for_bce.reshape(-1, inputs_for_bce.shape[-1])
        targets_for_bce = targets.view(-1)
        inputs = F.softmax(inputs, dim = 1)[:,1,:].squeeze() #(B,C,N) - > (B,N)     

        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = self.bce(inputs_for_bce, targets_for_bce)
        #BCE = F.binary_cross_entropy(inputs, targets.float(), reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class MultiShapeCrossEntropy(nn.Module):
    def __init__(self, num_classes: int, type: str = 'BCE', weights = None):
        super(MultiShapeCrossEntropy, self).__init__()
        """
        Class implements different loss functions
        
        :params num_classes: int, number of classes 
        :params type: str, type of loss function
        :params weights: list, a manual rescaling weights given to each class, used in BCE loss
        """
        
        self.type = type
        
        if weights is not None:
            self.bce = torch.nn.CrossEntropyLoss(torch.tensor(weights).float().cuda())
        else:
            self.bce = torch.nn.CrossEntropyLoss()
            
        self.dice = DiceBCELoss(weights=weights)

    def forward(self, logits_all_shapes, points_labels):
        """
        Function for loss calculation 
        
        :params logits_all_shapes: tuple, logits_all_shapes[0] is torch.tensor with unnormalized prediction of the model with size [batch_size, 2, number_of_points]
        :params points_labels: torch.tensor, tensor of shape [batch_size, number_of_points] with 0 and 1, contains 1 for FCD point and 0 for non-FCD
        
        :outputs loss: torch.tensor, loss
        """
                
        if self.type == 'BCE':
            return self.bce(logits_all_shapes[0], points_labels)
        
        elif self.type == 'DICE':
            return self.dice(logits_all_shapes, points_labels)