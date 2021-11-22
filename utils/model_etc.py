import os
import sys
sys.path.append(os.path.join("code", "PointCloudResNet", "datasets"))

import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data

from utils.config import config, update_config

from models.backbones import ResNet
from models.losses import MultiShapeCrossEntropy
from models.heads import MultiPartSegHeadResNet

def config_seting(cfg: str = "cfgs/brain/brain_pospoolxyz.yaml"):
    """
    Function returns config file of model
    
    :params cfg: str, path to config file
    
    :outputs: config: dict, dictinary with configuration of model
    """
    
    update_config(cfg)
    return config

class MultiPartSegmentationModel(nn.Module):
    def __init__(
        self,
        config,
        backbone,
        head,
        num_classes,
        num_parts,
        input_features_dim,
        radius,
        sampleDl,
        nsamples,
        npoints,
        width=144,
        depth=2,
        bottleneck_ratio=2,
    ):
        """
        Implementation of model
        
        :params config: dict, dictinary with configuration of model
        :params backbone: str, name of backbone 
        :params head: str, name of mdel's head 
        :params num_classes: ##################################################
        :params num_parts: ##################################################
        :params input_features_dim: ##################################################
        :params radius: ##################################################
        :params sampleDl: ##################################################
        :params nsamples: ##################################################
        :params npoints: ##################################################
        :params width: ##################################################
        :params depth: ##################################################
        :params bottleneck_ratio: ##################################################
        """
        
        super(MultiPartSegmentationModel, self).__init__()

        if backbone == "resnet":
            self.backbone = ResNet(
                config,
                input_features_dim,
                radius,
                sampleDl,
                nsamples,
                npoints,
                width=width,
                depth=depth,
                bottleneck_ratio=bottleneck_ratio,
            )
        else:
            raise NotImplementedError(
                f"Backbone {backbone} not implemented in Multi-Part Segmentation Model"
            )

        if head == "resnet_part_seg":
            self.segmentation_head = MultiPartSegHeadResNet(
                num_classes, width, radius, nsamples, num_parts
            )
        else:
            raise NotImplementedError(
                f"Head {backbone} not implemented in Multi-Part Segmentation Model"
            )

    def forward(self, xyz, mask, features):
        """
        Forward pass of model
        
        :params xyz: torch.tensor, tensor of coordinates with size [batch_size, number_of_points, number_of_coordinates=3]
        :params mask: torch.tensor, tensor of shape (number_of_points,) of 0 and 1, contains 1 for each unique point and 0 for every 2,3,4... repetition of the point
        :params features: torch.tensor, tensor of features with size [batch_size, number_of_features, number_of_points]
        
        :outputs prediction: torch.tensor, tensor of predictions with size [batch_size, 2, number_of_points]
        """
        end_points = self.backbone(xyz, mask, features)
        return self.segmentation_head(end_points)

    def init_weights(self):
        """
        Initialization of weights for each convolution layer 
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

def build_multi_part_segmentation(
    config, weights: list = None, type: str = "MultiShape"
):
    """
    Function creates model and criterion
    
    :params config: dict, dictionary with all the parameters of model
    :params weights: np.ndarray, a manual rescaling weight given to each class, used in BCE loss
    :params type: str, type of criterion to use
    
    :outputs: model
    :outputs: criterion
    """
    
    model = MultiPartSegmentationModel(
        config,
        config.backbone,
        config.head,
        config.num_classes,
        config.num_parts,
        config.input_features_dim,
        config.radius,
        config.sampleDl,
        config.nsamples,
        config.npoints,
        config.width,
        config.depth,
        config.bottleneck_ratio,
    )

    criterion = MultiShapeCrossEntropy(
        config.num_classes, type=type, weights=weights,
    )
    
    return model, criterion

def get_optimizer(model, config):

    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(),
                                 lr=config.base_learning_rate,
                                 weight_decay=config.weight_decay)
    elif config.optimizer == "adamW":
        return torch.optim.AdamW(model.parameters(),
                                  lr=config.base_learning_rate,
                                  weight_decay=config.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not supported")