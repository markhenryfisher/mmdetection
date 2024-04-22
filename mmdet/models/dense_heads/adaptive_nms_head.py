# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:36:06 2024

@author: mhf
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig
from .rpn_head import RPNHead

@MODELS.register_module()
class AdaptiveNMSHead(RPNHead):
    """Implementation of Adaptive NMS head.
    """
    
    def __init__(self,
                 in_channels: int,
                 num_classes: int = 1,
                 init_cfg: MultiConfig = dict(
                     type='Normal', layer='Conv2d', std=0.01),
                 num_convs: int = 1,
                 **kwargs) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)
        
    def _init_layers(self) -> None:
        super()._init_layers()

        # init the density prediction regressor
        in_channels = self.in_channels
        num_anchors = self.num_base_priors
        reg_dim = self.bbox_coder.encode_size
        cls_out_channels = self.cls_out_channels
        n_dns_channels = in_channels + (num_anchors * reg_dim) \
            + (num_anchors * cls_out_channels) 
        self.dpn_reg = nn.Conv2d(n_dns_channels, num_anchors, kernel_size=5, padding=2, stride=1)
        
    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
                dns_pred (Tensor): Density
        """
        x = self.rpn_conv(x)
        x = F.relu(x)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
       
        # density = []
        # for feature, obj, bbox_deltas in zip(x, objectness, pred_bbox_deltas):
        #     t = self.conv(feature)
        #     dns_input = torch.cat((t, obj, bbox_deltas), dim=1)            
        #     density.append(self.density_pred(dns_input))

        dns_input = torch.cat((x, rpn_cls_score, rpn_bbox_pred), dim=1)        
        rpn_dns_pred = self.dpn_reg(dns_input)
        
        return rpn_cls_score, rpn_bbox_pred, rpn_dns_pred