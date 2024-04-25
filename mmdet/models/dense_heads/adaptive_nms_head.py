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
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptInstanceList
from mmdet.structures.bbox import cat_boxes, get_box_tensor
from .rpn_head import RPNHead

from ..utils import images_to_levels, multi_apply

from mmdet.models.utils.adaptnms_utils import IoUGenerator

import pdb

@MODELS.register_module()
class AdaptiveNMSHead(RPNHead):
    """Implementation of Adaptive NMS head.
    """
    
    def __init__(self,
                 in_channels: int,
                 num_classes: int = 1,
                 loss_dns: ConfigType = dict(
                     type='SmoothL1Loss', loss_weight=1.0),

                 init_cfg: MultiConfig = dict(
                     type='Normal', layer='Conv2d', std=0.01),
                 num_convs: int = 1,
                 **kwargs) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)
        
        self.loss_dns = MODELS.build(loss_dns)

        
        
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
        # pdb.set_trace()
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

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            dns_pred: Tensor, dns_targets,
                            anchors: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor, avg_factor: int) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).

            dns_pred (list[Tensor]): Density for each scale level,
                has shape (N, num_anchors * 1, H, W).                
                               
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)
        
        # density loss
        # TODO! Test this!
        pdb.set_trace()
        dns_pred = dns_pred.permute(0, 2, 3,
                                    1).reshape(-1, self.dns_out_channels)
        loss_dns = self.loss_dns(
            dns_pred, dns_targets, avg_factor=avg_factor)
        
        # regression loss
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1,
                                                 self.bbox_coder.encode_size)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        return loss_cls, loss_bbox
    
    # mhf 23.04.24 new loss_by_feat for Adaptive NMS
    # based on AnchorHead.loss_by_feat
    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     dns_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) \
            -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
                
            dns_preds (list[Tensor]): Density for each scale level,
                has shape (N, num_anchors * num_classes, H, W).
                
            batch_gt_instances (list[obj:InstanceData]): Batch of gt_instance.
                It usually includes ``bboxes`` and ``labels`` attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[obj:InstanceData], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor) = cls_reg_targets

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            dns_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=avg_factor)
        
        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        
        # gt_bboxes = batch_gt_instances
        # dns_gt = IoUGenerator().generate_box_density(gt_bboxes)
        
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])