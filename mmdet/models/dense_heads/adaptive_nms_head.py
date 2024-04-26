# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:36:06 2024

@author: mhf
"""
from typing import List, Optional, Tuple, Union

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptInstanceList
from mmdet.structures.bbox import BaseBoxes, cat_boxes, get_box_tensor
from .rpn_head import RPNHead

from ..task_modules.prior_generators import anchor_inside_flags
from ..utils import images_to_levels, multi_apply, unmap

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
    
    
    # def get_targets(self,
    #                 anchor_list: List[List[Tensor]],
    #                 valid_flag_list: List[List[Tensor]],
    #                 batch_gt_instances: InstanceList,
    #                 batch_img_metas: List[dict],
    #                 batch_gt_instances_ignore: OptInstanceList = None,
    #                 unmap_outputs: bool = True,
    #                 return_sampling_results: bool = False) -> tuple:
    #     """Compute regression and classification targets for anchors in
    #     multiple images.

    #     Args:
    #         anchor_list (list[list[Tensor]]): Multi level anchors of each
    #             image. The outer list indicates images, and the inner list
    #             corresponds to feature levels of the image. Each element of
    #             the inner list is a tensor of shape (num_anchors, 4).
    #         valid_flag_list (list[list[Tensor]]): Multi level valid flags of
    #             each image. The outer list indicates images, and the inner list
    #             corresponds to feature levels of the image. Each element of
    #             the inner list is a tensor of shape (num_anchors, )
    #         batch_gt_instances (list[:obj:`InstanceData`]): Batch of
    #             gt_instance. It usually includes ``bboxes`` and ``labels``
    #             attributes.
    #         batch_img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #         batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
    #             Batch of gt_instances_ignore. It includes ``bboxes`` attribute
    #             data that is ignored during training and testing.
    #             Defaults to None.
    #         unmap_outputs (bool): Whether to map outputs back to the original
    #             set of anchors. Defaults to True.
    #         return_sampling_results (bool): Whether to return the sampling
    #             results. Defaults to False.

    #     Returns:
    #         tuple: Usually returns a tuple containing learning targets.

    #             - labels_list (list[Tensor]): Labels of each level.
    #             - label_weights_list (list[Tensor]): Label weights of each
    #               level.
    #             - bbox_targets_list (list[Tensor]): BBox targets of each level.
    #             - bbox_weights_list (list[Tensor]): BBox weights of each level.
    #             - avg_factor (int): Average factor that is used to average
    #               the loss. When using sampling method, avg_factor is usually
    #               the sum of positive and negative priors. When using
    #               `PseudoSampler`, `avg_factor` is usually equal to the number
    #               of positive priors.

    #         additional_returns: This function enables user-defined returns from
    #             `self._get_targets_single`. These returns are currently refined
    #             to properties at each feature map (i.e. having HxW dimension).
    #             The results will be concatenated after the end
    #     """
    #     # invoke the superclass
    #     cls_reg_targets = super().get_targets(
    #         anchor_list,
    #         valid_flag_list,
    #         batch_gt_instances,
    #         batch_img_metas,
    #         batch_gt_instances_ignore=batch_gt_instances_ignore)
    #     (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
    #      avg_factor) = cls_reg_targets
        
    #     return
    
    def _get_targets_single(self,
                            flat_anchors: Union[Tensor, BaseBoxes],
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor or :obj:`BaseBoxes`): Multi-level anchors
                of the image, which are concatenated into a single tensor
                or box type of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors, ).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.  Defaults to True.

        Returns:
            tuple:

                - labels (Tensor): Labels of each level.
                - label_weights (Tensor): Label weights of each level.
                - bbox_targets (Tensor): BBox targets of each level.
                - bbox_weights (Tensor): BBox weights of each level.
                - pos_inds (Tensor): positive samples indexes.
                - neg_inds (Tensor): negative samples indexes.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags]

        pred_instances = InstanceData(priors=anchors)
        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)

        # mhf 26.04.24 Assign densities, in similar way to labels
        assigned_densities = copy.deepcopy(assign_result.labels)
        assigned_gt_inds = assign_result.gt_inds
        gt_dens = gt_instances.bbox_densitie
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_densities[pos_inds] = gt_dens[assigned_gt_inds[pos_inds] -
                                                 1]

        # No sampling is required except for RPN and
        # Guided Anchoring algorithms
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)
        


        num_valid_anchors = anchors.shape[0]
        target_dim = gt_instances.bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        bbox_targets = anchors.new_zeros(num_valid_anchors, target_dim)
        bbox_weights = anchors.new_zeros(num_valid_anchors, target_dim)

        # TODO: Considering saving memory, is it necessary to be long?
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        # mhf 26.04.24
        dens_targets = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # `bbox_coder.encode` accepts tensor or box type inputs and generates
        # tensor targets. If regressing decoded boxes, the code will convert
        # box type `pos_bbox_targets` to tensor.
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
                pos_bbox_targets = get_box_tensor(pos_bbox_targets)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)
    
    

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
        # mhf 25.04.24
        # Generate density GT and update batch_gt_instances 
        for n, gt_instance in enumerate(batch_gt_instances):
            bboxes = gt_instance.bboxes
            dens = IoUGenerator().generate_box_density(bboxes)
            batch_gt_instances[n].bbox_densities = dens
        
        
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