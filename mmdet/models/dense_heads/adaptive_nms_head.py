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
from mmengine.config import ConfigDict
from mmcv.ops import batched_nms

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptInstanceList
from mmdet.structures.bbox import (BaseBoxes, cat_boxes, empty_box_as, get_box_tensor, 
                                   get_box_wh, scale_boxes)

from .rpn_head import RPNHead

from ..task_modules.prior_generators import anchor_inside_flags
from ..utils import images_to_levels, multi_apply, unmap

from mmdet.models.utils.adaptnms_utils import IoUGenerator

from ..utils import (select_single_mlvl)

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

        self.num_convs = num_convs
        assert num_classes == 1
        self.loss_dns_cfg = loss_dns
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)
       
        
    def _init_layers(self) -> None:
        # put density loss here
        self.loss_dns = MODELS.build(self.loss_dns_cfg)
        self.dns_out_channels = 1
        super(AdaptiveNMSHead, self)._init_layers() 

        # init the density prediction regressor
        in_channels = self.in_channels
        num_anchors = self.num_base_priors
        reg_dim = self.bbox_coder.encode_size
        cls_out_channels = self.cls_out_channels
        n_dns_channels = in_channels + (num_anchors * reg_dim) \
            + (num_anchors * cls_out_channels) 
        self.dpn_reg = nn.Conv2d(n_dns_channels, num_anchors, kernel_size=5, padding=2, stride=1)

        # for layer in self.modules():
        #     print(layer)
        
        layer = self.dpn_reg
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

        
    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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
       
        dns_input = torch.cat((x, rpn_cls_score, rpn_bbox_pred), dim=1)        
        dpn_dns_pred = self.dpn_reg(dns_input)
        
        # mhf 03.05.24 Check shape of dns_pred is consistent with cls_score 
        assert rpn_cls_score.shape == dpn_dns_pred.shape
        
        return rpn_cls_score, rpn_bbox_pred, dpn_dns_pred
    

    # mhf _get_targets_single overrides AnchorHead method
    # Invoked by AnchorHead.get_targets 
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
        # pdb.set_trace()
        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)

        # mhf 26.04.24 create tensor to store gt density 
        # cf max_iou_asigner.py...
        # assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        # pos_inds = torch.nonzero(
        #     assigned_gt_inds > 0, as_tuple=False).squeeze()
        # if pos_inds.numel() > 0:
        #     assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] -
        #                                           1]

        assigned_labels = assign_result.labels
        assigned_gt_inds = assign_result.gt_inds
        # mhf 23.05.24 bugfix
        # assigned_densities = \
        #     assigned_gt_inds.new_full((assigned_labels.size(0), ), -1 )
        assigned_densities = \
            assigned_gt_inds.new_full((assigned_labels.size(0), ), 0.0, dtype=torch.float )
        gt_dens = gt_instances.bbox_densities.to(assigned_gt_inds.device)
        
        # mhf assign gt dens - refer to assigned_labels (comment above)
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_densities[pos_inds] = gt_dens[assigned_gt_inds[pos_inds] -
                                                  1]
        # pdb.set_trace()    
        # mhf 02.05.24 check assigned densities and assigned_labels 
        # have same anchors
        # TODO bugfix 
        # assert torch.all(torch.eq(torch.nonzero(assigned_labels > -1), 
        #     torch.nonzero(assigned_densities > -1)))
        
        # Add user defined property `gt_dens' (gt density)
        assign_result._extra_properties['gt_dens']  = assigned_densities

        # No sampling is required except for RPN and
        # Guided Anchoring algorithms
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)
        
        # mhf 26.04.24 cf sampling_result.pos_gt_labels
        pos_gt_dens = assigned_densities[sampling_result.pos_inds]

        # mhf 03.05.24 check sampling of gt_dens is consistent with labels
        assert sampling_result.pos_gt_labels.size() == pos_gt_dens.size()


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

        # mhf 02.05.24 add densities. Note: Value of background densities is
        # 0. Value of foreground will be gt_density (i.e. value between 0.0 and 0.99).
        # mhf 23.05.24 bugfix datatype must be float
        densities = anchors.new_zeros((num_valid_anchors, ),
                                  dtype=torch.float)

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
            
            # mhf 29.04.24
            densities[pos_inds] = pos_gt_dens
                                
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            
            # mhf 29.04.24 guesswork
            densities = unmap(
                densities, num_total_anchors, inside_flags,
                fill=0)
            
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        # mhf check labels vs densities

        # mhf Note: last element in the tuple is user defined
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result, densities)


    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            dens_pred: Tensor,
                            anchors: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor, densities: Tensor,
                            avg_factor: int) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            dens_pred (Tensor): Box densities
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
            densities: Densities of each anchors with shape
                (N, num_total_anchors)
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
        
        # mhf 03.05.24 
        # density loss
        # pdb.set_trace()
        densities = densities.reshape(-1)
        dens_pred = dens_pred.permute(0, 2, 3,
                                    1).reshape(-1)

        # mhf This is how torchvision version finds density loss
        # densityness_loss = F.smooth_l1_loss(densityness[sampled_inds], 
        #                                     density_targets[sampled_inds])

        # mhf This was first attempt
        # loss_dens = self.loss_dns(
        #     dens_pred, densities, label_weights, avg_factor=avg_factor)
        
        # mhf For degug we can set ALL NMS to a value (e.g. 0.7)
        dpn_mode = self.train_cfg.get('dpn_mode', None)
        if dpn_mode is not None:
            assert dpn_mode['type'] in ['const', ]
            if dpn_mode['type'] == 'const':                
                const_densities = torch.ones_like(label_weights)  * dpn_mode['value']
                loss_dens = F.smooth_l1_loss(dens_pred[:], const_densities[:])
                
        else:
            loss_dens = self.loss_dns(
                dens_pred, densities, label_weights, avg_factor=avg_factor)

        # print('Loss= {:.2f}'.format(loss_dens))
        
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
        return loss_dens, loss_cls, loss_bbox
    
    
    
    # mhf loss_by_feat overrides RPNHead.loss_by_feat
    # Invoked by BaseDensityHead.loss_and_predict
    def loss_by_feat(self, 
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     dens_preds: List[Tensor],
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
        # mhf 23.05.24 Fixed bug in IoUGenerator
        # pdb.set_trace()
        for n, gt_instance in enumerate(batch_gt_instances):
            bboxes = gt_instance.bboxes
            dens = IoUGenerator().generate_box_density(bboxes)
            batch_gt_instances[n].bbox_densities = dens
        
        # pdb.set_trace()
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        
        # print('getting targets..')
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor, densities_list) = cls_reg_targets

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        # print('calculating losses...')
        losses_dens, losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            dens_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            densities_list,
            avg_factor=avg_factor)
        
        # print('\n')
        
        losses = dict(loss_dens=losses_dens, loss_cls=losses_cls, loss_bbox=losses_bbox)
        
        
        return dict(
            loss_rpn_dens=losses['loss_dens'], loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])


    # mhf 01.05.24 New _pedict_by_feat_single for Adaptive NMS
    # Note: dens_pred_list param.
    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                dens_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
                
            dens_pred_list

                
            score_factor_list (list[Tensor]): Be compatible with
                BaseDenseHead. Not used in RPNHead.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (ConfigDict, optional): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        # pdb.set_trace()
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        # mhf 01.05.24 
        mlvl_dens = []
        
        level_ids = []
        # mhf 01.05.24
        for level_idx, (cls_score, bbox_pred, dens_pred, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, dens_pred_list,
                              mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # mhf 01.05.25
            assert dens_pred.size()[-2] == cls_score.size()[-2]

            reg_dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, reg_dim)
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0] since mmdet v2.0
                # BG cat_id: 1
                scores = cls_score.softmax(-1)[:, :-1]

            scores = torch.squeeze(scores)
            
            # mhf 01.05.24 guesswork                    
            dens_pred = dens_pred.permute(1, 2, 0).reshape(-1, self.dns_out_channels)
            # mhf 15.05.24 take sigmoid!!!
            dens_pred = dens_pred.sigmoid()
            # mhf 01.05.24
            dens_pred = torch.squeeze(dens_pred)
            
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                bbox_pred = bbox_pred[topk_inds, :]
                priors = priors[topk_inds]
                # mhf 01.05.24
                dens_pred = dens_pred[topk_inds]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            # mhf 01.05.24
            mlvl_dens.append(dens_pred)

            # use level id to implement the separate level nms
            level_ids.append(
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.level_ids = torch.cat(level_ids)
        # mhf 01.05.24
        results.dens = torch.cat(mlvl_dens)

        return self._bbox_post_process(
            results=results, cfg=cfg, rescale=rescale, img_meta=img_meta)



    
    # mhf 01.05.24 New predict_by_feat for Adaptive NMS
    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        dens_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        # pdb.set_trace()
        assert len(cls_scores) == len(bbox_preds)
        # mhf 01.05.24
        assert len(dens_preds) == len(cls_scores)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)

            # mhf 01.05.24 do same for dens_preds
            dens_pred_list = select_single_mlvl(
                dens_preds, img_id, detach=True)
            
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            # print('predicting batch {}...'.format(img_id))
            # mhf add dens_pred_list
            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                dens_pred_list=dens_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list    
    

    # mhf 01.05.24 New _bbox_post_process for Adaptive NMS
    # Invoked by self.predict_by_feat_single    
    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        # print('post_processing bboxes...')
        
        assert with_nms, '`with_nms` must be True in RPNHead'
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.level_ids, cfg.nms)
            # mhf comment I think this filters all results
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]
            # TODO: This would unreasonably show the 0th class label
            #  in visualization
            results.labels = results.scores.new_zeros(
                len(results), dtype=torch.long)
            del results.level_ids
        else:
            # To avoid some potential error
            results_ = InstanceData()
            results_.bboxes = empty_box_as(results.bboxes)
            results_.scores = results.scores.new_zeros(0)
            results_.labels = results.scores.new_zeros(0)
            # mhf 01.05.24
            results_.dens = results.dens.new_zeros(0)
            results = results_
        return results