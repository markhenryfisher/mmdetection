# -*- coding: utf-8 -*-
"""iou_roi_head.py: RoI head with a bbox_head and a iou_head."""

from typing import List, Tuple
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox2roi, scale_boxes
from mmdet.utils import ConfigType, InstanceList
from ..utils import empty_instances, unpack_gt_instances
from mmdet.models.layers import batched_iou_nms

from .standard_roi_head import StandardRoIHead
from mmdet.models.utils.iounet_utils import RoIGenerator
from mmengine.structures import InstanceData
import torch

# import pdb


@MODELS.register_module()
class IoURoIHead(StandardRoIHead):
    """RoIHead for IoUNet.
    
    https://github.com/thisisi3/OpenMMLab-IoUNet/blob/main/mmdet/iounet/iou_roi_head.py
    
    """
    def __init__(self, *args, iou_head=None, roi_generator=None, **kwargs):
        super(IoURoIHead, self).__init__(*args, **kwargs)
        assert iou_head is not None, 'IoU head must be present for StandardIoUHead'
        assert not self.with_shared_head, 'shared head is not supported for now'
        assert not self.with_mask, 'mask is not supported for now'
        self.iou_head = MODELS.build(iou_head)
        self.roi_generator = roi_generator
        if roi_generator is not None:
            self.roi_generator = RoIGenerator(**roi_generator)

    def init_assigner_sampler(self):
        self.bbox_assigner = None
        self.bbox_sampler = None
        self.iou_assigner = None
        self.iou_sampler = None
        if self.train_cfg:
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.bbox_assigner)
            self.bbox_sampler = TASK_UTILS.build(
                self.train_cfg.bbox_sampler, default_args=dict(context=self))
            self.iou_assigner = TASK_UTILS.build(self.train_cfg.iou_assigner)
            self.iou_sampler = TASK_UTILS.build(
                self.train_cfg.iou_sampler, default_args=dict(context=self))


    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
              batch_data_samples: List[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.


        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
            data samples. It usually includes information such
            as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            Note:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
            gt_instance. It usually includes ``bboxes``, ``labels``, and
            ``masks`` attributes.


        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
                - `mask_targets` (Tensor): Mask target of each positive\
                    proposals in the image.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """        
        ################ the RCNN part #################  
        losses = super().loss(x, rpn_results_list,
                  batch_data_samples)
        

        ################ the IoU part #################
        # IoUHead forward and loss
        # next do forward train on iou_head, we follow following pipeline:
        # 1, use RoIGenerator to generate rois, it also controlls number of rois in each iou
        #    so it also does the sampling
        # 2, assign rois to gt_bboxes, use default MaxIoUAssigner
        # 3, do sampling, here we use PseudoSampler, since RoIGenerator has sampling inside.
        # TODO: it may be more reasonable to let sampler do the sampling
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = outputs
        num_imgs = len(batch_data_samples)

        iou_rois_list = []
        iou_sampling_results = []
        # # for i in range(len(img_metas)):
        for i in range(num_imgs):
            gt_bboxes = batch_gt_instances[i].bboxes
            gt_bboxes_ignore = batch_gt_instances_ignore[i].bboxes
                        
            iou_rois = self.roi_generator.generate_roi(
                gt_bboxes, batch_img_metas[i]['img_shape'][:2] )
            iou_rois_list.append(iou_rois)

            # refactor iou_rois as InstanceData
            iou_roi_instances = InstanceData()
            iou_roi_instances.priors = iou_rois
            
            iou_assign_result = self.iou_assigner.assign(
                iou_roi_instances, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
        
            # TODO: Why is this called a Pseudo-sampler?
            iou_sampling_result = self.iou_sampler.sample(
                iou_assign_result,
                iou_roi_instances,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
        
            iou_sampling_results.append(iou_sampling_result)
            
        # mhf - largely guesswork            
        gt_bboxes = [x.bboxes for x in batch_gt_instances]
        gt_labels = [x.labels for x in batch_gt_instances]
        img_metas = [x for x in batch_img_metas]
        
        iou_losses = self._iou_loss(
            x, iou_sampling_results, gt_bboxes, gt_labels, img_metas)
        
        losses.update(iou_losses)
        
        return losses

    def _iou_forward(self, x, rois):
        assert rois.size(1) == 5, 'dim of rois should be [K, 5]'
        iou_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        return self.iou_head(iou_feats)
        
    def _iou_loss(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        """Perform forward propagation and loss calculation of the iou head on
        the features of the upstream network.
        """
        rois = bbox2roi([res.pos_priors for res in sampling_results])
        iou_score = self._iou_forward(x, rois)
        loss_iou = self.iou_head.loss(
            iou_score, sampling_results, gt_bboxes, gt_labels, rois, img_metas)
        return dict(loss_iou=loss_iou)
        


    # since it calls roi_extractor, we put refinement in roi_head
    # TODO: this could be done together for all images
    def refine_by_iou(self, x, bbox, score, label, img_idx, img_meta, cfg):
        """Refine bboxes by gradient of iou_score w.r.t the bboxes in one image"""
        det_bboxes, det_scores, det_ious, det_labels = [], [], [], []
        with torch.set_grad_enabled(True):
            prev_bbox, prev_label, prev_score = bbox, label, score
            prev_bbox.requires_grad_(True)
            bbox_roi = torch.cat(
                [prev_bbox.new_full((prev_bbox.size(0), 1), img_idx), prev_bbox], dim=1)
            prev_iou = self._iou_forward(x, bbox_roi)
            prev_iou = prev_iou[torch.arange(prev_bbox.size(0)), prev_label]
            keep_mask = None
            # in the loop we do:
            #   1, backward to obtain bboxes' grad
            #   2, update bboxes according to the grad
            #   3, forward to obtain iou of new bboxes
            #   4, filter bboxes that need no more refinement
            for i in range(cfg.t):
                if prev_score.size(0) <= 0:
                    break
                # TODO! mhf check if this needs to be commented out?
                # prev_iou.sum().backward()
                prev_bbox_grad = torch.autograd.grad(
                    prev_iou.sum(), prev_bbox, only_inputs=True)[0]
                if keep_mask is not None:
                    # filter bbox and grad after backward
                    bbox_grad = prev_bbox_grad[~keep_mask]
                    prev_bbox = prev_bbox[~keep_mask]
                else:
                    bbox_grad = prev_bbox_grad
                w, h = prev_bbox[..., 2]-prev_bbox[..., 0], prev_bbox[..., 3]-prev_bbox[..., 1]
                scale = torch.stack([w, h, w, h], dim=1)
                delta = cfg.lamb * bbox_grad * scale
                # apply gradient ascent
                new_bbox = prev_bbox + delta
                new_bbox = new_bbox.detach().requires_grad_(True)
                bbox_roi = torch.cat(
                    [new_bbox.new_full((new_bbox.size(0), 1), img_idx), new_bbox], dim=1)
                new_iou = self._iou_forward(x, bbox_roi)
                new_iou = new_iou[torch.arange(new_iou.size(0)), prev_label]
                keep_mask = ((prev_iou - new_iou).abs() < cfg.omega_1) | \
                            ((new_iou - prev_iou) < cfg.omega_2)
                det_bboxes.append(new_bbox[keep_mask])
                det_ious.append(new_iou[keep_mask])
                det_scores.append(prev_score[keep_mask])
                det_labels.append(prev_label[keep_mask])
                # we will filter bbox and its grad after backward in next loop
                # because new_bbox[~keep_mask].grad will be None
                prev_bbox = new_bbox
                prev_iou = new_iou[~keep_mask]
                prev_score = prev_score[~keep_mask]
                prev_label = prev_label[~keep_mask]
            # add the rest of the bboxes
            if prev_score.size(0) > 0:
                det_bboxes.append(prev_bbox[~keep_mask])
                det_scores.append(prev_score)
                det_labels.append(prev_label)
                det_ious.append(prev_iou)
        # mind that det results are not sorted by score
        det_bboxes = torch.cat(det_bboxes)
        det_scores = torch.cat(det_scores)
        det_labels = torch.cat(det_labels)
        det_ious   = torch.cat(det_ious)
        if cfg.use_iou_score:
            det_scores *= det_ious
        return det_bboxes, det_scores, det_labels

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        for i in range(len(proposals)):
            assert proposals[i].size(1) == 4, 'dim of proposals should be [K, 4]'
        rois = bbox2roi(proposals)
        
        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)
        
        bbox_results = self._bbox_forward(x, rois)


        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_preds is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)

        iou_cfg = rcnn_test_cfg.get('iou', None)
        if iou_cfg is None:
            result_list = self.bbox_head.predict_by_feat(
                rois=rois,
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                batch_img_metas=batch_img_metas,
                rcnn_test_cfg=rcnn_test_cfg,
                rescale=rescale)
            return result_list
        
        # base data instance and results list to return info.
        results = InstanceData()
        result_list = []
        # next apply iou_head, it will use some of the configs from rcnn
        for i in range(len(proposals)):
            cur_cls_score = cls_scores[i].softmax(1)[:, :-1] # rm bg scores
            cur_max_score, cur_bbox_label = cur_cls_score.max(1)

            # TODO! mhf refactor legacy code
            regressed = self.bbox_head.legacy_regress_by_class(
                rois[i], cur_bbox_label, bbox_preds[i], batch_img_metas[i])
            
            cur_iou_score = self._iou_forward(x, regressed)

            if iou_cfg.nms.multiclass:
                nms_cls_score = cur_cls_score.reshape(-1)
                nms_iou_score = cur_iou_score.view(-1)
                nms_regressed = regressed[:, 1:].view(-1, 1, 4).repeat(
                    1, self.bbox_head.num_classes, 1).view(-1, 4)
                nms_label = torch.arange(
                    self.bbox_head.num_classes,
                    device=nms_cls_score.device).repeat(rois[i].size(0))
            else:
                nms_cls_score = cur_max_score
                nms_iou_score = cur_iou_score[
                    torch.arange(cur_iou_score.size(0)), cur_bbox_label]
                nms_regressed = regressed[:, 1:]
                nms_label = cur_bbox_label
                
            # apply iou_nms
            det_bbox, det_score, det_iou, det_label = batched_iou_nms(
                nms_regressed, nms_cls_score, nms_iou_score, nms_label,
                iou_cfg.nms.iou_threshold, rcnn_test_cfg.score_thr,
                guide=iou_cfg.nms.get('guide', 'rank'))
            
            if iou_cfg.get('refine', None) is not None and det_bbox.size(0) > 0:
                det_bbox  = det_bbox[:iou_cfg.refine.pre_refine]
                det_score = det_score[:iou_cfg.refine.pre_refine]
                det_iou   = det_iou[:iou_cfg.refine.pre_refine]
                det_label = det_label[:iou_cfg.refine.pre_refine]
                det_bbox, det_score, det_label = self.refine_by_iou(
                    x, det_bbox, det_score, det_label, i, batch_img_metas[i],
                    iou_cfg.refine)
                                        
            if rescale and det_bbox.size(0) > 0:
                img_meta = batch_img_metas[i]
                assert img_meta.get('scale_factor') is not None
                scale_factor = [1 / s for s in img_meta['scale_factor']]
                det_bbox = scale_boxes(det_bbox, scale_factor)


            det_score, srt_idx = det_score.sort(descending=True)
            det_bbox = det_bbox[srt_idx]
            det_label = det_label[srt_idx]
            # mhf why append score?
            # det_bbox = torch.cat([det_bbox, det_score.view(-1, 1)], dim=1)

            # assemble results
            results.scores = det_score
            results.labels = det_label
            results.bboxes = det_bbox
            result_list.append(results)
        
        return result_list
