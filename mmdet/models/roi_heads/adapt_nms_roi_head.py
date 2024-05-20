# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi, get_box_tensor
from mmdet.utils import ConfigType, InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from .standard_roi_head import StandardRoIHead
from mmdet.models.layers import multiclass_nms
from mmengine.structures import InstanceData

import pdb


# def nms_score2roi(nms_score_list: List[Tensor]) -> Tensor:
# def nms_score2roi(nms_score_list):
#     """Convert a list of nms_scores to roi format. cf bbox2roi

#     Args:
#         nms_score_list (List[Tensor]): a list of nms_scores
#             corresponding to a batch of images.
# score
#     Returns:
#         Tensor: shape (n, 2),  Each row of data
#         indicates [batch_ind, score].
#     """
#     # pdb.set_trace()
#     rois_list = []
#     for img_id, scores in enumerate(nms_score_list):
#         img_inds = scores.new_full((scores.size(0), 1), img_id)
#         scores=torch.unsqueeze(scores,1)
#         rois = torch.cat([img_inds, scores], dim=-1)
#         rois_list.append(rois)
#     rois = torch.cat(rois_list, 0)
#     return rois


@MODELS.register_module()
class AdaptNMSRoIHead(StandardRoIHead):
    """RoIHead for IoUNet."""
    
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
        # pdb.set_trace()
        proposals = [res.bboxes for res in rpn_results_list]
        # mhf 17.05.24 recover the densities from the dpn
        nms_scores_list = [res.dens for res in rpn_results_list] 
        # mhf 17.05.24 put scores in roi format
        # nms_scores = nms_score2roi(nms_scores_list)
        
        rois = bbox2roi(proposals)
        
        # mhf 17.05.24 box_dim should be 4
        box_dim = rois.size(-1) -1

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
        # mhf 17.05.24 guess we do same with nms_scores!!!!! No
        # nms_scores = nms_scores.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_preds will be None
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

        # mhf 17.05.24 set rcnn_test_cfg = None to disable nms in bbox_head
        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            # rcnn_test_cfg=rcnn_test_cfg,
            rcnn_test_cfg=None,
            rescale=rescale)
        
        pdb.set_trace()
        # mhf 17.05.24 apply batched nms filter (later change to adaptive)
        nms_results = InstanceData()
        nms_result_list=[]
        for img_id in range(len(batch_img_metas)):
            bboxes = result_list[img_id].bboxes
            scores = result_list[img_id].scores
            
            # mhf 17.05.24 put nms_scores into same format as scores
            nms_scores = nms_scores_list[img_id].unsqueeze(1)           
            nms_scores = nms_scores.repeat_interleave(scores.size(1), dim=1)
            # nms_scores = nms_scores[:scores.size(0), :]

            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim,
                multi_nms_scores=nms_scores)
            nms_results.bboxes = det_bboxes[:, :-1]
            nms_results.scores = det_bboxes[:, -1]
            nms_results.labels = det_labels
            
            nms_result_list.append(nms_results)
            
            
                    
        return nms_result_list    