# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:47:41 2024

@author: mhf
"""
import torch
from mmdet.structures.bbox import bbox_overlaps

class IoUGenerator(torch.nn.Module):
    """
        Generates Box Density - computes density d_i as defined by
        Liu et al. Adaptive NMS: Refining Pedestrian Detection in a Crowd. 2019

    """
    
    def __init__(self):
        super(IoUGenerator, self).__init__()


    def generate_box_density(self, gt_bboxes):
        """     
        """
        if not torch.is_tensor(gt_bboxes):
            raise ValueError('box_density: input boxes is not Tensor')
        
        if len(gt_bboxes) == 0:
            raise ValueError('box_density: empty input boxes')
        elif len(gt_bboxes) == 1:
            result = torch.Tensor([0.0])
        else:
                
            # More efficient to compute (self) iou matrix  
            # iou_mat = box_ops.box_iou(boxes, boxes)
            # replace above with bbox_overlaps
            iou_mat = bbox_overlaps(gt_bboxes, gt_bboxes)
            # set main diagonal to 0.0 (iou of box with itself = 1.0) 
            iou_mat.fill_diagonal_(0.0)
            max_iou_per_gt_bbox, ind = iou_mat.max(dim=1)
            result = max_iou_per_gt_bbox
                          
        return result.to(torch.long)