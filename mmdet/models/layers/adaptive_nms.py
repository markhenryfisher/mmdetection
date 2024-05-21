# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:39:26 2024

@author: mhf
"""
import torch
# mhf 17.05.24 check this gets IoUs
from mmdet.structures.bbox import bbox_overlaps

def adaptive_nms(boxes, scores, labels, nms_scores, nms_cfg):
    """
    Launch soft_nms in adaptive mode

    Args:
        boxes (torch.Tensor or np.ndarray): boxes in shape (N, 4).
        scores (torch.Tensor or np.ndarray): scores in shape (N, ).

    Returns:
        tuple: kept dets and indice.

        - boxes (Tensor): Bboxes with score after nms, has shape
          (num_bboxes, 5). last dimension 5 arrange as
          (x1, y1, x2, y2, score)
        - keep (Tensor): The indices of remaining boxes in input
          boxes.

    """
    nms_cfg_ = nms_cfg.copy()
    
    nms_mode = nms_cfg_.pop('type', 'adaptive_nms')
    
    
    
    inds = soft_nms(boxes, scores, nms_scores, nms_mode)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    
    return dets, inds
    

def hard_nms(dets, scores, iou_threshold):
    # dets: bounding box (x1,y1,x2,y2)
    # scores: bounding box confidence score
    # iou_threshold: IoU
    
    # 1. sort confidence scores.
    _, order = scores.sort(0, descending=True)
    
    # 2. Select the prediction with the highest score; remove it and add it to
    #    the final predictions list 'keep'. 'keep' is empty initially. 
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order
        else:
            i = order[0]
        keep.append(i)
    
        if order.numel() == 1:
            break
    
        # Compare this prediction with all others. Calculate the IoU of this
        # prediction with all others. If the IoU is greater than the threshold
        # then remove it.
        ovr = bbox_overlaps(dets[i].unsqueeze(0), dets[order[1:]]).squeeze()
        inds = torch.nonzero(ovr < iou_threshold).squeeze()
        if inds.numel() == 0:
            break
        # order, dets, scores overlap bbox.
        order = order[inds + 1]
    return torch.LongTensor(keep) 


def soft_nms(dets, scores, iou_threshold, method='greedy', sigma=0.5, score_thr=None, debug=False):
    """
    Based on:
    https://github.com/OneDirection9/soft-nms/blob/master/py_nms.py
    """
    if method not in ['adaptive_nms']:
        raise ValueError('soft_nms method must be adaptive_nms')

    # mhf 07.12.23 incorporate adaptive nms
    # create a tensor of (repeated) iou_thresholds - similar to density 
    if method != 'adaptive_nms':
        iou_threshold = [iou_threshold for i in range(len(scores))]
        iou_threshold = torch.tensor(iou_threshold, dtype=torch.float)
        
    if score_thr is None:
        score_thr = torch.min(scores)
    
    ind = torch.arange(len(scores)).unsqueeze(1).long()

    if scores.is_cuda:
        ind = ind.cuda()
        iou_threshold = iou_threshold.cuda()
    
    # add 2 columns to dets (dets[0-3] = box coords; dets[4] = score; dets[5] = index)    
    # dets = torch.cat((dets, scores.unsqueeze(1), ind), 1)
    # mhf 07.12.23 add extra column to dets to account for iou_threshold i.e.
    # (dets[0-3] = box coords; dets[4] = score; dets[5] = iou_threshold, dets[6] = index)
    dets = torch.cat((dets, scores.unsqueeze(1), iou_threshold.unsqueeze(1), ind), 1)
    
    keep = []
#    while dets.numel() > 0:
    while dets.size(dim=0) > 0:
        if debug:
            temp = [int(i.item()) for i in keep]
            print('size: dets= {}\n keep= {}'.format(dets.size(dim=0), temp))
        
        max_idx = torch.argmax(dets[:, 4], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        # mhf 07.12.23 iou_thresh is now dets[0, 5]
        # keep.append(dets[0, 5])
        keep.append(dets[0, 6])
        
        if dets.size(dim=0) == 1:
            break
       
        # calculate iou
        iou = bbox_overlaps(dets[0, 0:4].unsqueeze(0), dets[1:, 0:4]).squeeze()

        # mhf 07.12.23 recover the threshold
        iou_threshold = dets[0, 5]

        if method == 'linear':
            weight = torch.ones_like(iou)
            # weight[iou > iou_thr] -= iou[iou > iou_thr]
            weight[iou >= iou_threshold] -= iou[iou >= iou_threshold]
        elif method == 'gaussian':
            # weight = np.exp(-(iou * iou) / sigma)
            weight = torch.exp(-(iou * iou) / sigma)
        else:  # traditional (greedy) nms or adaptive nms 
            weight = torch.ones_like(iou)
            weight[iou >= iou_threshold] = 0
            
        # print(dets[1:10, 4])        
        dets[1:, 4] *= weight
        # print(dets[1:10, 4])
        retained_idx = torch.where(dets[1:, 4] >= score_thr)[0]
        dets = dets[retained_idx + 1, :]
    
    return torch.Tensor(keep).to(torch.int)
    
    

# def py_soft_nms(dets, method='linear', iou_thr=0.3, sigma=0.5, score_thr=0.001):
#     """Pure python implementation of soft NMS as described in the paper
#     `Improving Object Detection With One Line of Code`_.

#     Args:
#         dets (numpy.array): Detection results with shape `(num, 5)`,
#             data in second dimension are [x1, y1, x2, y2, score] respectively.
#         method (str): Rescore method. Only can be `linear`, `gaussian`
#             or 'greedy'.
#         iou_thr (float): IOU threshold. Only work when method is `linear`
#             or 'greedy'.
#         sigma (float): Gaussian function parameter. Only work when method
#             is `gaussian`.
#         score_thr (float): Boxes that score less than the.

#     Returns:
#         numpy.array: Retained boxes.

#     .. _`Improving Object Detection With One Line of Code`:
#         https://arxiv.org/abs/1704.04503
#     """
#     if method not in ('linear', 'gaussian', 'greedy'):
#         raise ValueError('method must be linear, gaussian or greedy')

#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]

#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     # expand dets with areas, and the second dimension is
#     # x1, y1, x2, y2, score, area
#     dets = np.concatenate((dets, areas[:, None]), axis=1)

#     retained_box = []
#     while dets.size > 0:
#         max_idx = np.argmax(dets[:, 4], axis=0)
#         dets[[0, max_idx], :] = dets[[max_idx, 0], :]
#         retained_box.append(dets[0, :-1])

#         xx1 = np.maximum(dets[0, 0], dets[1:, 0])
#         yy1 = np.maximum(dets[0, 1], dets[1:, 1])
#         xx2 = np.minimum(dets[0, 2], dets[1:, 2])
#         yy2 = np.minimum(dets[0, 3], dets[1:, 3])

#         w = np.maximum(xx2 - xx1 + 1, 0.0)
#         h = np.maximum(yy2 - yy1 + 1, 0.0)
#         inter = w * h
#         iou = inter / (dets[0, 5] + dets[1:, 5] - inter)

#         if method == 'linear':
#             weight = np.ones_like(iou)
#             weight[iou > iou_thr] -= iou[iou > iou_thr]
#         elif method == 'gaussian':
#             weight = np.exp(-(iou * iou) / sigma)
#         else:  # traditional nms
#             weight = np.ones_like(iou)
#             weight[iou > iou_thr] = 0

#         dets[1:, 4] *= weight
#         retained_idx = np.where(dets[1:, 4] >= score_thr)[0]
#         dets = dets[retained_idx + 1, :]

#     return np.vstack(retained_box)
