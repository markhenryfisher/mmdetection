# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:04:34 2024

@author: mhf
"""
_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'


# Modify model for adaptive nms            
model=dict(
    rpn_head=dict(
        type='AdaptiveNMSHead',
        loss_dns=dict(type='SmoothL1Loss', loss_weight=1.0),
        ),
    roi_head=dict(
        type='AdaptNMSRoIHead',
        bbox_head=dict(num_classes=1)
        ),
    train_cfg=dict(
        rpn=dict(
            # dpn_mode=dict(type='const', value=0.7)
            dpn_mode=None
            )),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.5,
            nms=dict(type='adaptive_nms'),
            # nms=dict(type='smpl_nms', iou_threshold=0.7),
            max_per_img=100)
        # Note: multi-class nms is not supported)
        # Simple (class agnostic) nms is supported for rcnn testing:
        # e.g. nms=dict(type='smpl_nms', iou_threshold=0.5),
            ))
            
# Modify learning rate (reduced by factor of 10)
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001))

# Modify dataset related settings
data_root = 'data/belt_data_natural/MRV SCOTIA/'
metainfo = {
    'classes': ('fish_unknown', ),
    'palette': [
        (220, 20, 60),
    ]
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annotation_coco.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

# Modify training schedule max_epochs 
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'val/annotation_coco.json')
test_evaluator = val_evaluator

# We can use the pre-trained Faster RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
# load_from = './work_dirs/faster-rcnn_nms_r50_fpn_1x_coco/epoch_1.pth'  # noqa
