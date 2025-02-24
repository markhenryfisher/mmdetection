# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:04:34 2025
mmdetection config. for MRVSCOTIA Synthetic Footage.
e.g. useage:    (openmmlab) S:\mhf\GitHub\mmdetection>python tools/test.py 
                configs/catchmonitor/faster-rcnn_r50_fpn_1x_coco_mrvscotia_syn_baseline.py 
                work_dirs/faster-rcnn_r50_fpn_1x_coco_mrvscotia_baseline/epoch_12.pth 
                --show --show-dir val_syn_show --wait-time 10  

@author: mhf
"""
_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
# And setup nms
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms'),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
            
# Modify learning rate (reduced by factor of 10)
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001))

# Modify dataset related settings
data_root = 'data/belt_data_synthetic/Experiment180/mrv_scotia/'
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

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'val/annotation_coco.json')
test_evaluator = val_evaluator

# Modify the training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10))

# We can use the pre-trained Faster RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa

