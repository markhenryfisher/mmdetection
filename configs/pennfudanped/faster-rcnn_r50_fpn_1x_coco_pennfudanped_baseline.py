# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:58:34 2024

@author: mhf
https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#train-with-customized-datasets
"""
# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)))
        
# Modify learning rate (reduced by factor of 10)
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001))

# Modify dataset related settings
data_root = 'data/PennFudanPed/'
metainfo = {
    'classes': ('PASpersonWalking', ),
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

# We can use the pre-trained Faster RCNN model to obtain higher performance
# same as catchmonitor baseline.
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
