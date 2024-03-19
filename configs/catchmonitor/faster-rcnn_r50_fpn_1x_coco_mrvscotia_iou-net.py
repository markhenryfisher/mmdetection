# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 10:34:34 2024

@author: mhf
"""
_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
# And setup nms
# model = dict(
    # roi_head=dict(
        # bbox_head=dict(num_classes=1)),
    # test_cfg=dict(
        # rpn=dict(
            # nms_pre=1000,
            # max_per_img=1000,
            # nms=dict(type='nms'),
            # min_bbox_size=0),
        # rcnn=dict(
            # score_thr=0.05,
            # nms=dict(type='nms', iou_threshold=0.5),
            # max_per_img=100)))
            
# We need to change the num_classes in head to match the dataset's annotation            
model=dict(
    roi_head=dict(
        type='IoURoIHead',
        bbox_head=dict(num_classes=1),
        bbox_roi_extractor=dict(
            _delete_=True,
            type='PrRoIExtractor',
            roi_layer=dict(type='PrRoIPool2D', pooled_height=7, pooled_width=7),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        roi_generator=dict(
            pre_sample=4000,
            xy_steps=16,
            wh_steps=16,
            xy_range=(0, 1),
            area_range=(1/3, 3),
            nonlinearity=2,
            per_iou=None,
            sample_num=1000,
            max_num=1000,
            compensate=None),
        iou_head=dict(
            type='IoUHead',
            in_channels=256 * 7 * 7,
            fc_channels=[1024, 1024],
            num_classes=1,
            class_agnostic=False,
            target_norm=dict(mean=0.5, std=0.5),
            # loss_iou=dict(type='SmoothL1Loss', loss_weight=5.0))),
            loss_iou=dict(type='SmoothL1Loss', loss_weight=1.0))),
    train_cfg=dict(
        max_epochs = 12, type = 'EpochBasedTrainLoop', val_interval=1,
        rcnn=dict(
            _delete_=True,
            bbox_assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            bbox_sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            iou_assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            iou_sampler=dict(type='PseudoSampler'),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rcnn=dict(
            iou=dict(
                nms=dict(multiclass=True, iou_threshold=0.5),
                refine=dict(
                    pre_refine=100,
                    t=5,
                    omega_1=0.001,
                    omega_2=-0.01,
                    lamb=0.5,
                    use_iou_score=True)
                ))))            
            
# Modify learning rate (reduced by factor of 10)
# optimizer
# optim_wrapper = dict(
    # type='OptimWrapper',
    # optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001))

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

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'val/annotation_coco.json')
test_evaluator = val_evaluator

# We can use a pre-trained Faster RCNN model, iou-net trained for a futher 10 epochs  
load_from = './work_dirs/faster-rcnn_iou_r50_fpn_1x_coco/epoch_10.pth'  # noqa

