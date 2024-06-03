_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model=dict(
    rpn_head=dict(
        type='AdaptiveNMSHead',
        loss_dns=dict(type='SmoothL1Loss', loss_weight=1.0)
        ),
    roi_head=dict(
        type='AdaptNMSRoIHead',
        ),
    train_cfg=dict(
        rpn=dict(
            # dpn_mode=dict(type='const', value=0.7)
            dpn_mode=None
            ),
        max_epochs = 4, type = 'EpochBasedTrainLoop', val_interval=1),
    test_cfg=dict(
        rcnn=dict(
            nms=dict(type='adaptive_nms')
            )))

# currently using default test_cfg       
    # test_cfg=dict(
        # rpn=dict(
            # nms_pre=1000,
            # max_per_img=1000,
            # nms=dict(type='nms', iou_threshold=0.7),
            # min_bbox_size=0),
        # rcnn=dict(
            # score_thr=0.05,
            # nms=dict(type='nms', iou_threshold=0.5),
            # max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)

                
# Modify learning rate (reduced by factor of 10)
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)) 

# For better, more stable performance initialize from COCO                    
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa

# Use a Faster RCNN model, adaptive-nms-net pre-trained on coco for 12 epochs  
# load_from = './work_dirs/faster-rcnn_nms_r50_fpn_1x_coco/epoch_12.pth' 