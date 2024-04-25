_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model=dict(
    rpn_head=dict(
        type='AdaptiveNMSHead',
        loss_dns=dict(type='SmoothL1Loss', loss_weight=1.0)),
        ),
    train_cfg=dict(
        max_epochs = 12, type = 'EpochBasedTrainLoop', val_interval=1),
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

# For better, more stable performance initialize from COCO                    
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
