_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model=dict(  
    roi_head=dict(
        type='AdaptiveNMSRoIHead',
        bbox_roi_extractor=dict(
            _delete_=True,
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        iou_generator=dict(),
        nms_head=dict(
            type='NMSHead',
            in_channels=256 * 7 * 7,
            fc_channels=[1024, 1024],
            num_classes=80,
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
            # iou_assigner=dict(
                # type='MaxIoUAssigner',
                # pos_iou_thr=0.5,
                # neg_iou_thr=0.5,
                # min_pos_iou=0.5,
                # match_low_quality=False,
                # ignore_iof_thr=-1),
            # iou_sampler=dict(type='PseudoSampler'),
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

# For better, more stable performance initialize from COCO                    
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
