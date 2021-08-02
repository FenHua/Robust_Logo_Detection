model = dict(
    type='CascadeRCNN',
    # 须提前下载预训练检查点，上传服务器
    pretrained='/apdcephfs/share_1290939/jiaxiaojun/Open_brand/checkpoints/resnet50-19c8e357.pth',
    backbone=dict(
        type='DetectoRS_ResNet',
        depth=50,       # resnet 50
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
    # FPN->RFP
    neck=dict(
        type='RFP',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            # resnet-50 imagenet预训练模型
            pretrained='/apdcephfs/share_1290939/jiaxiaojun/Open_brand/checkpoints/resnet50-19c8e357.pth',
            style='pytorch')),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[1.0, 1.5, 2.5, 3.0],   # kmeans聚类选取
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),     # 使用交叉熵完成前景背景的分类
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    # cascade级联结构
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=515,               # 类别数修改
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(type='EQLv2'),  # 均衡损失
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),  # CIOU,DIOU,GIOU 无效
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=515,               # 类别数修改
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(type='EQLv2'),  # 均衡损失
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),  # CIOU,DIOU,GIOU 无效
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=515,              # 类别数修改
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(type='EQLv2'),  # 均衡损失
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))  # CIOU,DIOU,GIOU 无效
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,               # 修改后无效
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,           # 修改后无效
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,         # 修改后无效
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,       # 修改后无效
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=5000,          # 越高越不易漏检 2000
            max_per_img=5000,      # 越高越不易漏检 2000
            nms=dict(type='nms', iou_threshold=0.7),   # 0.65 涨点
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.01,        # 越低虚景越多，轻微涨点  0.05
            nms = dict(type='soft_nms', iou_thr=0.5, min_score=0.001),
            #nms=dict(type='nms', iou_threshold=0.5),  # soft-nms 有效
            max_per_img=100)))     # 默认最佳
dataset_type = 'OpenBrandDataset'
data_root = 'data/MM2021/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        # 训练阶段的多尺度
        img_scale=[(1333, 800), (1333, 900), (1333, 1000), (1333, 1100)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='RGBShift',
                        r_shift_limit=10,
                        g_shift_limit=10,
                        b_shift_limit=10,
                        p=1.0),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0)
                ],
                p=0.1),
            dict(
                type='JpegCompression',
                quality_lower=85,
                quality_upper=95,
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                # 提点
                type='OneOf',
                transforms=[
                    dict(type='MotionBlur', p=1.0),
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='RandomFog', fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.1, p=1.0),
                    dict(type='RandomRain', brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1.0)
                ],
                p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='IAAAdditiveGaussianNoise'),
                    dict(type='GaussNoise')
                ],
                p=0.1)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='RandomFlip', flip_ratio=0.5),  # 翻转
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 800), (1333, 864), (1333, 928), (1333,992),
                   (1333, 1056), (1333,1120), (1333,1184),(1333,1248)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),   # 32倍数
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.002,    # 重采样
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/openbrand_train.json',
            img_prefix=data_root + 'train-images/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Resize',
                    img_scale=[(1333, 800), (1333, 900), (1333, 1000), (1333, 1100)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='Albu',
                    transforms=[
                        dict(
                            type='ShiftScaleRotate',
                            shift_limit=0.0625,
                            scale_limit=0.0,
                            rotate_limit=0,
                            interpolation=1,
                            p=0.5),
                        dict(
                            type='RandomBrightnessContrast',
                            brightness_limit=[0.1, 0.3],
                            contrast_limit=[0.1, 0.3],
                            p=0.2),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(
                                    type='RGBShift',
                                    r_shift_limit=10,
                                    g_shift_limit=10,
                                    b_shift_limit=10,
                                    p=1.0),
                                dict(
                                    type='HueSaturationValue',
                                    hue_shift_limit=20,
                                    sat_shift_limit=30,
                                    val_shift_limit=20,
                                    p=1.0)
                            ],
                            p=0.1),
                        dict(
                            type='JpegCompression',
                            quality_lower=85,
                            quality_upper=95,
                            p=0.2),
                        dict(type='ChannelShuffle', p=0.1),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(type='MotionBlur', p=1.0),
                                dict(type='Blur', blur_limit=3, p=1.0),
                                dict(type='MedianBlur', blur_limit=3, p=1.0)
                            ],
                            p=0.1),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(type='RandomFog', fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.1, p=1.0),
                                dict(type='RandomRain', brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1.0)
                            ],
                            p=0.1),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(type='IAAAdditiveGaussianNoise'),
                                dict(type='GaussNoise')
                            ],
                            p=0.1)
                    ],
                    bbox_params=dict(
                        type='BboxParams',
                        format='pascal_voc',
                        label_fields=['gt_labels'],
                        min_visibility=0.0,
                        filter_lost_elements=True),
                    keymap=dict(img='image', gt_bboxes='bboxes'),
                    update_pad_shape=False,
                    skip_img_without_anno=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ])),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_brand_val_hybrid2.json',
        img_prefix=data_root + 'brand_val_images_hybrid_2/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 1000),
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/testB_imgList.json',
        img_prefix=data_root + 'testB-images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(1333, 800), (1333, 864), (1333, 928), (1333,992),
                   (1333, 1056), (1333,1120), (1333,1184),(1333,1248)],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8,12,16])
runner = dict(type='EpochBasedRunner', max_epochs=18)
checkpoint_config = dict(interval=1)
log_config = dict(interval=64, hooks=[dict(type='TextLoggerHook')])
#custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from="/apdcephfs/share_1290939/jiaxiaojun/OpenBrandData/output_detectors_ratio-multinode/epoch_14.pth"
resume_from=None
workflow = [('train', 1)]
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='MotionBlur',p=1.0),
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
    # 加入自然干扰
    dict(
        type='OneOf',
        transforms=[
            dict(type='RandomFog', fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.1, p=1.0),
            dict(type='RandomRain', brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1.0)
        ],
        p=0.1),
    # 加高斯噪声
    dict(
        type='OneOf',
        transforms=[
            dict(type='IAAAdditiveGaussianNoise'),
            dict(type='GaussNoise')
        ],
        p=0.1)
]
work_dir = '/apdcephfs/share_1290939/jiaxiaojun/OpenBrandData/outputEnd_detectors_r50'

