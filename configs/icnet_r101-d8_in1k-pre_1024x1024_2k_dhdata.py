norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ICNet',
        backbone_cfg=dict(
            type='ResNetV1c',
            in_channels=3,
            depth=101,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet101_v1c')),
        in_channels=3,
        layer_channels=(512, 2048),
        light_branch_middle_channels=32,
        psp_out_channels=512,
        out_channels=(64, 256, 256),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False),
    neck=dict(
        type='ICNeck',
        in_channels=(64, 256, 256),
        out_channels=128,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=128,
        channels=128,
        num_convs=1,
        in_index=2,
        dropout_ratio=0,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        concat_input=False,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=128,
            num_convs=1,
            num_classes=2,
            in_index=0,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=128,
            num_convs=1,
            num_classes=2,
            in_index=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'DhSegDataset'
data_root = '/home/zwang/dh/dhdata/manual_data/'
crop_size = (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='DhSegDataset',
        data_root=data_root,
        img_dir='images/train',
        ann_dir='ann/train',
        pipeline=train_pipeline),
    val=dict(
        type='DhSegDataset',
        data_root=data_root,
        img_dir='images/val',
        ann_dir='ann/val',
        pipeline=test_pipeline),
    test=dict(
        type='DhSegDataset',
        data_root=data_root,
        img_dir='images/val',
        ann_dir='ann/val',
        pipeline=test_pipeline))

log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=2000)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=500, metric='mIoU', pre_eval=True)
work_dir = './work_dirs/icnet_r101-d8_in1k-pre_832x832_80k_cityscapes'
gpu_ids = [0]
auto_resume = False
