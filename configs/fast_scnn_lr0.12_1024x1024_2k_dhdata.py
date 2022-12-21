norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.01)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='FastSCNN',
        downsample_dw_channels=(32, 48),
        global_in_channels=64,
        global_block_channels=(64, 96, 128),
        global_block_strides=(2, 2, 1),
        global_out_channels=128,
        higher_in_channels=64,
        lower_in_channels=128,
        fusion_out_channels=128,
        out_indices=(0, 1, 2),
        norm_cfg=dict(type='SyncBN', requires_grad=True, momentum=0.01),
        align_corners=False),
    decode_head=dict(
        type='DepthwiseSeparableFCNHead',
        in_channels=128,
        channels=128,
        concat_input=False,
        num_classes=2,
        in_index=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True, momentum=0.01),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=32,
            num_convs=1,
            num_classes=2,
            in_index=-2,
            norm_cfg=dict(type='SyncBN', requires_grad=True, momentum=0.01),
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=64,
            channels=32,
            num_convs=1,
            num_classes=2,
            in_index=-3,
            norm_cfg=dict(type='SyncBN', requires_grad=True, momentum=0.01),
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4))
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
optimizer = dict(type='SGD', lr=0.12, momentum=0.9, weight_decay=4e-05)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=2000)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=500, metric='mIoU', pre_eval=True)
work_dir = './work_dirs/fast_scnn_lr0.12_8x4_160k_cityscapes'
gpu_ids = [0]
auto_resume = False
