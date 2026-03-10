_base_ = ["../_base_/datasets/cuhk_detection.py", "../_base_/default_runtime.py"]

train_pipeline = None

model = dict(
    type="PSTR",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(type="PSTRMapper", in_channels=[512, 1024, 2048], out_channels=256),
    bbox_head=dict(
        type="PSTRHead",
        num_query=100,
        num_person=5532,
        queue_size=5000,
        flag_tri=True,
        num_classes=1,
        in_channels=256,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        transformer=dict(
            type="PstrTransformer",
            encoder=dict(
                type="DetrTransformerEncoder",
                num_layers=3,
                transformerlayers=dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=dict(
                        type="MultiScaleDeformableAttention",
                        num_levels=1,
                        embed_dims=256,
                    ),
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=("self_attn", "norm", "ffn", "norm"),
                ),
            ),
            decoder=dict(
                type="DeformableDetrTransformerDecoder",
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="MultiScaleDeformableAttention",
                            num_levels=1,
                            embed_dims=256,
                        ),
                    ],
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
            decoder1=dict(
                type="DeformableDetrTransformerDecoder",
                num_layers=1,
                return_intermediate=False,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="PartAttention",
                            num_levels=1,
                            num_points=4,
                            embed_dims=256,
                        ),
                        dict(
                            type="PartAttention",
                            num_levels=1,
                            num_points=4,
                            embed_dims=256,
                        ),
                    ],
                    feedforward_channels=256,
                    operation_order=("cross_attn", "cross_attn"),
                ),
            ),
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True, offset=-0.5
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
    ),
    test_cfg=dict(max_per_img=100),
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1500, 900),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=1),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

# change the path of the datasetz
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=2,
    test=dict(
        ann_file="outputs/test.json",
        img_prefix="data/cuhk_sysu_pedes/mmlab/test/",
        pipeline=test_pipeline,
    ),
)
