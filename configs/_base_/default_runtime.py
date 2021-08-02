checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
#custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
#load_from = None
load_from = "/apdcephfs/share_1290939/jiaxiaojun/OpenBrandData/outputB_detectors_ratio_eqlv2_r50/epoch_2.pth"
resume_from = None
workflow = [('train', 1)]

work_dir="/apdcephfs/share_1290939/jiaxiaojun/OpenBrandData/outputC_detectors_eqlv2_r50"
