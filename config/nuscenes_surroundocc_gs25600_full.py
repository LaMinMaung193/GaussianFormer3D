_base_ = ['./nuscenes_surroundocc_gs25600.py']

# Full-scale Stage A training (Step 3 final decision): all 700 train scenes,
# 70 val scenes (not the full 150 -- keeps per-epoch validation overhead
# proportional, decided given the ~6-8hr/epoch real cost at full val size).
train_dataset_config = dict(
    imageset='/media/user/1TSSD/min/gf3d_infos/nuscenes_infos_gf3d_train_full.pkl',
)
val_dataset_config = dict(
    imageset='/media/user/1TSSD/min/gf3d_infos/nuscenes_infos_gf3d_val_full.pkl',
)

# Decided after small-tier gate check confirmed healthy mIoU trend
# (8.54 -> 9.53 -> 10.36 -> 10.51 -> 10.52) plateauing by epoch 3-4 at only
# 50 scenes -- 700 scenes gives far more diversity, 6 epochs is a deliberate,
# time-budget-driven choice, not GF3D's own default 24.
max_epochs = 12  # extended from 6 -- resuming from epoch_3.pth per professor's request
amp = False  # tried True to address a real OOM on resume -- reverted: the custom
              # ms_depth_score_sample_cuda_forward CUDA kernel does not support
              # half precision at all ("not implemented for 'Half'"), a hard
              # incompatibility, not a tuning knob
