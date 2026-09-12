_base_ = ['./nuscenes_surroundocc_gs25600.py']

# Small-tier gate check (Step 3): 50 train scenes, 10 val scenes -- everything
# else (N_g=25600, resolution, architecture) stays identical to the real config.
train_dataset_config = dict(
    imageset='/media/user/1TSSD/min/gf3d_infos/nuscenes_infos_gf3d_train_small.pkl',
)
val_dataset_config = dict(
    imageset='/media/user/1TSSD/min/gf3d_infos/nuscenes_infos_gf3d_val_small.pkl',
)

# Cheap signal check, not full convergence -- a few real epochs is enough to see
# whether held-out mIoU trends in a healthy direction.
max_epochs = 6
