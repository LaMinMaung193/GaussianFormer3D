"""
scripts/check_release_full150.py

ONE-OFF SANITY CHECK ONLY -- not part of the final report's evaluation
pipeline (eval_static_stageA_per_frame.py, the real 140-scene held-out
script, is left completely untouched).

Checks whether the GaussianFormer3D authors' own released checkpoint
(surroundocc_release.pth) reproduces something close to their reported 27.1
mIoU, using the full, real 150-scene nuScenes-SurroundOcc val split -- one
frame (frame 0) per scene, same "fresh single-frame reconstruction"
methodology as our own 140-scene script, just widened to all real val
scenes and simplified to frame 0 (no motion-pair filtering needed, since
there's no deformation involved here at all).

Run from GaussianFormer3D repo root, in the gf3d env:
    PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 python occ4dgs_scripts/check_release_full150.py
"""
import os
import sys

os.environ["WANDB_MODE"] = "disabled"

import torch
import wandb
wandb.init(mode="disabled")

from mmengine import Config
from mmseg.models import build_segmentor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # GaussianFormer3D repo root -- script now lives one level deeper, in occ4dgs_scripts/
import model  # noqa: F401
from dataset import OPENOCC_DATASET, custom_collate_fn_temporal
from torch.utils.data.dataloader import DataLoader

CONFIG_PATH = "config/nuscenes_surroundocc_gs25600_full.py"
CHECKPOINT = "out/nuscenes_surroundocc_gs25600_full/surroundocc_release.pth"
VAL_PKL = "/media/user/1TSSD/min/gf3d_infos/nuscenes_infos_gf3d_val.pkl"


def move_dict_to_cuda(data):
    for k in list(data.keys()):
        if isinstance(data[k], torch.Tensor):
            data[k] = data[k].cuda()
        if isinstance(data[k], dict):
            for kk in data[k]:
                if isinstance(data[k][kk], torch.Tensor):
                    data[k][kk] = data[k][kk].cuda()
        if isinstance(data[k], list):
            for kk in range(len(data[k])):
                if isinstance(data[k][kk], torch.Tensor):
                    data[k][kk] = data[k][kk].cuda()
                if isinstance(data[k][kk], dict):
                    for kkk in data[k][kk]:
                        if isinstance(data[k][kk][kkk], torch.Tensor):
                            data[k][kk][kkk] = data[k][kk][kkk].cuda()
    return data


def main():
    cfg = Config.fromfile(CONFIG_PATH)

    print(f"Loading released checkpoint: {CHECKPOINT}")
    segmentor = build_segmentor(cfg.model)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    segmentor.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    segmentor = segmentor.cuda().eval()
    for p in segmentor.parameters():
        p.requires_grad_(False)

    from misc.metric_util import MeanIoU
    miou_metric = MeanIoU(
        list(range(1, 17)), 17,
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
        True, 17, filter_minmax=False,
    )
    miou_metric.reset()

    print(f"Building full real val dataset (150 scenes) from {VAL_PKL}...")
    ds_config = dict(cfg.val_dataset_config)
    ds_config["imageset"] = VAL_PKL
    val_dataset = OPENOCC_DATASET.build(ds_config)
    print(f"  {len(val_dataset)} real val samples (all keyframes, will use frame 0 of each scene only)")

    loader = DataLoader(
        dataset=val_dataset, batch_size=1, shuffle=False,
        collate_fn=custom_collate_fn_temporal, num_workers=2, pin_memory=True,
    )

    n_scenes_done = 0
    seen_scenes = set()

    print("\nRunning fresh Stage A reconstruction on frame 0 of each val scene...")
    with torch.no_grad():
        for i, data in enumerate(loader):
            scene_token = val_dataset.keyframes[i][0]
            if scene_token in seen_scenes:
                continue
            seen_scenes.add(scene_token)

            data = move_dict_to_cuda(data)
            input_imgs = data.pop("img")
            input_points = data.pop("points") if "points" in data else None
            input_lidar_features = data.pop("lidar_feature_maps") if "lidar_feature_maps" in data else None
            input_dpt = data.pop("dpt") if "dpt" in data else None
            input_anchor_points = data.pop("anchor_points") if "anchor_points" in data else None

            representation = segmentor(
                imgs=input_imgs, points=input_points,
                lidar_feature_maps=input_lidar_features, dpt=input_dpt,
                anchor_points=input_anchor_points, metas=data, rep_only=True,
            )
            head_out = segmentor.head(representation=representation, metas=data)

            pred = head_out["pred_occ"][-1][0]
            pred_occ = pred.argmax(0)
            gt_occ = head_out["sampled_label"][0]
            if "occ3d_mask_camera" in head_out:
                miou_metric._after_step(pred_occ, gt_occ, head_out["occ3d_mask_camera"])
            else:
                miou_metric._after_step(pred_occ, gt_occ)

            n_scenes_done += 1
            if n_scenes_done % 20 == 0:
                print(f"  {n_scenes_done} scenes processed...")

    miou, iou2 = miou_metric._after_epoch()
    print(f"\n{'='*60}")
    print(f"SANITY CHECK: released checkpoint, full real 150-scene val, frame 0 only")
    print(f"  Scenes evaluated: {n_scenes_done}")
    print(f"  mIoU:  {float(miou):.4f}  (paper reports 27.1 via dense, all-keyframe eval)")
    print(f"  iou2:  {float(iou2):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
