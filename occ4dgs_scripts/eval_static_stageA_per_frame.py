"""
scripts/eval_static_stageA_per_frame.py

Table 1 (of 3): "Static 3DGS" reference -- runs Stage A's FULL pipeline fresh,
independently, on each val scene's OWN frame_next (not reusing G_0 at all),
producing a genuine per-frame single-frame reconstruction, then evaluates it
against that same frame's real GT via the same real MeanIoU machinery used
for the do-nothing baseline (Table 2) and Stage B (Table 3) -- so all three
tables are directly comparable, same 140 held-out scenes, same metric.

Adapts extract_g0_cache.py's PROVEN model-calling pattern (rep_only=True,
confirmed against train.py's real batch-unpacking logic) -- applied to
frame_next instead of frame 0, reusing StageBTrainingDataset's own sample
indexing to get the correct flat index per scene.

Run CAREFULLY alongside Stage B training (VRAM is tight, ~4.4GB free) --
one scene at a time, no_grad throughout, explicit cache-clearing between
scenes.

Run from GaussianFormer3D repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python occ4dgs_scripts/eval_static_stageA_per_frame.py
"""
import os
import pickle
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

OCC4DGS_ROOT = os.path.expanduser("~/Documents/min/Occ4DGS")
sys.path.insert(0, OCC4DGS_ROOT)
from src.datasets.stageb_dataset import StageBTrainingDataset

CONFIG_PATH = "config/nuscenes_surroundocc_gs25600_full.py"
CHECKPOINT = "out/nuscenes_surroundocc_gs25600_full/surroundocc_release.pth"  # official GaussianFormer3D authors' released checkpoint

STAGEB_DIR = "/media/user/1TSSD/min/stageb_training"
VAL_PAIRS_PKL = os.path.join(STAGEB_DIR, "nuscenes_infos_gf3d_stageb_pairs_val.pkl")
VAL_MANIFEST = os.path.join(STAGEB_DIR, "stageb_manifest_val.json")
G0_CACHE_DIR = "/media/user/1TSSD/min/g0_cache"


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

    print("Loading Stage A checkpoint...")
    segmentor = build_segmentor(cfg.model)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    segmentor.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    segmentor = segmentor.cuda().eval()
    for p in segmentor.parameters():
        p.requires_grad_(False)

    from misc.metric_util import MeanIoU

    def extract_per_class_ious(miou_metric):
        per_class = {}
        for i, label in enumerate(miou_metric.label_str):
            if miou_metric.total_seen[i] == 0:
                iou = 1.0
            else:
                iou = (miou_metric.total_correct[i] / (
                    miou_metric.total_seen[i] + miou_metric.total_positive[i]
                    - miou_metric.total_correct[i]
                )).item()
            per_class[label] = iou * 100
        return per_class

    miou_metric = MeanIoU(
        list(range(1, 17)),
        17,
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
        True, 17, filter_minmax=False,
    )
    miou_metric.reset()

    print("Building Stage B VAL dataset (reused only for its sample-index "
          "logic -- to find each scene's correct frame_next flat index)...")
    ds_config = dict(cfg.val_dataset_config)
    ds_config["imageset"] = VAL_PAIRS_PKL
    underlying = OPENOCC_DATASET.build(ds_config)
    with open(VAL_PAIRS_PKL, "rb") as f:
        raw_infos = pickle.load(f)["infos"]
    val_dataset = StageBTrainingDataset(underlying, VAL_MANIFEST, G0_CACHE_DIR, raw_infos)
    print(f"  {len(val_dataset)} val scenes")

    print(f"\nRunning FULL Stage A, fresh, on each scene's own frame_next "
          f"(genuine single-frame reconstruction, no G_0 reuse at all)...")

    with torch.no_grad():
        for i in range(len(val_dataset)):
            scene_token, flat_frame0_idx, flat_next_idx = val_dataset.samples[i]

            data_next = underlying[flat_next_idx]
            data_next = custom_collate_fn_temporal([data_next])
            data_next = move_dict_to_cuda(data_next)

            input_imgs = data_next.pop("img")
            input_points = data_next.pop("points") if "points" in data_next else None
            input_lidar_features = data_next.pop("lidar_feature_maps") if "lidar_feature_maps" in data_next else None
            input_dpt = data_next.pop("dpt") if "dpt" in data_next else None
            input_anchor_points = data_next.pop("anchor_points") if "anchor_points" in data_next else None

            representation = segmentor(
                imgs=input_imgs, points=input_points,
                lidar_feature_maps=input_lidar_features, dpt=input_dpt,
                anchor_points=input_anchor_points, metas=data_next, rep_only=True,
            )

            head_out = segmentor.head(representation=representation, metas=data_next)

            pred = head_out["pred_occ"][-1][0]
            pred_occ = pred.argmax(0)
            gt_occ = head_out["sampled_label"][0]
            if "occ3d_mask_camera" in head_out:
                miou_metric._after_step(pred_occ, gt_occ, head_out["occ3d_mask_camera"])
            else:
                miou_metric._after_step(pred_occ, gt_occ)

            torch.cuda.empty_cache()

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(val_dataset)} scenes processed...")

    miou, iou2 = miou_metric._after_epoch()
    per_class = extract_per_class_ious(miou_metric)
    import json
    result = {"method": "static_3dgs_oracle", "mIoU": float(miou), "iou2": float(iou2), "per_class_iou": per_class}
    out_path = "/media/user/1TSSD/min/stageb_training/eval_results/static_3dgs_oracle.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n{'='*60}")
    print(f"TABLE 1: Static 3DGS (fresh single-frame Stage A, {len(val_dataset)} held-out scenes)")
    print(f"  mIoU:  {float(miou):.4f}")
    print(f"  iou2:  {float(iou2):.4f}")
    print(f"  Saved -> {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
