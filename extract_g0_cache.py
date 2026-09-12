"""
scripts/extract_g0_cache.py

Stage B prep: runs the trained Stage A checkpoint once per scene (frame 0 only,
via nuscenes_infos_gf3d_frame0_all.pkl), caching each scene's G_0 (Gaussian
means/scales/rotations/opacities/semantics) to its own file -- so Stage B
experiments never need to re-run Stage A's forward pass.

Batch-unpacking logic (input_imgs/input_points/input_dpt/etc., rep_only=True)
mirrors train.py's real training loop EXACTLY (confirmed against train.py lines
200-243) -- not assumed or reimplemented independently.

Usage:
    PYTHONNOUSERSITE=1 python extract_g0_cache.py --checkpoint out/nuscenes_surroundocc_gs25600_full/latest.pth
"""
import argparse
import os
import sys
import time

import torch
from mmengine import Config
from mmseg.models import build_segmentor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model  # noqa: F401 -- triggers registry decorators
from dataset import OPENOCC_DATASET, custom_collate_fn_temporal
from torch.utils.data.dataloader import DataLoader

CONFIG_PATH = "config/nuscenes_surroundocc_gs25600_full.py"
FRAME0_PKL = "/media/user/1TSSD/min/gf3d_infos/nuscenes_infos_gf3d_frame0_all.pkl"
OUT_DIR = "/media/user/1TSSD/min/g0_cache_pretrained_release"  # separate from the old epoch_3-based cache -- regenerating with the authors' own released checkpoint


def build_frame0_dataset(cfg):
    # Reuse val_dataset_config's real pipeline (test_pipeline -- no augmentation,
    # correct for a canonical, deterministic Stage A reconstruction), pointed at
    # our frame-0-only info file instead of the real val split.
    ds_config = dict(cfg.val_dataset_config)
    ds_config["imageset"] = FRAME0_PKL
    return OPENOCC_DATASET.build(ds_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N scenes -- for a cheap test")
    args = parser.parse_args()

    cfg = Config.fromfile(CONFIG_PATH)

    print(f"Building model and loading checkpoint: {args.checkpoint}")
    model = build_segmentor(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model = model.cuda().eval()

    dataset = build_frame0_dataset(cfg)
    print(f"Frame-0 dataset: {len(dataset)} scenes")

    loader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False,
        collate_fn=custom_collate_fn_temporal, num_workers=2, pin_memory=True,
    )

    os.makedirs(OUT_DIR, exist_ok=True)

    n_done, n_skipped = 0, 0
    t_start = time.time()
    with torch.no_grad():
        for i, data in enumerate(loader):
            if args.limit is not None and i >= args.limit:
                break

            # Do NOT rely on scene_token surviving inside the collated batch --
            # confirmed via real test that it does not (fell through to the
            # scene_N fallback every time). Instead, use the dataset's own known,
            # deterministic order directly: with shuffle=False, batch_size=1,
            # item i always corresponds exactly to dataset.keyframes[i].
            scene_token = dataset.keyframes[i][0]
            out_path = os.path.join(OUT_DIR, f"{scene_token}.pt")
            if os.path.exists(out_path):
                n_skipped += 1
                continue

            # Move tensors to CUDA -- mirrors train.py's real logic exactly
            # (nested dict/list handling included, same as the real training loop).
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

            input_imgs = data.pop("img")
            input_points = data.pop("points") if "points" in data else None
            input_lidar_features = data.pop("lidar_feature_maps") if "lidar_feature_maps" in data else None
            input_dpt = data.pop("dpt") if "dpt" in data else None
            input_anchor_points = data.pop("anchor_points") if "anchor_points" in data else None

            representation = model(
                imgs=input_imgs, points=input_points,
                lidar_feature_maps=input_lidar_features, dpt=input_dpt,
                anchor_points=input_anchor_points, metas=data, rep_only=True,
            )
            gaussian = representation[-1]["gaussian"]

            torch.save({
                "means": gaussian.means.detach().cpu(),
                "scales": gaussian.scales.detach().cpu(),
                "rotations": gaussian.rotations.detach().cpu(),
                "opacities": gaussian.opacities.detach().cpu(),
                "semantics": gaussian.semantics.detach().cpu(),
                "scene_token": scene_token,
            }, out_path)

            n_done += 1
            if n_done % 20 == 0:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{len(dataset)}] {n_done} written, {n_skipped} "
                      f"skipped, elapsed {elapsed:.1f}s")

    elapsed = time.time() - t_start
    print(f"\nDone. Wrote {n_done}, skipped {n_skipped} already-existing, "
          f"to {OUT_DIR}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
