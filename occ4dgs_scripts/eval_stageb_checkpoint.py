"""
scripts/eval_stageb_checkpoint.py

Standalone evaluation of a specific, saved Stage B checkpoint (e.g. the
official L=4 epoch_33.pth) on the real held-out val set -- didn't exist
before now, since training only ever evaluated inline during the loop, never
as a separate, reusable step. Captures per-class IoU cleanly as structured
JSON (MeanIoU's own _after_epoch() computes per-class values internally but
never returns or stores them -- replicated here externally from its own
persistent total_seen/total_correct/total_positive attributes, confirmed
against real source, not guessed).

Run from GaussianFormer3D repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python occ4dgs_scripts/eval_stageb_checkpoint.py --checkpoint <path> --num_blocks <L> --out <json_path>
"""
import argparse
import json
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
from src.models.stage_b_temporal.current_frame_encoder import CurrentFrameEncoder
from src.models.stage_b_temporal.gf3d_faithful_deform import GF3DFaithfulDeform
from src.models.stage_b_temporal.buffer import GaussianState
from src.models.stage_b_temporal.deform_heads import transform_anchor_for_projection
from src.datasets.stageb_dataset import StageBTrainingDataset

CONFIG_PATH = "config/nuscenes_surroundocc_gs25600_full.py"
STAGE_A_CHECKPOINT = "out/nuscenes_surroundocc_gs25600_full/surroundocc_release.pth"  # released checkpoint, matching what train_stageb.py used

STAGEB_DIR = "/media/user/1TSSD/min/stageb_training"
VAL_PAIRS_PKL = os.path.join(STAGEB_DIR, "nuscenes_infos_gf3d_stageb_pairs_val.pkl")
VAL_MANIFEST = os.path.join(STAGEB_DIR, "stageb_manifest_val.json")
G0_CACHE_DIR = "/media/user/1TSSD/min/g0_cache_pretrained_release"  # regenerated with the released checkpoint, matching training
N_G = 25600
EMBED_DIMS = 128

ANCHOR_ENCODER_CFG = dict(
    type="SparseGaussian3DEncoder", embed_dims=128, include_opa=True,
    semantics=True, semantic_dim=17,
)
NORM_CFG = dict(type="LN", normalized_shape=128)
FFN_CFG = dict(
    type="AsymmetricFFN", in_channels=256, embed_dims=128,
    feedforward_channels=512, num_fcs=2, ffn_drop=0.1,
    act_cfg=dict(type="ReLU", inplace=True), pre_norm=dict(type="LN"),
)
DEFORMABLE_MODEL_CFG = dict(
    type="DeformableFeatureAggregation3D", embed_dims=128,
    kps_generator=dict(
        type="SparseGaussian3DKeyPointsGenerator3D", embed_dims=128,
        phi_activation="sigmoid", xyz_coordinate="cartesian", num_learnable_pts=2,
        fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0], [0, 0.45, 0],
                   [0, -0.45, 0], [0, 0, 0.45], [0, 0, -0.45]],
        pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
        scale_range=[0.01, 1.8],
    ),
    d_bound=[2.0, 58, 0.5], im2col_step=32, use_visibility=False,
    use_sampling_offsets=True, num_pts_per_keypoint=2, value_projection=False,
    num_cams=6, num_groups=4, num_levels=4, residual_mode="cat",
    use_camera_embed=True, use_deformable_func=True, attn_drop=0.15,
)


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
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to a saved Stage B checkpoint (e.g. epoch_33.pth)")
    parser.add_argument("--num_blocks", type=int, required=True, help="L used for this checkpoint")
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args()

    cfg = Config.fromfile(CONFIG_PATH)

    print("Loading Stage A checkpoint (frozen)...")
    segmentor = build_segmentor(cfg.model)
    ckpt = torch.load(STAGE_A_CHECKPOINT, map_location="cpu")
    segmentor.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    segmentor = segmentor.cuda().eval()
    for p in segmentor.parameters():
        p.requires_grad_(False)
    encoder = CurrentFrameEncoder(segmentor)

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

    print(f"Loading Stage B checkpoint (L={args.num_blocks}): {args.checkpoint}")
    deform = GF3DFaithfulDeform(
        num_blocks=args.num_blocks, embed_dims=EMBED_DIMS, num_anchor=N_G,
        anchor_encoder_cfg=ANCHOR_ENCODER_CFG,
        deformable_model_cfg=DEFORMABLE_MODEL_CFG,
        norm_cfg=NORM_CFG, ffn_cfg=FFN_CFG,
    ).cuda()
    stageb_ckpt = torch.load(args.checkpoint, map_location="cuda")
    deform.load_state_dict(stageb_ckpt["model_state_dict"])
    deform.eval()
    print(f"  Loaded from epoch {stageb_ckpt['epoch']}")

    print("Building held-out val dataset (140 scenes)...")
    ds_config = dict(cfg.val_dataset_config)
    ds_config["imageset"] = VAL_PAIRS_PKL
    val_underlying = OPENOCC_DATASET.build(ds_config)
    with open(VAL_PAIRS_PKL, "rb") as f:
        val_raw_infos = pickle.load(f)["infos"]
    val_dataset = StageBTrainingDataset(val_underlying, VAL_MANIFEST, G0_CACHE_DIR, val_raw_infos)
    print(f"  {len(val_dataset)} val scenes")

    print("\nRunning evaluation...")
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            sample = val_dataset[idx]
            data_next = custom_collate_fn_temporal([sample["data_next"]])
            data_next = move_dict_to_cuda(data_next)
            pose_prev = sample["pose_prev"].cuda()
            pose_curr = sample["pose_curr"].cuda()

            imgs = data_next.pop("img")
            dpt = data_next.pop("dpt") if "dpt" in data_next else None
            ms_img_feats, dpt_dist, out_dpt_multiscale = encoder.encode(imgs, dpt, data_next)

            g0_means = sample["g0_means"].cuda()
            g0_rotations = sample["g0_rotations"].cuda()
            g0_scales = sample["g0_scales"].cuda()
            g0_opacities = sample["g0_opacities"].cuda()
            g0_semantics = sample["g0_semantics"].cuda()
            mu_proj, r_proj = transform_anchor_for_projection(
                g0_means, g0_rotations, pose_prev, pose_curr
            )
            eps = 0.01
            pc_range_min = torch.tensor([-50.0 + eps, -50.0 + eps, -5.0 + eps], device=mu_proj.device)
            pc_range_max = torch.tensor([50.0 - eps, 50.0 - eps, 3.0 - eps], device=mu_proj.device)
            mu_proj = torch.clamp(mu_proj, min=pc_range_min, max=pc_range_max)

            g_prev = GaussianState(
                means=mu_proj, rotations=r_proj, scales=g0_scales,
                opacities=g0_opacities, semantics=g0_semantics,
            )
            g_1 = deform(g_prev, ms_img_feats, out_dpt_multiscale, data_next)

            from model.encoder.gaussian_encoder.utils import GaussianPrediction
            gaussian_pred = GaussianPrediction(
                means=g_1.means.unsqueeze(0), scales=g_1.scales.unsqueeze(0),
                rotations=g_1.rotations.unsqueeze(0), opacities=g_1.opacities.unsqueeze(0),
                semantics=g_1.semantics.unsqueeze(0),
            )
            representation = [{"gaussian": gaussian_pred}]
            head_out = segmentor.head(representation=representation, metas=data_next)

            pred = head_out["pred_occ"][-1][0]
            pred_occ = pred.argmax(0)
            gt_occ = head_out["sampled_label"][0]
            if "occ3d_mask_camera" in head_out:
                miou_metric._after_step(pred_occ, gt_occ, head_out["occ3d_mask_camera"])
            else:
                miou_metric._after_step(pred_occ, gt_occ)

            if (idx + 1) % 20 == 0:
                print(f"  {idx+1}/{len(val_dataset)} scenes processed...")

    miou, iou2 = miou_metric._after_epoch()
    per_class = extract_per_class_ious(miou_metric)

    result = {
        "checkpoint": args.checkpoint,
        "num_blocks": args.num_blocks,
        "epoch": stageb_ckpt["epoch"],
        "mIoU": float(miou),
        "iou2": float(iou2),
        "per_class_iou": per_class,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"mIoU: {float(miou):.4f}  iou2: {float(iou2):.4f}")
    print(f"Saved -> {args.out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
