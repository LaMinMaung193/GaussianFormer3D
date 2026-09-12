"""
scripts/render_qualitative.py

Qualitative visualization: for a handful of representative held-out scenes,
runs all three methods (do-nothing baseline, Stage B L=4, Static-3DGS oracle)
on the SAME real sampled 3D points, captures GT + all three predictions, and
saves the raw data (not the render itself -- rendering is a separate,
fast step) for building BEV comparison figures.

Run from GaussianFormer3D repo root, in the gf3d env:
    PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 python render_qualitative.py
"""
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
STAGE_A_CHECKPOINT = "out/nuscenes_surroundocc_gs25600_full/surroundocc_release.pth"  # released checkpoint
STAGEB_CHECKPOINT = "/media/user/1TSSD/min/stageb_training/checkpoints_L2_pretrained_release/epoch_24.pth"  # new official checkpoint

STAGEB_DIR = "/media/user/1TSSD/min/stageb_training"
VAL_PAIRS_PKL = os.path.join(STAGEB_DIR, "nuscenes_infos_gf3d_stageb_pairs_val.pkl")
VAL_MANIFEST = os.path.join(STAGEB_DIR, "stageb_manifest_val.json")
G0_CACHE_DIR = "/media/user/1TSSD/min/g0_cache_pretrained_release"  # regenerated with the released checkpoint
OUT_DIR = os.path.join(STAGEB_DIR, "qualitative_data_L2_pretrained_release")  # separate from the old L=2 data

N_G = 25600
EMBED_DIMS = 128
NUM_BLOCKS = 2  # regenerating against the official L=2 result -- second scene batch

SCENE_INDICES = [0, 20, 45, 70, 95, 120]  # first batch, matching L=4's first batch

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
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = Config.fromfile(CONFIG_PATH)

    print("Loading Stage A checkpoint (frozen)...")
    segmentor = build_segmentor(cfg.model)
    ckpt = torch.load(STAGE_A_CHECKPOINT, map_location="cpu")
    segmentor.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    segmentor = segmentor.cuda().eval()
    for p in segmentor.parameters():
        p.requires_grad_(False)
    encoder = CurrentFrameEncoder(segmentor)

    print(f"Loading Stage B (L={NUM_BLOCKS}) checkpoint...")
    deform = GF3DFaithfulDeform(
        num_blocks=NUM_BLOCKS, embed_dims=EMBED_DIMS, num_anchor=N_G,
        anchor_encoder_cfg=ANCHOR_ENCODER_CFG,
        deformable_model_cfg=DEFORMABLE_MODEL_CFG,
        norm_cfg=NORM_CFG, ffn_cfg=FFN_CFG,
    ).cuda()
    stageb_ckpt = torch.load(STAGEB_CHECKPOINT, map_location="cuda")
    deform.load_state_dict(stageb_ckpt["model_state_dict"])
    deform.eval()

    print("Building val dataset...")
    ds_config = dict(cfg.val_dataset_config)
    ds_config["imageset"] = VAL_PAIRS_PKL
    val_underlying = OPENOCC_DATASET.build(ds_config)
    with open(VAL_PAIRS_PKL, "rb") as f:
        val_raw_infos = pickle.load(f)["infos"]
    val_dataset = StageBTrainingDataset(val_underlying, VAL_MANIFEST, G0_CACHE_DIR, val_raw_infos)

    from model.encoder.gaussian_encoder.utils import GaussianPrediction

    with torch.no_grad():
        for scene_idx in SCENE_INDICES:
            sample = val_dataset[scene_idx]
            scene_token = sample["scene_token"]
            print(f"\nProcessing scene {scene_idx}: {scene_token}...")

            data_next = custom_collate_fn_temporal([sample["data_next"]])
            data_next = move_dict_to_cuda(data_next)
            pose_prev = sample["pose_prev"].cuda()
            pose_curr = sample["pose_curr"].cuda()

            imgs = data_next.pop("img")
            dpt = data_next.pop("dpt") if "dpt" in data_next else None

            g0_means = sample["g0_means"].cuda()
            g0_rotations = sample["g0_rotations"].cuda()
            g0_scales = sample["g0_scales"].cuda()
            g0_opacities = sample["g0_opacities"].cuda()
            g0_semantics = sample["g0_semantics"].cuda()

            mu_proj, r_proj = transform_anchor_for_projection(
                g0_means, g0_rotations, pose_prev, pose_curr
            )
            eps = 0.01
            pc_min = torch.tensor([-50.0 + eps, -50.0 + eps, -5.0 + eps], device=mu_proj.device)
            pc_max = torch.tensor([50.0 - eps, 50.0 - eps, 3.0 - eps], device=mu_proj.device)
            mu_proj_clamped = torch.clamp(mu_proj, min=pc_min, max=pc_max)

            gaussian_donothing = GaussianPrediction(
                means=mu_proj_clamped.unsqueeze(0), scales=g0_scales.unsqueeze(0),
                rotations=r_proj.unsqueeze(0), opacities=g0_opacities.unsqueeze(0),
                semantics=g0_semantics.unsqueeze(0),
            )
            head_out_donothing = segmentor.head(
                representation=[{"gaussian": gaussian_donothing}], metas=data_next
            )

            ms_img_feats, dpt_dist, out_dpt_multiscale = encoder.encode(imgs, dpt, data_next)
            g_prev = GaussianState(
                means=mu_proj_clamped, rotations=r_proj, scales=g0_scales,
                opacities=g0_opacities, semantics=g0_semantics,
            )
            g_1 = deform(g_prev, ms_img_feats, out_dpt_multiscale, data_next)
            gaussian_stageb = GaussianPrediction(
                means=g_1.means.unsqueeze(0), scales=g_1.scales.unsqueeze(0),
                rotations=g_1.rotations.unsqueeze(0), opacities=g_1.opacities.unsqueeze(0),
                semantics=g_1.semantics.unsqueeze(0),
            )
            head_out_stageb = segmentor.head(
                representation=[{"gaussian": gaussian_stageb}], metas=data_next
            )

            scene_token2, flat_frame0_idx, flat_next_idx = val_dataset.samples[scene_idx]
            data_next_raw = val_underlying[flat_next_idx]
            data_next_raw = custom_collate_fn_temporal([data_next_raw])
            data_next_raw = move_dict_to_cuda(data_next_raw)
            input_imgs = data_next_raw.pop("img")
            input_points = data_next_raw.pop("points") if "points" in data_next_raw else None
            input_lidar_features = data_next_raw.pop("lidar_feature_maps") if "lidar_feature_maps" in data_next_raw else None
            input_dpt = data_next_raw.pop("dpt") if "dpt" in data_next_raw else None
            input_anchor_points = data_next_raw.pop("anchor_points") if "anchor_points" in data_next_raw else None
            representation_oracle = segmentor(
                imgs=input_imgs, points=input_points,
                lidar_feature_maps=input_lidar_features, dpt=input_dpt,
                anchor_points=input_anchor_points, metas=data_next_raw, rep_only=True,
            )
            head_out_oracle = segmentor.head(representation=representation_oracle, metas=data_next_raw)
            torch.cuda.empty_cache()

            xyz_stageb = head_out_stageb["sampled_xyz"][0]
            xyz_donothing = head_out_donothing["sampled_xyz"][0]
            xyz_oracle = head_out_oracle["sampled_xyz"][0]
            assert torch.allclose(xyz_stageb, xyz_donothing, atol=1e-4), (
                "sampled_xyz differs between stageb and donothing calls -- "
                "GaussianHead's sampling is NOT deterministic, this "
                "visualization approach needs rethinking"
            )
            assert torch.allclose(xyz_stageb, xyz_oracle, atol=1e-4), (
                "sampled_xyz differs between stageb and oracle calls -- "
                "GaussianHead's sampling is NOT deterministic, this "
                "visualization approach needs rethinking"
            )

            sampled_xyz = xyz_stageb.cpu().numpy()
            sampled_label = head_out_stageb["sampled_label"][0].cpu().numpy()
            pred_donothing = head_out_donothing["pred_occ"][-1][0].argmax(0).cpu().numpy()
            pred_stageb = head_out_stageb["pred_occ"][-1][0].argmax(0).cpu().numpy()
            pred_oracle = head_out_oracle["pred_occ"][-1][0].argmax(0).cpu().numpy()

            import numpy as np
            out_path = os.path.join(OUT_DIR, f"scene_{scene_idx}_{scene_token[:12]}.npz")
            np.savez(
                out_path,
                scene_token=scene_token,
                sampled_xyz=sampled_xyz,
                sampled_label=sampled_label,
                pred_donothing=pred_donothing,
                pred_stageb=pred_stageb,
                pred_oracle=pred_oracle,
            )
            print(f"  Saved -> {out_path} ({len(sampled_xyz)} points)")

    print("\nAll scenes processed.")


if __name__ == "__main__":
    main()
