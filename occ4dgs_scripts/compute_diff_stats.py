"""
scripts/compute_diff_stats.py

Aggregate improved/regressed statistic across the FULL 140-scene held-out val
set: for every scene, compares Dynamic 3DGS (Stage B, L=4) and the do-nothing
baseline against real GT, accumulating counts of where each voxel falls into
one of four categories (improved / regressed / both correct / both wrong) --
the same categories render_qualitative.py's diff map uses per-scene, summed
here across all 140 scenes for one clean, reportable aggregate ratio.

Does NOT save full point clouds (140 scenes x ~28MB would be ~4GB, not
needed) -- only scalar counts, per-scene and aggregate.

Run from GaussianFormer3D repo root, in the gf3d env:
    PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 python occ4dgs_scripts/compute_diff_stats.py
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
STAGE_A_CHECKPOINT = "out/nuscenes_surroundocc_gs25600_full/epoch_3.pth"
STAGEB_CHECKPOINT = "/media/user/1TSSD/min/stageb_training/checkpoints_L2/epoch_40.pth"

STAGEB_DIR = "/media/user/1TSSD/min/stageb_training"
VAL_PAIRS_PKL = os.path.join(STAGEB_DIR, "nuscenes_infos_gf3d_stageb_pairs_val.pkl")
VAL_MANIFEST = os.path.join(STAGEB_DIR, "stageb_manifest_val.json")
G0_CACHE_DIR = "/media/user/1TSSD/min/g0_cache"
OUT_JSON = os.path.join(STAGEB_DIR, "eval_results", "diff_stats_full_val_L2.json")

N_G = 25600
EMBED_DIMS = 128
NUM_BLOCKS = 2  # regenerating against the official L=2 result

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

EMPTY_LABEL = 17


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
    cfg = Config.fromfile(CONFIG_PATH)

    print("Loading Stage A checkpoint (frozen)...")
    segmentor = build_segmentor(cfg.model)
    ckpt = torch.load(STAGE_A_CHECKPOINT, map_location="cpu")
    segmentor.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    segmentor = segmentor.cuda().eval()
    for p in segmentor.parameters():
        p.requires_grad_(False)
    encoder = CurrentFrameEncoder(segmentor)

    print(f"Loading Stage B / Dynamic 3DGS (L={NUM_BLOCKS}) checkpoint...")
    deform = GF3DFaithfulDeform(
        num_blocks=NUM_BLOCKS, embed_dims=EMBED_DIMS, num_anchor=N_G,
        anchor_encoder_cfg=ANCHOR_ENCODER_CFG,
        deformable_model_cfg=DEFORMABLE_MODEL_CFG,
        norm_cfg=NORM_CFG, ffn_cfg=FFN_CFG,
    ).cuda()
    stageb_ckpt = torch.load(STAGEB_CHECKPOINT, map_location="cuda")
    deform.load_state_dict(stageb_ckpt["model_state_dict"])
    deform.eval()

    print("Building held-out val dataset (140 scenes)...")
    ds_config = dict(cfg.val_dataset_config)
    ds_config["imageset"] = VAL_PAIRS_PKL
    val_underlying = OPENOCC_DATASET.build(ds_config)
    with open(VAL_PAIRS_PKL, "rb") as f:
        val_raw_infos = pickle.load(f)["infos"]
    val_dataset = StageBTrainingDataset(val_underlying, VAL_MANIFEST, G0_CACHE_DIR, val_raw_infos)
    print(f"  {len(val_dataset)} val scenes")

    from model.encoder.gaussian_encoder.utils import GaussianPrediction

    total = {"improved": 0, "regressed": 0, "agree_correct": 0, "agree_wrong": 0}
    per_scene = []

    print("\nRunning both methods on every val scene...")
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

            g_prev = GaussianState(
                means=mu_proj_clamped, rotations=r_proj, scales=g0_scales,
                opacities=g0_opacities, semantics=g0_semantics,
            )
            g_1 = deform(g_prev, ms_img_feats, out_dpt_multiscale, data_next)
            gaussian_dynamic = GaussianPrediction(
                means=g_1.means.unsqueeze(0), scales=g_1.scales.unsqueeze(0),
                rotations=g_1.rotations.unsqueeze(0), opacities=g_1.opacities.unsqueeze(0),
                semantics=g_1.semantics.unsqueeze(0),
            )
            head_out_dynamic = segmentor.head(
                representation=[{"gaussian": gaussian_dynamic}], metas=data_next
            )

            gt = head_out_dynamic["sampled_label"][0]
            pred_donothing = head_out_donothing["pred_occ"][-1][0].argmax(0)
            pred_dynamic = head_out_dynamic["pred_occ"][-1][0].argmax(0)

            mask = gt != EMPTY_LABEL
            gt_f = gt[mask]
            donothing_f = pred_donothing[mask]
            dynamic_f = pred_dynamic[mask]

            donothing_correct = donothing_f == gt_f
            dynamic_correct = dynamic_f == gt_f

            n_improved = int((dynamic_correct & ~donothing_correct).sum().item())
            n_regressed = int((~dynamic_correct & donothing_correct).sum().item())
            n_agree_correct = int((dynamic_correct & donothing_correct).sum().item())
            n_agree_wrong = int((~dynamic_correct & ~donothing_correct).sum().item())

            total["improved"] += n_improved
            total["regressed"] += n_regressed
            total["agree_correct"] += n_agree_correct
            total["agree_wrong"] += n_agree_wrong

            per_scene.append({
                "scene_token": sample["scene_token"],
                "improved": n_improved, "regressed": n_regressed,
                "agree_correct": n_agree_correct, "agree_wrong": n_agree_wrong,
            })

            if (idx + 1) % 20 == 0:
                ratio_so_far = total["improved"] / max(1, total["regressed"])
                print(f"  {idx+1}/{len(val_dataset)} scenes | running ratio "
                      f"(improved:regressed) = {ratio_so_far:.2f}:1")

    ratio = total["improved"] / max(1, total["regressed"])
    total_voxels = sum(total.values())
    result = {
        "num_scenes": len(val_dataset),
        "total_counts": total,
        "total_voxels_compared": total_voxels,
        "improved_to_regressed_ratio": ratio,
        "pct_improved": 100 * total["improved"] / total_voxels,
        "pct_regressed": 100 * total["regressed"] / total_voxels,
        "pct_agree_correct": 100 * total["agree_correct"] / total_voxels,
        "pct_agree_wrong": 100 * total["agree_wrong"] / total_voxels,
        "per_scene": per_scene,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"AGGREGATE DIFF STATS ({len(val_dataset)} held-out val scenes)")
    print(f"{'='*60}")
    print(f"  Improved voxels:  {total['improved']:>10,} ({result['pct_improved']:.2f}%)")
    print(f"  Regressed voxels: {total['regressed']:>10,} ({result['pct_regressed']:.2f}%)")
    print(f"  Both correct:     {total['agree_correct']:>10,} ({result['pct_agree_correct']:.2f}%)")
    print(f"  Both wrong:       {total['agree_wrong']:>10,} ({result['pct_agree_wrong']:.2f}%)")
    print(f"\n  Improved:Regressed ratio = {ratio:.2f}:1")
    print(f"  Saved -> {OUT_JSON}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
