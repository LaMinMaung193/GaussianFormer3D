"""
check_real_feature_wiring.py

Gate 2 (real data): wires CurrentFrameEncoder (real Stage A backbone/neck/depth-head,
frozen) into GF3DFaithfulDeform, using ONE real scene's real frame-0 data and its real
cached G_0. Confirms the whole real-data path runs end to end -- not just synthetic
shapes.

Run from GaussianFormer3D repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python occ4dgs_scripts/check_real_feature_wiring.py
"""
import os
import sys

import torch
from mmengine import Config
from mmseg.models import build_segmentor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # GaussianFormer3D repo root -- script now lives one level deeper, in occ4dgs_scripts/
import model  # noqa: F401 -- GaussianFormer3D's own registry decorators

from dataset import OPENOCC_DATASET, custom_collate_fn_temporal
from torch.utils.data.dataloader import DataLoader

OCC4DGS_ROOT = os.path.expanduser("~/Documents/min/Occ4DGS")
sys.path.insert(0, OCC4DGS_ROOT)
from src.models.stage_b_temporal.current_frame_encoder import CurrentFrameEncoder
from src.models.stage_b_temporal.gf3d_faithful_deform import GF3DFaithfulDeform
from src.models.stage_b_temporal.buffer import GaussianState

CONFIG_PATH = "config/nuscenes_surroundocc_gs25600_full.py"
CHECKPOINT = "out/nuscenes_surroundocc_gs25600_full/epoch_3.pth"
FRAME0_PKL = "/media/user/1TSSD/min/gf3d_infos/nuscenes_infos_gf3d_frame0_all.pkl"
G0_CACHE_DIR = "/media/user/1TSSD/min/g0_cache"
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


def move_to_cuda(data):
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

    print("Loading Stage A checkpoint...")
    segmentor = build_segmentor(cfg.model)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = segmentor.load_state_dict(state_dict, strict=False)
    print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    segmentor = segmentor.cuda().eval()
    for p in segmentor.parameters():
        p.requires_grad_(False)

    encoder = CurrentFrameEncoder(segmentor)

    print("\nBuilding real frame-0 dataset...")
    ds_config = dict(cfg.val_dataset_config)
    ds_config["imageset"] = FRAME0_PKL
    dataset = OPENOCC_DATASET.build(ds_config)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                         collate_fn=custom_collate_fn_temporal, num_workers=0)

    data = next(iter(loader))
    scene_token = dataset.keyframes[0][0]
    print(f"  Using real scene: {scene_token}")

    data = move_to_cuda(data)
    imgs = data.pop("img")
    dpt = data.pop("dpt") if "dpt" in data else None
    print(f"  imgs: {tuple(imgs.shape)}, dpt: {tuple(dpt.shape) if dpt is not None else None}")
    print(f"  metas keys available: {sorted(data.keys())}")

    print("\nRunning CurrentFrameEncoder on real data...")
    with torch.no_grad():
        ms_img_feats, dpt_dist, out_dpt_multiscale = encoder.encode(imgs, dpt, data)
    print(f"  Got {len(ms_img_feats)} real feature levels:")
    for i, f in enumerate(ms_img_feats):
        print(f"    level {i}: {tuple(f.shape)}")

    print(f"\nLoading real cached G_0 for the SAME scene ({scene_token})...")
    g0_path = os.path.join(G0_CACHE_DIR, f"{scene_token}.pt")
    g0_data = torch.load(g0_path, map_location="cpu")
    g_prev = GaussianState(
        means=g0_data["means"].squeeze(0).cuda(),
        scales=g0_data["scales"].squeeze(0).cuda(),
        rotations=g0_data["rotations"].squeeze(0).cuda(),
        opacities=g0_data["opacities"].squeeze(0).cuda(),
        semantics=g0_data["semantics"].squeeze(0).cuda(),
    )
    print(f"  Loaded {g_prev.num_gaussians} Gaussians")

    print("\nBuilding GF3DFaithfulDeform (L=4)...")
    deform = GF3DFaithfulDeform(
        num_blocks=4, embed_dims=EMBED_DIMS, num_anchor=N_G,
        anchor_encoder_cfg=ANCHOR_ENCODER_CFG,
        deformable_model_cfg=DEFORMABLE_MODEL_CFG,
        norm_cfg=NORM_CFG, ffn_cfg=FFN_CFG,
    ).cuda()

    print("\nRunning GF3DFaithfulDeform with REAL camera/depth features...")
    with torch.no_grad():
        g_t = deform(g_prev, ms_img_feats, out_dpt_multiscale, data)

    print("\n=== Output check ===")
    print(f"  means: {tuple(g_t.means.shape)}, rotations: {tuple(g_t.rotations.shape)}")
    assert g_t.num_gaussians == N_G
    assert torch.allclose(g_t.scales, g_prev.scales), "scales should be frozen"
    assert torch.allclose(g_t.opacities, g_prev.opacities), "opacities should be frozen"
    assert torch.allclose(g_t.semantics, g_prev.semantics), "semantics should be frozen"
    moved = not torch.allclose(g_t.means, g_prev.means)
    rotated = not torch.allclose(g_t.rotations, g_prev.rotations)
    print(f"  Frozen properties confirmed unchanged. Position changed: {moved}, "
          f"Rotation changed: {rotated}")

    print("\nGate 2 (real data): PASS -- full real-data path runs end to end.")


if __name__ == "__main__":
    main()
