"""
check_gradient_vram_nan.py

Pre-training checklist: gradient flow, Stage A freeze integrity, NaN/Inf, and
real (forward+backward) VRAM -- everything the forward-only tests so far never
exercised. Reuses the same real Stage A checkpoint + real cross-frame data
setup already verified working in check_cross_frame.py.

Uses a deliberately simple PLACEHOLDER loss (sum of squares over all of G_1's
values) -- NOT the real training objective, which doesn't exist yet. This is
purely to exercise the full backward graph and confirm gradients genuinely
flow, not to test anything about loss design.

Run from GaussianFormer3D repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python check_gradient_vram_nan.py
"""
import os
import sys
import pickle

import torch
from mmengine import Config
from mmseg.models import build_segmentor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # GaussianFormer3D repo root -- script now lives one level deeper, in occ4dgs_scripts/
import model  # noqa: F401

from dataset import OPENOCC_DATASET, custom_collate_fn_temporal
from torch.utils.data.dataloader import DataLoader

OCC4DGS_ROOT = os.path.expanduser("~/Documents/min/Occ4DGS")
sys.path.insert(0, OCC4DGS_ROOT)
from src.models.stage_b_temporal.current_frame_encoder import CurrentFrameEncoder
from src.models.stage_b_temporal.gf3d_faithful_deform import GF3DFaithfulDeform
from src.models.stage_b_temporal.buffer import GaussianState
from src.models.stage_b_temporal.deform_heads import transform_anchor_for_projection

sys.path.insert(0, os.path.expanduser("~/Documents/min/GaussianFormer3D"))
from dataset.utils import get_lidar2global

CONFIG_PATH = "config/nuscenes_surroundocc_gs25600_full.py"
CHECKPOINT = "out/nuscenes_surroundocc_gs25600_full/epoch_3.pth"
TRAIN_PKL = "/media/user/1TSSD/min/gf3d_infos/nuscenes_infos_gf3d_train.pkl"
G0_CACHE_DIR = "/media/user/1TSSD/min/g0_cache"
TEST_SCENE_TOKEN = "0053e9c440a94c1b84bd9c4223efc4b0"
TMP_PKL = "/media/user/1TSSD/min/gf3d_infos/nuscenes_infos_gf3d_2frame_test.pkl"
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


def find_next_keyframe_index(scene_infos, after_index=0):
    for i in range(after_index + 1, len(scene_infos)):
        if scene_infos[i].get("is_key_frame"):
            return i
    raise ValueError(f"No keyframe found after index {after_index}")


def find_next_moving_keyframe_index(scene_infos, after_index, min_translation_m=0.5):
    idx = after_index
    while True:
        next_idx = find_next_keyframe_index(scene_infos, after_index=idx)
        t0 = scene_infos[after_index]["data"]["LIDAR_TOP"]["pose"]["translation"]
        t1 = scene_infos[next_idx]["data"]["LIDAR_TOP"]["pose"]["translation"]
        delta = sum((a - b) ** 2 for a, b in zip(t0, t1)) ** 0.5
        if delta >= min_translation_m:
            return next_idx, delta
        idx = next_idx


def get_real_pose(info):
    lidar_entry = info["data"]["LIDAR_TOP"]
    lidar2global = get_lidar2global(lidar_entry["calib"], lidar_entry["pose"])
    return torch.from_numpy(lidar2global).float()


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


def check_nan_inf(tensor, name):
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    status = "OK" if not (has_nan or has_inf) else "FAIL"
    print(f"    [{status}] {name}: nan={has_nan}, inf={has_inf}")
    return not (has_nan or has_inf)


def main():
    cfg = Config.fromfile(CONFIG_PATH)

    print("Loading Stage A checkpoint (frozen)...")
    segmentor = build_segmentor(cfg.model)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    segmentor.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    segmentor = segmentor.cuda().eval()
    for p in segmentor.parameters():
        p.requires_grad_(False)
    encoder = CurrentFrameEncoder(segmentor)

    print("\nLoading real cross-frame data (reusing the already-built 2-frame pkl)...")
    with open(TRAIN_PKL, "rb") as f:
        full_data = pickle.load(f)
    scene_infos = full_data["infos"][TEST_SCENE_TOKEN]
    next_kf_idx, delta = find_next_moving_keyframe_index(scene_infos, after_index=0)
    info_frame0 = full_data["infos"][TEST_SCENE_TOKEN][0]
    info_frame1 = full_data["infos"][TEST_SCENE_TOKEN][next_kf_idx]
    pose_prev = get_real_pose(info_frame0).cuda()
    pose_curr = get_real_pose(info_frame1).cuda()
    print(f"  Real translation: {delta:.3f} m")

    ds_config = dict(cfg.val_dataset_config)
    ds_config["imageset"] = TMP_PKL
    dataset = OPENOCC_DATASET.build(ds_config)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                         collate_fn=custom_collate_fn_temporal, num_workers=0)
    loader_iter = iter(loader)
    _ = next(loader_iter)
    data_frame1 = next(loader_iter)
    data_frame1 = move_to_cuda(data_frame1)
    imgs = data_frame1.pop("img")
    dpt = data_frame1.pop("dpt") if "dpt" in data_frame1 else None
    with torch.no_grad():
        ms_img_feats, dpt_dist, out_dpt_multiscale = encoder.encode(imgs, dpt, data_frame1)

    g0_data = torch.load(os.path.join(G0_CACHE_DIR, f"{TEST_SCENE_TOKEN}.pt"), map_location="cpu")
    g0_means = g0_data["means"].squeeze(0).cuda()
    g0_rotations = g0_data["rotations"].squeeze(0).cuda()
    g0_scales = g0_data["scales"].squeeze(0).cuda()
    g0_opacities = g0_data["opacities"].squeeze(0).cuda()
    g0_semantics = g0_data["semantics"].squeeze(0).cuda()
    with torch.no_grad():
        mu_proj, r_proj = transform_anchor_for_projection(g0_means, g0_rotations, pose_prev, pose_curr)
    g_prev = GaussianState(
        means=mu_proj, rotations=r_proj, scales=g0_scales,
        opacities=g0_opacities, semantics=g0_semantics,
    )

    print("\nBuilding GF3DFaithfulDeform (L=4)...")
    deform = GF3DFaithfulDeform(
        num_blocks=4, embed_dims=EMBED_DIMS, num_anchor=N_G,
        anchor_encoder_cfg=ANCHOR_ENCODER_CFG,
        deformable_model_cfg=DEFORMABLE_MODEL_CFG,
        norm_cfg=NORM_CFG, ffn_cfg=FFN_CFG,
    ).cuda()

    # ---- Forward-only VRAM (baseline, matching prior tests) ----
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = deform(g_prev, ms_img_feats, out_dpt_multiscale, data_frame1)
    peak_forward_only = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nForward-only peak VRAM: {peak_forward_only:.2f} GB")

    # ---- Forward + backward, WITH grad enabled ----
    print("\nRunning forward WITH grad enabled + backward (placeholder loss)...")
    torch.cuda.reset_peak_memory_stats()
    deform.zero_grad()
    g_1 = deform(g_prev, ms_img_feats, out_dpt_multiscale, data_frame1)

    # Deliberately simple placeholder loss -- NOT the real training objective
    # (that doesn't exist yet). Depends on every output value, to genuinely
    # exercise the full backward graph.
    loss = g_1.means.pow(2).sum() + g_1.rotations.pow(2).sum()
    loss.backward()
    peak_fwd_bwd = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Loss value: {loss.item():.4f}")
    print(f"Forward+backward peak VRAM: {peak_fwd_bwd:.2f} GB "
          f"(+{peak_fwd_bwd - peak_forward_only:.2f} GB vs forward-only)")

    # ---- Check 1: NaN/Inf in output ----
    print("\n=== NaN/Inf check: output ===")
    out_ok = True
    out_ok &= check_nan_inf(g_1.means, "g_1.means")
    out_ok &= check_nan_inf(g_1.rotations, "g_1.rotations")

    # ---- Check 2: gradients on Stage B's own parameters ----
    print("\n=== Gradient check: Stage B (should ALL have real, non-None, "
          "non-zero, finite gradients) ===")
    stage_b_ok = True
    n_checked = 0
    n_none = 0
    n_zero = 0
    n_nan_inf = 0
    for name, p in deform.named_parameters():
        n_checked += 1
        if p.grad is None:
            n_none += 1
            print(f"    [FAIL] {name}: grad is None")
            stage_b_ok = False
            continue
        if torch.all(p.grad == 0):
            n_zero += 1
            print(f"    [WARN] {name}: grad is all-zero")
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            n_nan_inf += 1
            print(f"    [FAIL] {name}: grad has NaN/Inf")
            stage_b_ok = False
    print(f"  Checked {n_checked} parameter tensors: {n_none} None, "
          f"{n_zero} all-zero, {n_nan_inf} NaN/Inf")
    if n_none == 0 and n_nan_inf == 0:
        print("  [OK] All Stage B parameters have real, finite gradients.")

    # Specifically confirm q0_table -- the parameter flagged earlier as
    # at-risk if this module were ever wired with a lazy-init pattern.
    q0_ok = deform.q0_table.grad is not None and not torch.all(deform.q0_table.grad == 0)
    print(f"  q0_table specifically: grad is None = {deform.q0_table.grad is None}, "
          f"all-zero = {torch.all(deform.q0_table.grad == 0).item() if deform.q0_table.grad is not None else 'N/A'} "
          f"[{'OK' if q0_ok else 'FAIL'}]")

    # ---- Check 3: Stage A freeze integrity -- confirm ZERO leakage ----
    print("\n=== Freeze integrity check: Stage A (should ALL have grad=None, "
          "confirming zero gradient leakage into the frozen checkpoint) ===")
    stage_a_leaked = 0
    n_stage_a_checked = 0
    for name, p in segmentor.named_parameters():
        n_stage_a_checked += 1
        if p.grad is not None:
            stage_a_leaked += 1
            print(f"    [FAIL] {name}: grad is NOT None -- LEAK into frozen Stage A weights")
    print(f"  Checked {n_stage_a_checked} Stage A parameter tensors: {stage_a_leaked} leaked")
    if stage_a_leaked == 0:
        print("  [OK] Stage A is genuinely frozen -- zero gradient leakage confirmed.")

    print("\n" + "=" * 60)
    all_ok = out_ok and stage_b_ok and q0_ok and (stage_a_leaked == 0)
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL -- see above for details'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
