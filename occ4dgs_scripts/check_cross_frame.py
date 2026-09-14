"""
check_cross_frame.py

Real cross-frame test: G_0 (frame 0) -> transform_anchor_for_projection (real
pose_prev/pose_curr) -> GF3DFaithfulDeform, using frame 1's REAL camera/depth
features -> G_1. The one thing the frame-0-only test never exercised: an actual,
non-identity ego-motion transform.

Run from GaussianFormer3D repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python occ4dgs_scripts/check_cross_frame.py
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
TEST_SCENE_TOKEN = "0053e9c440a94c1b84bd9c4223efc4b0"  # same scene as the previous test
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
    """scene_infos is a LIST (confirmed via real inspection, not a dict --
    NuScenesDataset's own [index] access works as list indexing either way,
    but membership/bounds checks need to know which). Most entries are
    non-keyframe sweeps (is_key_frame=False); real keyframes appear roughly
    every ~6 entries, not at every consecutive index. Search explicitly rather
    than assume index 1 is the next keyframe -- confirmed via real inspection
    that it is not."""
    for i in range(after_index + 1, len(scene_infos)):
        if scene_infos[i].get("is_key_frame"):
            return i
    raise ValueError(f"No keyframe found after index {after_index}")


def find_next_MOVING_keyframe_index(scene_infos, after_index, min_translation_m=0.5):
    """The immediately-next keyframe can genuinely be a near-zero-motion pair --
    confirmed via real inspection: our first attempt found frame 0 -> frame 6
    differed by ~0.0000016m (sub-micrometer, real sensor noise, not motion) --
    the vehicle was stationary (parked) for that segment, a real, valid data
    point, just not useful for testing an ego-motion transform. Search forward
    for a keyframe pair with GENUINE, non-trivial motion instead of assuming
    the next chronological keyframe involves real movement."""
    idx = after_index
    while True:
        next_idx = find_next_keyframe_index(scene_infos, after_index=idx)
        t0 = scene_infos[after_index]["data"]["LIDAR_TOP"]["pose"]["translation"]
        t1 = scene_infos[next_idx]["data"]["LIDAR_TOP"]["pose"]["translation"]
        delta = sum((a - b) ** 2 for a, b in zip(t0, t1)) ** 0.5
        if delta >= min_translation_m:
            return next_idx, delta
        idx = next_idx  # keep searching further into the scene


def make_2frame_pkl():
    """Builds a tiny pkl with frame 0 and the REAL next keyframe of our test
    scene, using the same verified (temp file + fsync + read-back) write
    pattern established after this drive's own history of silent write
    corruption."""
    with open(TRAIN_PKL, "rb") as f:
        data = pickle.load(f)
    scene_infos = data["infos"][TEST_SCENE_TOKEN]
    next_kf_idx, delta = find_next_MOVING_keyframe_index(scene_infos, after_index=0)
    print(f"  Found a genuinely-moving keyframe pair: index 0 -> index {next_kf_idx} "
          f"(real translation: {delta:.3f} m)")
    new_infos = {TEST_SCENE_TOKEN: scene_infos}
    new_metadata = [(TEST_SCENE_TOKEN, 0), (TEST_SCENE_TOKEN, next_kf_idx)]
    out_data = {"infos": new_infos, "metadata": new_metadata}

    tmp_path = TMP_PKL + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(out_data, f)
        f.flush()
        os.fsync(f.fileno())
    with open(tmp_path, "rb") as f:
        verify = pickle.load(f)
    assert len(verify["metadata"]) == 2
    os.replace(tmp_path, TMP_PKL)
    print(f"  Wrote and verified 2-frame test pkl -> {TMP_PKL}")


def get_real_pose(info):
    """Confirmed via real inspection: info['data']['LIDAR_TOP'] already
    contains fully-resolved 'calib' and 'pose' dicts directly (each with
    real 'rotation'/'translation' keys) -- no separate NuScenes devkit
    lookup needed at all, unlike what we assumed at first."""
    lidar_entry = info["data"]["LIDAR_TOP"]
    lidar2global = get_lidar2global(lidar_entry["calib"], lidar_entry["pose"])  # (4,4) numpy
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


def main():
    print("Building 2-frame test pkl...")
    make_2frame_pkl()

    with open(TRAIN_PKL, "rb") as f:
        full_data = pickle.load(f)
    scene_infos_check = full_data["infos"][TEST_SCENE_TOKEN]
    next_kf_idx, _ = find_next_MOVING_keyframe_index(scene_infos_check, after_index=0)
    info_frame0 = full_data["infos"][TEST_SCENE_TOKEN][0]
    info_frame1 = full_data["infos"][TEST_SCENE_TOKEN][next_kf_idx]
    pose_prev = get_real_pose(info_frame0).cuda()
    pose_curr = get_real_pose(info_frame1).cuda()
    translation_delta = (pose_curr[:3, 3] - pose_prev[:3, 3]).norm().item()
    print(f"  Real ego translation between frame 0 and frame 1: {translation_delta:.3f} m")
    assert translation_delta > 0.001, "Poses are suspiciously identical -- check scene/frame indices"

    cfg = Config.fromfile(CONFIG_PATH)

    print("\nLoading Stage A checkpoint...")
    segmentor = build_segmentor(cfg.model)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    segmentor.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    segmentor = segmentor.cuda().eval()
    for p in segmentor.parameters():
        p.requires_grad_(False)
    encoder = CurrentFrameEncoder(segmentor)

    print("\nLoading frame 1's real camera/depth data...")
    ds_config = dict(cfg.val_dataset_config)
    ds_config["imageset"] = TMP_PKL
    dataset = OPENOCC_DATASET.build(ds_config)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                         collate_fn=custom_collate_fn_temporal, num_workers=0)
    loader_iter = iter(loader)
    _ = next(loader_iter)  # frame 0 -- not needed, we already have its G_0 cached
    data_frame1 = next(loader_iter)
    print(f"  Loaded frame 1 (keyframe index: {dataset.keyframes[1]})")

    data_frame1 = move_to_cuda(data_frame1)
    imgs = data_frame1.pop("img")
    dpt = data_frame1.pop("dpt") if "dpt" in data_frame1 else None
    with torch.no_grad():
        ms_img_feats, dpt_dist, out_dpt_multiscale = encoder.encode(imgs, dpt, data_frame1)
    print(f"  Real frame-1 features: {len(ms_img_feats)} levels")

    print(f"\nLoading real cached G_0 for scene {TEST_SCENE_TOKEN}...")
    g0_data = torch.load(os.path.join(G0_CACHE_DIR, f"{TEST_SCENE_TOKEN}.pt"), map_location="cpu")
    g0_means = g0_data["means"].squeeze(0).cuda()
    g0_rotations = g0_data["rotations"].squeeze(0).cuda()
    g0_scales = g0_data["scales"].squeeze(0).cuda()
    g0_opacities = g0_data["opacities"].squeeze(0).cuda()
    g0_semantics = g0_data["semantics"].squeeze(0).cuda()

    print("\nApplying transform_anchor_for_projection (Section 3.3) -- the piece "
          "never exercised by the frame-0-only test...")
    mu_proj, r_proj = transform_anchor_for_projection(g0_means, g0_rotations, pose_prev, pose_curr)
    position_shift = (mu_proj - g0_means).norm(dim=-1).mean().item()
    print(f"  Mean per-Gaussian position shift from the transform: {position_shift:.3f} m "
          f"(should be a real, non-trivial number, roughly on the order of the "
          f"{translation_delta:.3f}m ego translation)")

    g_prev_for_projection = GaussianState(
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

    print("\nRunning GF3DFaithfulDeform: G_0 (frame-transformed) + real frame-1 "
          "features -> G_1...")
    with torch.no_grad():
        g_1 = deform(g_prev_for_projection, ms_img_feats, out_dpt_multiscale, data_frame1)

    print("\n=== Output check ===")
    print(f"  means: {tuple(g_1.means.shape)}, rotations: {tuple(g_1.rotations.shape)}")
    assert g_1.num_gaussians == N_G
    assert torch.allclose(g_1.scales, g0_scales), "scales should be frozen"
    assert torch.allclose(g_1.opacities, g0_opacities), "opacities should be frozen"
    assert torch.allclose(g_1.semantics, g0_semantics), "semantics should be frozen"
    print("  Frozen properties confirmed unchanged across the full cross-frame path.")

    print("\nGate 2b (real cross-frame): PASS -- full pipeline, including the real "
          "frame-transform step, runs end to end on genuine, different-pose frames.")


if __name__ == "__main__":
    main()
