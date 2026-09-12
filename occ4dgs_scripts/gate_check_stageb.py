"""
gate_check_stageb.py

Small-scale gate check for Stage B, mirroring Stage A's own small-tier gate
check before its full 700-scene run: train on a small, FIXED subset of scenes
for many real epochs, and watch whether the SAME scenes' loss actually trends
down with repeated exposure. This is the first real test of "can this
mechanism learn anything at all" -- the earlier 20-iteration run only sampled
20 different, never-repeated scenes, which tells us nothing about learning.

Run from GaussianFormer3D repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python gate_check_stageb.py
"""
import os
import pickle
import sys
import time

import torch
from mmengine import Config
from mmseg.models import build_segmentor
import model  # noqa: F401
from loss import OPENOCC_LOSS
from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal

OCC4DGS_ROOT = os.path.expanduser("~/Documents/min/Occ4DGS")
sys.path.insert(0, OCC4DGS_ROOT)
from src.models.stage_b_temporal.current_frame_encoder import CurrentFrameEncoder
from src.models.stage_b_temporal.gf3d_faithful_deform import GF3DFaithfulDeform
from src.models.stage_b_temporal.buffer import GaussianState
from src.models.stage_b_temporal.deform_heads import transform_anchor_for_projection
from src.datasets.stageb_dataset import StageBTrainingDataset

CONFIG_PATH = "config/nuscenes_surroundocc_gs25600_full.py"
CHECKPOINT = "out/nuscenes_surroundocc_gs25600_full/epoch_3.pth"
PAIRS_PKL = "/media/user/1TSSD/min/gf3d_infos/nuscenes_infos_gf3d_stageb_pairs.pkl"
MANIFEST = "/media/user/1TSSD/min/gf3d_infos/stageb_manifest.json"
G0_CACHE_DIR = "/media/user/1TSSD/min/g0_cache"
N_G = 25600
EMBED_DIMS = 128
NUM_BLOCKS = 4
LR = 1e-4
WARMUP_EPOCHS = 5  # linear warmup: LR ramps from LR/WARMUP_EPOCHS up to LR
                    # over the first WARMUP_EPOCHS epochs, then holds constant.
                    # Added after the first gate check run showed real overall
                    # improvement (5.70 -> 4.47 over 30 epochs) but a genuine
                    # mid-training hump (epochs 8-14 rising back to ~5.6) --
                    # a classic signature of early LR instability from random
                    # init, not evidence the mechanism doesn't work. Testing
                    # whether warmup removes this hump, for a cleaner signal.

SUBSET_SIZE = 20
NUM_EPOCHS = 30
FIXED_SEED = 42

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
    return data


def run_one_sample(sample, encoder, deform, segmentor, loss_func, cfg, optimizer=None):
    data_next = custom_collate_fn_temporal([sample["data_next"]])
    data_next = move_dict_to_cuda(data_next)
    pose_prev = sample["pose_prev"].cuda()
    pose_curr = sample["pose_curr"].cuda()

    imgs = data_next.pop("img")
    dpt = data_next.pop("dpt") if "dpt" in data_next else None
    with torch.no_grad():
        ms_img_feats, dpt_dist, out_dpt_multiscale = encoder.encode(imgs, dpt, data_next)

    g0_means = sample["g0_means"].cuda()
    g0_rotations = sample["g0_rotations"].cuda()
    g0_scales = sample["g0_scales"].cuda()
    g0_opacities = sample["g0_opacities"].cuda()
    g0_semantics = sample["g0_semantics"].cuda()
    with torch.no_grad():
        mu_proj, r_proj = transform_anchor_for_projection(
            g0_means, g0_rotations, pose_prev, pose_curr
        )
    g_prev = GaussianState(
        means=mu_proj, rotations=r_proj, scales=g0_scales,
        opacities=g0_opacities, semantics=g0_semantics,
    )

    grad_ctx = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with grad_ctx:
        if optimizer is not None:
            optimizer.zero_grad()
        g_1 = deform(g_prev, ms_img_feats, out_dpt_multiscale, data_next)

        from model.encoder.gaussian_encoder.utils import GaussianPrediction
        gaussian_pred = GaussianPrediction(
            means=g_1.means.unsqueeze(0), scales=g_1.scales.unsqueeze(0),
            rotations=g_1.rotations.unsqueeze(0), opacities=g_1.opacities.unsqueeze(0),
            semantics=g_1.semantics.unsqueeze(0),
        )
        representation = [{"gaussian": gaussian_pred}]
        head_out = segmentor.head(representation=representation, metas=data_next)

        loss_input = {"metas": data_next}
        for k, v in cfg.loss_input_convertion.items():
            loss_input[k] = head_out[v]
        loss, loss_dict = loss_func(loss_input)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

    return loss.item()


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

    print("Building real OccupancyLoss...")
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()

    print("Building Stage B dataset...")
    ds_config = dict(cfg.val_dataset_config)
    ds_config["imageset"] = PAIRS_PKL
    underlying = OPENOCC_DATASET.build(ds_config)
    with open(PAIRS_PKL, "rb") as f:
        raw_infos = pickle.load(f)["infos"]
    full_dataset = StageBTrainingDataset(underlying, MANIFEST, G0_CACHE_DIR, raw_infos)

    g = torch.Generator().manual_seed(FIXED_SEED)
    subset_indices = torch.randperm(len(full_dataset), generator=g)[:SUBSET_SIZE].tolist()
    print(f"\nFixed subset: {SUBSET_SIZE} scenes (seed={FIXED_SEED})")
    subset_samples = [full_dataset[i] for i in subset_indices]
    for s in subset_samples:
        print(f"  {s['scene_token']}")

    print(f"\nBuilding GF3DFaithfulDeform (L={NUM_BLOCKS})...")
    deform = GF3DFaithfulDeform(
        num_blocks=NUM_BLOCKS, embed_dims=EMBED_DIMS, num_anchor=N_G,
        anchor_encoder_cfg=ANCHOR_ENCODER_CFG,
        deformable_model_cfg=DEFORMABLE_MODEL_CFG,
        norm_cfg=NORM_CFG, ffn_cfg=FFN_CFG,
    ).cuda()
    optimizer = torch.optim.AdamW(deform.parameters(), lr=LR)

    print(f"\n{'='*70}")
    print(f"Gate check: {SUBSET_SIZE} fixed scenes x {NUM_EPOCHS} epochs "
          f"({SUBSET_SIZE * NUM_EPOCHS} total iterations)")
    print(f"{'='*70}\n")

    epoch_avg_losses = []
    t_total_start = time.time()

    for epoch in range(NUM_EPOCHS):
        # Linear warmup: ramp LR from LR/WARMUP_EPOCHS up to LR over the
        # first WARMUP_EPOCHS epochs, then hold constant.
        if epoch < WARMUP_EPOCHS:
            current_lr = LR * (epoch + 1) / WARMUP_EPOCHS
        else:
            current_lr = LR
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        deform.train()
        order = torch.randperm(SUBSET_SIZE).tolist()
        epoch_losses = []
        for i in order:
            loss_val = run_one_sample(
                subset_samples[i], encoder, deform, segmentor, loss_func, cfg,
                optimizer=optimizer,
            )
            epoch_losses.append(loss_val)
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_avg_losses.append(avg_loss)
        elapsed = time.time() - t_total_start
        print(f"  epoch {epoch+1:3d}/{NUM_EPOCHS} | lr={current_lr:.2e} | avg_loss={avg_loss:8.4f} | "
              f"min={min(epoch_losses):.4f} max={max(epoch_losses):.4f} | "
              f"elapsed={elapsed/60:.1f}min")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Epoch 1 avg loss:  {epoch_avg_losses[0]:.4f}")
    print(f"  Epoch {NUM_EPOCHS} avg loss: {epoch_avg_losses[-1]:.4f}")
    print(f"  Best epoch avg loss: {min(epoch_avg_losses):.4f} (epoch {epoch_avg_losses.index(min(epoch_avg_losses))+1})")
    print(f"  Total time: {(time.time()-t_total_start)/60:.1f} minutes")
    print(f"\n  Full per-epoch trend:")
    for i, l in enumerate(epoch_avg_losses):
        bar = "#" * int(l * 3)
        print(f"    epoch {i+1:3d}: {l:7.4f} {bar}")


if __name__ == "__main__":
    main()
