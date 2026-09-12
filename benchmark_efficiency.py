"""
scripts/benchmark_efficiency.py

Measures real inference latency, peak VRAM, and parameter count for:
  1. Static 3DGS (full Stage A pipeline, fresh reconstruction) -- reuses the
     exact forward-pass pattern from eval_static_stageA_per_frame.py.
  2. Dynamic Occ4DGS (L=2 and L=4) -- reuses the exact forward-pass pattern
     from eval_stageb_checkpoint.py, separately timing the shared backbone
     encoding step and each L's own deformation step.

Averaged over N_SCENES real held-out val scenes (cycling through a
different real scene on every timed iteration, not repeating one scene),
with GPU warmup and torch.cuda.synchronize() around every timed block
(required for accurate GPU timing, since CUDA calls are asynchronous).

Run from GaussianFormer3D repo root, in the gf3d env:
    PYTHONNOUSERSITE=1 python benchmark_efficiency.py
"""
import json
import os
import pickle
import sys
import time

os.environ["WANDB_MODE"] = "disabled"
import torch
import wandb
wandb.init(mode="disabled")

from mmengine import Config
from mmseg.models import build_segmentor
import model  # noqa: F401
from dataset import OPENOCC_DATASET, custom_collate_fn_temporal

OCC4DGS_ROOT = os.path.expanduser("~/Documents/min/Occ4DGS")
sys.path.insert(0, OCC4DGS_ROOT)
from src.models.stage_b_temporal.current_frame_encoder import CurrentFrameEncoder
from src.models.stage_b_temporal.gf3d_faithful_deform import GF3DFaithfulDeform
from src.models.stage_b_temporal.buffer import GaussianState
from src.models.stage_b_temporal.deform_heads import transform_anchor_for_projection
from src.datasets.stageb_dataset import StageBTrainingDataset
from model.encoder.gaussian_encoder.utils import GaussianPrediction

CONFIG_PATH = "config/nuscenes_surroundocc_gs25600_full.py"
STAGE_A_CHECKPOINT = "out/nuscenes_surroundocc_gs25600_full/surroundocc_release.pth"

STAGEB_DIR = "/media/user/1TSSD/min/stageb_training"
VAL_PAIRS_PKL = os.path.join(STAGEB_DIR, "nuscenes_infos_gf3d_stageb_pairs_val.pkl")
VAL_MANIFEST = os.path.join(STAGEB_DIR, "stageb_manifest_val.json")
G0_CACHE_DIR = "/media/user/1TSSD/min/g0_cache_pretrained_release"
N_G = 25600
EMBED_DIMS = 128

L2_CHECKPOINT = os.path.join(STAGEB_DIR, "checkpoints_L2_pretrained_release/epoch_24.pth")
L4_CHECKPOINT = os.path.join(STAGEB_DIR, "checkpoints_L4_pretrained_release/epoch_33.pth")

N_SCENES = 20   # real scenes to average over
N_WARMUP = 3    # untimed warmup iterations before measuring

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


def deep_copy_except_tensors(obj):
    """Recursively copies dicts/lists fresh (so in-place mutation by a model's
    forward pass on one call doesn't persist into a later call on the same
    pre-loaded sample), while leaving tensor leaves shared (cheap, and not
    the likely target of any such mutation)."""
    if isinstance(obj, dict):
        return {k: deep_copy_except_tensors(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_copy_except_tensors(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(deep_copy_except_tensors(v) for v in obj)
    return obj.clone() if torch.is_tensor(obj) else obj  # clone tensors too -- protects against in-place mutation (e.g. reshape_) persisting across repeated calls


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def timed_run(fn, samples, n_warmup, n_measure):
    """Runs fn(sample) n_warmup times untimed (cycling through samples),
    then n_measure times timed, each on a DIFFERENT real sample (cycling
    if n_measure > len(samples)). Returns (mean_ms, peak_vram_mb, times)
    using proper CUDA synchronization."""
    for i in range(n_warmup):
        print(f"    [debug] warmup call {i+1}/{n_warmup}...")
        fn(samples[i % len(samples)])
        print(f"    [debug] warmup call {i+1} succeeded")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for i in range(n_measure):
        s = samples[i % len(samples)]
        torch.cuda.synchronize()
        t0 = time.time()
        fn(s)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)  # ms

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    mean_ms = sum(times) / len(times)
    return mean_ms, peak_vram_mb, times


def main():
    cfg = Config.fromfile(CONFIG_PATH)

    print("Loading Stage A checkpoint (shared, frozen, released checkpoint)...")
    segmentor = build_segmentor(cfg.model)
    ckpt = torch.load(STAGE_A_CHECKPOINT, map_location="cpu")
    segmentor.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    segmentor = segmentor.cuda().eval()
    for p in segmentor.parameters():
        p.requires_grad_(False)
    encoder = CurrentFrameEncoder(segmentor)

    stage_a_params = count_params(segmentor)
    print(f"  Stage A (full) params: {stage_a_params/1e6:.2f}M")

    print(f"\nBuilding val dataset ({N_SCENES} scenes for benchmarking)...")
    ds_config = dict(cfg.val_dataset_config)
    ds_config["imageset"] = VAL_PAIRS_PKL
    val_underlying = OPENOCC_DATASET.build(ds_config)
    with open(VAL_PAIRS_PKL, "rb") as f:
        val_raw_infos = pickle.load(f)["infos"]
    val_dataset = StageBTrainingDataset(val_underlying, VAL_MANIFEST, G0_CACHE_DIR, val_raw_infos)

    # Dynamic samples: via StageBTrainingDataset (matching eval_stageb_checkpoint.py exactly)
    samples = []
    # Static samples: via the RAW underlying dataset directly (matching
    # eval_static_stageA_per_frame.py exactly -- StageBTrainingDataset's own
    # data_next has a different structure not compatible with a raw Static
    # forward pass, confirmed by the camera-shape mismatch when reused directly)
    static_samples = []

    for idx in range(min(N_SCENES, len(val_dataset))):
        sample = val_dataset[idx]
        data_next = custom_collate_fn_temporal([sample["data_next"]])
        data_next = move_dict_to_cuda(data_next)
        pose_prev = sample["pose_prev"].cuda()
        pose_curr = sample["pose_curr"].cuda()
        g0_means = sample["g0_means"].cuda()
        g0_rotations = sample["g0_rotations"].cuda()
        g0_scales = sample["g0_scales"].cuda()
        g0_opacities = sample["g0_opacities"].cuda()
        g0_semantics = sample["g0_semantics"].cuda()
        samples.append(dict(
            data_next=data_next, pose_prev=pose_prev, pose_curr=pose_curr,
            g0_means=g0_means, g0_rotations=g0_rotations, g0_scales=g0_scales,
            g0_opacities=g0_opacities, g0_semantics=g0_semantics,
        ))

        _, _, flat_next_idx = val_dataset.samples[idx]
        static_data_next = val_underlying[flat_next_idx]
        static_data_next = custom_collate_fn_temporal([static_data_next])
        static_data_next = move_dict_to_cuda(static_data_next)
        static_samples.append(dict(data_next=static_data_next))

    print(f"  Loaded {len(samples)} real samples for benchmarking "
          f"({len(static_samples)} via the raw Static-matching path).")

    results = {}

    print("\nBenchmarking Static 3DGS (full Stage A pipeline)...")

    def static_forward(s):
        data = deep_copy_except_tensors(s["data_next"])  # recursive copy of containers -- protects against in-place mutation across repeated calls
        imgs = data.pop("img")
        points = data.pop("points") if "points" in data else None
        lidar_feats = data.pop("lidar_feature_maps") if "lidar_feature_maps" in data else None
        dpt = data.pop("dpt") if "dpt" in data else None
        anchor_pts = data.pop("anchor_points") if "anchor_points" in data else None
        with torch.no_grad():
            representation = segmentor(
                imgs=imgs, points=points, lidar_feature_maps=lidar_feats,
                dpt=dpt, anchor_points=anchor_pts, metas=data, rep_only=True,
            )
            _ = segmentor.head(representation=representation, metas=data)

    mean_ms, peak_mb, _ = timed_run(static_forward, static_samples, N_WARMUP, len(static_samples))
    results["static"] = {
        "latency_ms": mean_ms, "fps": 1000.0 / mean_ms,
        "peak_vram_mb": peak_mb, "params_m": stage_a_params / 1e6,
    }
    print(f"  Static: {mean_ms:.1f}ms/frame ({1000/mean_ms:.2f} FPS), "
          f"peak {peak_mb:.0f}MB, {stage_a_params/1e6:.2f}M params")

    print("\nBenchmarking shared backbone encoding (Dynamic, current-frame)...")

    def encode_forward(s):
        data = deep_copy_except_tensors(s["data_next"])  # recursive copy of containers -- protects against in-place mutation across repeated calls
        imgs = data.pop("img")
        dpt = data.pop("dpt") if "dpt" in data else None
        with torch.no_grad():
            _ = encoder.encode(imgs, dpt, data)

    encode_ms, encode_peak_mb, _ = timed_run(encode_forward, samples, N_WARMUP, len(samples))
    print(f"  Backbone encoding: {encode_ms:.1f}ms/frame, peak {encode_peak_mb:.0f}MB")

    for L, ckpt_path in [(2, L2_CHECKPOINT), (4, L4_CHECKPOINT)]:
        print(f"\nBenchmarking Dynamic Occ4DGS (L={L})...")
        deform = GF3DFaithfulDeform(
            num_blocks=L, embed_dims=EMBED_DIMS, num_anchor=N_G,
            anchor_encoder_cfg=ANCHOR_ENCODER_CFG,
            deformable_model_cfg=DEFORMABLE_MODEL_CFG,
            norm_cfg=NORM_CFG, ffn_cfg=FFN_CFG,
        ).cuda()
        stageb_ckpt = torch.load(ckpt_path, map_location="cuda")
        deform.load_state_dict(stageb_ckpt["model_state_dict"])
        deform.eval()
        deform_params = count_params(deform)

        def full_dynamic_forward(s, deform=deform):
            data = deep_copy_except_tensors(s["data_next"])  # recursive copy of containers -- protects against in-place mutation across repeated calls
            imgs = data.pop("img")
            dpt = data.pop("dpt") if "dpt" in data else None
            with torch.no_grad():
                ms_img_feats, dpt_dist, out_dpt_multiscale = encoder.encode(imgs, dpt, data)
                mu_proj, r_proj = transform_anchor_for_projection(
                    s["g0_means"], s["g0_rotations"], s["pose_prev"], s["pose_curr"]
                )
                eps = 0.01
                lo = torch.tensor([-50.0 + eps, -50.0 + eps, -5.0 + eps], device=mu_proj.device)
                hi = torch.tensor([50.0 - eps, 50.0 - eps, 3.0 - eps], device=mu_proj.device)
                mu_proj = torch.clamp(mu_proj, min=lo, max=hi)
                g_prev = GaussianState(
                    means=mu_proj, rotations=r_proj, scales=s["g0_scales"],
                    opacities=s["g0_opacities"], semantics=s["g0_semantics"],
                )
                g_1 = deform(g_prev, ms_img_feats, out_dpt_multiscale, data)
                gaussian_pred = GaussianPrediction(
                    means=g_1.means.unsqueeze(0), scales=g_1.scales.unsqueeze(0),
                    rotations=g_1.rotations.unsqueeze(0), opacities=g_1.opacities.unsqueeze(0),
                    semantics=g_1.semantics.unsqueeze(0),
                )
                representation = [{"gaussian": gaussian_pred}]
                _ = segmentor.head(representation=representation, metas=data)

        mean_ms, peak_mb, _ = timed_run(full_dynamic_forward, samples, N_WARMUP, len(samples))
        results[f"dynamic_L{L}"] = {
            "latency_ms": mean_ms, "fps": 1000.0 / mean_ms,
            "peak_vram_mb": peak_mb, "params_m": deform_params / 1e6,
            "encode_only_ms": encode_ms,
            "deform_only_ms_approx": mean_ms - encode_ms,
        }
        print(f"  L={L}: {mean_ms:.1f}ms/frame total ({1000/mean_ms:.2f} FPS), "
              f"peak {peak_mb:.0f}MB, {deform_params/1e6:.2f}M params "
              f"(encode ~{encode_ms:.1f}ms + deform+head ~{mean_ms-encode_ms:.1f}ms)")

        del deform
        torch.cuda.empty_cache()

    out_path = os.path.join(STAGEB_DIR, "eval_results", "efficiency_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("EFFICIENCY BENCHMARK SUMMARY")
    print(f"{'='*70}")
    for k, v in results.items():
        print(f"  {k}: {v['latency_ms']:.1f}ms ({v['fps']:.2f} FPS), "
              f"{v['peak_vram_mb']:.0f}MB peak, {v['params_m']:.2f}M params")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
