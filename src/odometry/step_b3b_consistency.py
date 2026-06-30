"""
STEP B3b - Multi-view geometric consistency filtering
=====================================================

Attacks the root cause of blobby/noisy OBJECTS: monocular depth is not
multi-view consistent, so fusing it averages disagreeing per-view guesses into
blobs. This is the classical MVS geometric-consistency check applied to our
LiDAR-anchored learned depth - and it needs only torch/numpy (no gsplat / no
CUDA extension build), so it runs as-is on torch 2.12 / cu132.

For every image (the reference), each valid pixel is back-projected to a 3D point
(refined pose + K + B3 aligned depth) and reprojected into a set of covisible
neighbour frames. The pixel is CONFIRMED by a neighbour when the reprojected
depth agrees with that neighbour's depth there (within TAU, relative). Pixels
confirmed by few views are the inconsistent monocular guesses that smear objects
-> they get down-weighted/culled; the multi-view consensus surface survives.

Output is a per-pixel weight in [0,1] per frame; `Camera.confidence()` folds it
in, so the existing confidence-gated TSDF (step_d_mesh) automatically meshes only
the consistent geometry. Then re-run the TSDF at a finer voxel (the input is now
clean enough to afford it) for crisper objects.

Run (from repo root, after step_b3_align):
    python src/odometry/step_b3b_consistency.py

Output (under <session>/output/depth/):
    consistency/<stem>.npy   uint8 weight map (0..255) per image
    consistency.json         params + per-image paths (merged by common.py)
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import common


# --- Tunables ---------------------------------------------------------------
MIN_R, MAX_R = 0.2, 12.0     # valid depth range (m)
AXIS_COS_THR = 0.6           # neighbour must look within ~53 deg of reference
K_NEIGHBORS = 10             # neighbours checked per reference (spread in time)
TAU = 0.05                   # relative depth-agreement tolerance
MIN_SEEN = 3                 # only judge a pixel if >= this many neighbours saw it
K_MIN = 2                    # of those, need >= this many to AGREE to keep it
K_FULL = 5                   # full weight at this many agreements
STRIDE = 1                   # reference-pixel stride (1 = full res mask)


def select_neighbors(i, cos, k=K_NEIGHBORS, thr=AXIS_COS_THR):
    """Indices of frames looking the same way as i, spread in time for baseline."""
    c = cos[i].copy()
    c[i] = -1.0
    cand = np.where(c > thr)[0]
    if len(cand) <= k:
        return cand
    return np.unique(cand[np.linspace(0, len(cand) - 1, k).astype(int)])


def consistency_map(i, nbrs, depth, R, t, K, Kinv, dev):
    """uint8 (H,W) weight map: multi-view confirmation weight per ref pixel."""
    zr = depth[i]
    H, W = zr.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=dev),
                            torch.arange(W, device=dev), indexing="ij")
    sel = torch.isfinite(zr) & (zr > MIN_R) & (zr < MAX_R)
    if STRIDE > 1:
        s = torch.zeros_like(sel)
        s[::STRIDE, ::STRIDE] = True
        sel &= s
    u = xs[sel].float(); v = ys[sel].float(); z = zr[sel]
    if u.numel() == 0:
        return np.zeros((H, W), np.uint8)

    P = torch.stack([u, v, torch.ones_like(u)], 0)          # (3,N)
    Xc = (Kinv[i] @ P) * z[None]                            # (3,N) ref-cam
    Xw = R[i] @ Xc + t[i][:, None]                          # (3,N) world

    seen = torch.zeros(u.numel(), device=dev)    # neighbours that COULD check
    agree = torch.zeros(u.numel(), device=dev)   # ...and that confirmed it
    for n in nbrs:
        Xcn = R[n].T @ (Xw - t[n][:, None])                 # world -> nbr cam
        p = K[n] @ Xcn
        zp = p[2]
        un, vn = p[0] / zp, p[1] / zp
        Hn, Wn = depth[n].shape
        inb = (zp > 1e-3) & (un >= 0) & (un < Wn) & (vn >= 0) & (vn < Hn)
        gx = un / (Wn - 1) * 2 - 1
        gy = vn / (Hn - 1) * 2 - 1
        grid = torch.stack([gx, gy], -1).view(1, 1, -1, 2)
        dn = F.grid_sample(depth[n].view(1, 1, Hn, Wn), grid, mode="nearest",
                           align_corners=True, padding_mode="zeros").view(-1)
        valid_n = inb & (dn > 1e-3)
        seen += valid_n.float()
        agree += (valid_n & ((zp - dn).abs() < TAU * zp)).float()

    # Only JUDGE a pixel when enough neighbours actually saw it. Pixels too few
    # neighbours could check (e.g. grazing floor strips that barely overlap across
    # the spin) are unverifiable -> keep them (w=1), don't cull. Verified pixels
    # are kept by agreement and culled if confirmed-inconsistent.
    verified = seen >= MIN_SEEN
    w_ver = torch.where(agree >= K_MIN, torch.clamp(agree / K_FULL, 0, 1),
                        torch.zeros_like(agree))
    w = torch.where(verified, w_ver, torch.ones_like(agree))
    wmap = torch.zeros(H, W, device=dev)
    wmap[sel] = w
    return (wmap * 255).round().to(torch.uint8).cpu().numpy()


def run(data_root: str):
    cams = common.load_camera_manifest(data_root)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    poses = np.stack([c.pose for c in cams]).astype(np.float64)
    Ks = np.stack([c.K for c in cams]).astype(np.float64)
    axes = poses[:, :3, 2]
    axes /= np.linalg.norm(axes, axis=1, keepdims=True) + 1e-9
    cos = axes @ axes.T

    R = torch.from_numpy(poses[:, :3, :3]).float().to(dev)
    t = torch.from_numpy(poses[:, :3, 3]).float().to(dev)
    K = torch.from_numpy(Ks).float().to(dev)
    Kinv = torch.from_numpy(np.linalg.inv(Ks)).float().to(dev)
    # Cache aligned depths on GPU (nan -> 0 == invalid).
    depth = [torch.from_numpy(np.nan_to_num(c.read_depth_aligned(), nan=0.0))
             .float().to(dev) for c in cams]

    out_dir = os.path.join(os.path.abspath(data_root), "output", "depth",
                           "consistency")
    os.makedirs(out_dir, exist_ok=True)

    records, kept_frac, no_nbr = [], [], 0
    t0 = time.time()
    with torch.no_grad():
        for i, cam in enumerate(cams):
            nbrs = select_neighbors(i, cos)
            stem = os.path.splitext(os.path.basename(cam.source_image))[0]
            H, W = depth[i].shape
            if len(nbrs) < 3:
                wmap = np.full((H, W), 255, np.uint8)   # unverifiable -> trust
                no_nbr += 1
            else:
                wmap = consistency_map(i, nbrs, depth, R, t, K, Kinv, dev)
                valid = np.isfinite(cam.read_depth_aligned())
                if valid.sum():
                    kept_frac.append(float((wmap[valid] > 0).mean()))
            path = os.path.join(out_dir, f"{stem}.npy")
            np.save(path, wmap)
            records.append({"source_image": cam.source_image, "path": path})
            if i % 100 == 0:
                print(f"[{i + 1:04d}/{len(cams)}] {stem} nbrs={len(nbrs)}")

    manifest = {
        "session": os.path.basename(os.path.abspath(data_root)),
        "k_neighbors": K_NEIGHBORS, "tau": TAU, "k_min": K_MIN, "k_full": K_FULL,
        "count": len(records), "images": records,
    }
    with open(os.path.join(os.path.abspath(data_root), "output", "depth",
                           "consistency.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nB3b: consistency for {len(records)} frames in {time.time()-t0:.0f}s")
    if kept_frac:
        print(f"    kept (consistent) pixel fraction: median "
              f"{np.median(kept_frac)*100:.0f}% of valid depth")
    print(f"    {no_nbr} frames had <3 covisible neighbours (passed through)")
    print(f"Saved -> output/depth/consistency/ + consistency.json")


def main():
    run("./captures/session_05_20260624")


if __name__ == "__main__":
    main()
