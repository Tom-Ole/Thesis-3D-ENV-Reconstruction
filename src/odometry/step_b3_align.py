"""
STEP B3 - LiDAR-anchored metric alignment  (image branch, after B2)  <- FUSION GLUE
===================================================================================

Turns B2's monocular depth (metric-ish, but with a global scale error and slow
spatial warp) into depth that AGREES with the LiDAR everywhere it can be checked.
This is the glue that makes the dense image geometry metric and drift-consistent
with the LiDAR scaffold, so B4 can back-project it into one fused cloud.

Per image:
  1. ANCHORS.  Project the time-nearest LiDAR scan (B1) into the image, in the
     REFINED odom frame (scan placed by its Step-1 pose, camera at its B1 refined
     pose - the same frame B4/fusion live in).  -> sparse metric depth samples
     (uv, z_lidar).  Anchors on steep predicted-depth gradients are dropped
     (edges sample unreliably).
  2. GLOBAL AFFINE in INVERSE-DEPTH (disparity).  Robustly fit
         1/z_lidar ~ a*(1/z_pred) + b        (IRLS trimming)
     Disparity space linearises the monocular error far better than metres
     (probe: R^2 0.65->0.79, 0.76->0.92, ...), and a robust fit is mandatory
     because the anchor residuals are heavy-tailed.
  3. SPLINE (smoothed residual grid).  Fit a low-frequency correction of the
     leftover disparity residual over the image (coarse robust-binned grid,
     hole-filled + Gaussian-smoothed).  GATED by held-out error so it can only
     help, never overfit the sparse anchors.
  4. QUALITY + CONFIDENCE.  Honest held-out residual + inlier fraction -> a
     per-image ``fit_quality`` scalar (frames vary from R^2 0.27 to 0.99, so the
     fusion MUST down-weight the bad ones).  B4 reads a per-pixel confidence
     (quality x edge-weight x range-validity) off this.

Output is PARAMS-ONLY (no second copy of the depth): the affine is exact and the
spline is a tiny coarse grid.  ``common.Camera.read_depth_aligned()`` applies
them on the fly; ``.confidence()`` derives the per-pixel weight.

Run (from repo root, after step1 -> step_b1 -> step_b2):
    python src/odometry/step_b3_align.py

Output (under <session>/output/depth/):
    align_b3.json              per-image a,b,quality,held-out residual, grid ref
    aligned/<stem>_corr.npy    coarse disparity-residual grid (only if spline used)
    debug/<stem>_align.jpg     aligned depth + per-anchor error check
"""

import os
import json
import time

import numpy as np
import cv2
from scipy import ndimage

import common


# --- Tunables ---------------------------------------------------------------
MIN_RANGE = 0.2          # valid LiDAR/depth range for anchoring (m)
MAX_RANGE = 12.0
GRAD_QUANTILE = 0.85     # drop anchors above this predicted-grad percentile
AFFINE_ITERS = 3         # IRLS trimming passes
AFFINE_K = 2.5           # trim threshold in robust sigma
GW, GH = 16, 12          # residual-grid resolution (cols, rows)
GRID_MIN_PER_CELL = 3    # anchors needed to seed a grid cell
GRID_SMOOTH_SIGMA = 1.0  # Gaussian smoothing of the grid (cells)
SPLINE_GATE_FACTOR = 0.97   # keep spline only if held-out med < this * affine
SPLINE_GATE_ABS_M = 0.001   # ...and improves by at least 1 mm
QUALITY_RES_SCALE = 0.10    # held-out residual (m) that halves-ish the quality


# --- Small numerics ---------------------------------------------------------

def _bilinear(grid: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinearly sample a 2D map at float pixel coords (clamped)."""
    H, W = grid.shape
    u0 = np.clip(np.floor(u).astype(int), 0, W - 2)
    v0 = np.clip(np.floor(v).astype(int), 0, H - 2)
    fu, fv = u - u0, v - v0
    g = grid
    return (g[v0, u0] * (1 - fu) * (1 - fv) + g[v0, u0 + 1] * fu * (1 - fv)
            + g[v0 + 1, u0] * (1 - fu) * fv + g[v0 + 1, u0 + 1] * fu * fv)


def _robust_affine(dp: np.ndarray, dl: np.ndarray):
    """IRLS-trimmed fit dl ~ a*dp + b. Returns (a, b, inlier_mask)."""
    A = np.vstack([dp, np.ones_like(dp)]).T
    w = np.ones_like(dp)
    coef = np.array([1.0, 0.0])
    s = 1.0
    for _ in range(AFFINE_ITERS):
        Aw = A * w[:, None]
        coef, *_ = np.linalg.lstsq(Aw, dl * w, rcond=None)
        res = dl - A @ coef
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        w = (np.abs(res) < AFFINE_K * s).astype(float)
    inl = np.abs(dl - A @ coef) < AFFINE_K * s
    return float(coef[0]), float(coef[1]), inl


def _fill_nan(grid: np.ndarray) -> np.ndarray:
    """Replace NaN cells with their nearest finite neighbour (EDT)."""
    if np.all(np.isnan(grid)):
        return np.zeros_like(grid)
    nan = np.isnan(grid)
    idx = ndimage.distance_transform_edt(
        nan, return_distances=False, return_indices=True)
    return grid[tuple(idx)]


def _fit_residual_grid(uv, resid, W, H):
    """Coarse robust-binned, hole-filled, smoothed disparity-residual grid."""
    grid = np.full((GH, GW), np.nan)
    gx = np.clip((uv[:, 0] / W * GW).astype(int), 0, GW - 1)
    gy = np.clip((uv[:, 1] / H * GH).astype(int), 0, GH - 1)
    for j in range(GH):
        rmask = gy == j
        if not rmask.any():
            continue
        gxx, rr = gx[rmask], resid[rmask]
        for i in range(GW):
            cell = gxx == i
            if cell.sum() >= GRID_MIN_PER_CELL:
                grid[j, i] = np.median(rr[cell])
    grid = _fill_nan(grid)
    return ndimage.gaussian_filter(grid, GRID_SMOOTH_SIGMA,
                                   mode="nearest").astype(np.float32)


def _sample_grid(grid, uv, W, H):
    """Upsample the coarse grid to full res and sample at uv."""
    full = cv2.resize(grid, (W, H), interpolation=cv2.INTER_LINEAR)
    return _bilinear(full, uv[:, 0], uv[:, 1])


# --- Anchors ----------------------------------------------------------------

def gather_anchors(cam, depth_pred, refined_pts):
    """(uv, z_pred, z_lidar) sparse anchors for one image, edge-filtered."""
    uv, zl, _ = cam.project(refined_pts)
    if len(uv) < 50:
        return None
    gy, gx = np.gradient(np.log(np.clip(depth_pred, MIN_RANGE, None)))
    gmag = np.hypot(gx, gy)
    zp = _bilinear(depth_pred, uv[:, 0], uv[:, 1])
    gs = _bilinear(gmag, uv[:, 0], uv[:, 1])
    good = (np.isfinite(zp) & (zp > MIN_RANGE)
            & (zl > MIN_RANGE) & (zl < MAX_RANGE))
    if good.sum() < 50:
        return None
    thr = np.quantile(gs[good], GRAD_QUANTILE)
    good &= gs <= thr
    return uv[good], zp[good], zl[good]


# --- Per-image fit ----------------------------------------------------------

def _heldout_median(uv, zp, zl, W, H, use_grid, seed=0):
    """Honest held-out median |z_err| (m): affine, and affine+grid."""
    n = len(zp)
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    tr, te = idx[:n // 2], idx[n // 2:]
    if len(tr) < 20 or len(te) < 20:
        return None, None
    dp, dl = 1.0 / zp, 1.0 / zl
    a, b, inl = _robust_affine(dp[tr], dl[tr])
    Dte = a * dp[te] + b
    med_aff = float(np.median(np.abs(1.0 / np.clip(Dte, 1e-6, None) - zl[te])))
    if not use_grid:
        return med_aff, med_aff
    resid = dl[tr] - (a * dp[tr] + b)
    grid = _fit_residual_grid(uv[tr][inl], resid[inl], W, H)
    gte = _sample_grid(grid, uv[te], W, H)
    med_grid = float(np.median(
        np.abs(1.0 / np.clip(Dte + gte, 1e-6, None) - zl[te])))
    return med_aff, med_grid


def fit_image(uv, zp, zl, W, H, use_spline):
    """Final fit on all anchors. Returns a dict of params + a grid (or None)."""
    dp, dl = 1.0 / zp, 1.0 / zl
    a, b, inl = _robust_affine(dp, dl)
    inlier_frac = float(inl.mean())

    med_aff, med_grid = _heldout_median(uv, zp, zl, W, H, use_grid=use_spline)
    spline = (use_spline and med_grid is not None
              and med_grid < med_aff * SPLINE_GATE_FACTOR
              and (med_aff - med_grid) > SPLINE_GATE_ABS_M)
    med_final = med_grid if spline else med_aff

    grid = None
    if spline:
        resid = dl - (a * dp + b)
        grid = _fit_residual_grid(uv[inl], resid[inl], W, H)

    quality = float(inlier_frac * np.exp(-(med_final or 0.0) / QUALITY_RES_SCALE))
    return {
        "a": a, "b": b,
        "inlier_frac": inlier_frac,
        "n_anchors": int(len(zp)),
        "heldout_med_affine_m": med_aff,
        "heldout_med_final_m": med_final,
        "use_spline": bool(spline),
        "fit_quality": quality,
    }, grid


# --- Debug ------------------------------------------------------------------

def _draw_anchor_error(cam, params, grid, uv, zl):
    """Aligned-depth preview + per-anchor error (green=match, red=off)."""
    import step_b2_depth as b2
    z = cam.read_depth()
    dp = 1.0 / np.clip(z, 1e-6, None)
    D = params["a"] * dp + params["b"]
    if grid is not None:
        D = D + cv2.resize(grid, (cam.width, cam.height),
                           interpolation=cv2.INTER_LINEAR)
    z_corr = np.where(D > 1e-6, 1.0 / D, np.nan).astype(np.float32)

    vis = b2._colorize(z_corr)
    zc = _bilinear(np.nan_to_num(z_corr, nan=0.0), uv[:, 0], uv[:, 1])
    err = np.abs(zc - zl)
    for (u, v), e in zip(uv.astype(int), err):
        t = min(e, 0.30) / 0.30
        cv2.circle(vis, (u, v), 2, (0, int(255 * (1 - t)), int(255 * t)), -1)
    cv2.putText(vis, f"a{params['a']:.2f} b{params['b']:.2f} "
                f"q{params['fit_quality']:.2f} "
                f"err{params['heldout_med_final_m']*1e3:.0f}mm "
                f"{'spline' if params['use_spline'] else 'affine'}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.hstack([cam.read_image(), vis])


# --- Driver -----------------------------------------------------------------

def run(data_root: str, use_spline: bool = True, overlay: bool = True,
        overlay_every: int = 50, limit: int | None = None):
    cams = common.load_camera_manifest(data_root)
    if limit:
        cams = cams[:limit]
    ds = common.SessionDataset(data_root)
    traj = np.load(os.path.join(ds.output_dir, "trajectory_poses_world.npy"))

    out_dir = os.path.join(os.path.abspath(data_root), "output", "depth")
    grid_dir = os.path.join(out_dir, "aligned")
    debug_dir = os.path.join(out_dir, "debug")
    os.makedirs(grid_dir, exist_ok=True)
    if overlay:
        os.makedirs(debug_dir, exist_ok=True)

    refined_cache = {}

    def refined_points(li):
        if li not in refined_cache:
            pts = np.asarray(ds.frames[li].read_cloud_sensor()
                             .transform(traj[li]).points)
            refined_cache[li] = pts
        return refined_cache[li]

    records, skipped, n_spline = [], 0, 0
    med_aff_all, med_fin_all, quals = [], [], []
    t0 = time.time()
    for n, cam in enumerate(cams):
        if cam.depth_path is None or cam.lidar_index is None:
            skipped += 1
            continue
        depth_pred = cam.read_depth()
        anchors = gather_anchors(cam, depth_pred, refined_points(cam.lidar_index))
        if anchors is None:
            skipped += 1
            continue
        uv, zp, zl = anchors
        params, grid = fit_image(uv, zp, zl, cam.width, cam.height, use_spline)

        stem = os.path.splitext(os.path.basename(cam.source_image))[0]
        grid_path = None
        if grid is not None:
            grid_path = os.path.join(grid_dir, f"{stem}_corr.npy")
            np.save(grid_path, grid.astype(np.float16))
            n_spline += 1

        records.append({
            "source_image": cam.source_image,
            "source": cam.source,
            "camera": cam.camera,
            "align_grid_path": grid_path,
            **params,
        })
        med_aff_all.append(params["heldout_med_affine_m"])
        med_fin_all.append(params["heldout_med_final_m"])
        quals.append(params["fit_quality"])

        if overlay and n % overlay_every == 0:
            cv2.imwrite(os.path.join(debug_dir, f"{stem}_align.jpg"),
                        _draw_anchor_error(cam, params, grid, uv, zl))
        if n % 100 == 0:
            print(f"[{n + 1:04d}/{len(cams)}] {stem} q={params['fit_quality']:.2f} "
                  f"err {params['heldout_med_final_m']*1e3:.0f}mm "
                  f"{'spline' if params['use_spline'] else 'affine'}")

    manifest = {
        "session": os.path.basename(os.path.abspath(data_root)),
        "depth_aligned_kind": "metric_aligned_m",
        "fit_space": "inverse_depth",
        "grid_shape": [GH, GW],
        "count": len(records),
        "images": records,
    }
    out_path = os.path.join(out_dir, "align_b3.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    dt = time.time() - t0
    quals = np.array(quals)
    print(f"\nB3: aligned {len(records)} images in {dt:.1f}s ({skipped} skipped)")
    if records:
        print(f"    held-out median err: affine {np.median(med_aff_all)*1e3:.0f}mm "
              f"-> final {np.median(med_fin_all)*1e3:.0f}mm")
        print(f"    spline kept on {n_spline}/{len(records)} images")
        print(f"    fit_quality: median {np.median(quals):.2f}, "
              f"{(quals < 0.3).sum()} low-quality (<0.3) frames to down-weight")
    print(f"Saved -> {out_path}")
    if overlay:
        print(f"        aligned-depth checks in {debug_dir}")
    return records


def main():
    data_root = "./captures/session_05_20260624"

    use_spline = True       # gated smoothed-residual correction on top of affine
    overlay = True          # write aligned-depth + per-anchor error checks
    overlay_every = 50
    limit = None            # int for a quick first-N check

    run(data_root, use_spline=use_spline, overlay=overlay,
        overlay_every=overlay_every, limit=limit)


if __name__ == "__main__":
    main()
