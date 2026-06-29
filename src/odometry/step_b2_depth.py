"""
STEP B2 - Dense monocular depth per image  (image branch, after B1)
===================================================================

Predicts a DENSE per-pixel depth map for every prepared image with a learned
monocular metric-depth model.  This is the densification engine: it fills the
textureless walls / ceiling / floor that the sparse Velodyne never hits, so the
fusion (Phase C) has real surfaces to mesh.

Model: Depth Anything V2, the METRIC INDOOR ViT-L checkpoint (Hypersim-trained,
~0-20 m), via HuggingFace transformers.  It outputs depth directly in METRES, so
B3 starts from a near-correct scale; B3 still re-anchors it to the LiDAR samples
(global scale+shift, then optional per-pixel correction) because a monocular
model cannot get absolute scale right on its own.  B2 does NOT touch scale - it
only produces clean, locally-consistent depth.

The model is wrapped behind a tiny ``DepthBackend`` so UniDepth v2 / Metric3D
(which can also consume our per-image K) can drop in later without touching the
driver.

Run (from the repo root, after step1 -> step_b1_image_poses; GPU strongly
preferred - RTX 5070 / sm_120 with torch cu13x is verified working):
    python src/odometry/step_b2_depth.py

Output (under <session>/output/depth/):
    <stem>.npy                 float16 depth map (H,W), metres (model-native)
    depth_b2.json              manifest (model id, depth_kind, per-image paths)
    debug/<stem>_depth.jpg     turbo-colourised depth + LiDAR-vs-pred check
"""

import os
import json
import time

import numpy as np
import cv2

import common


MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


# --- Depth backend ----------------------------------------------------------

class DepthAnythingV2Metric:
    """Depth Anything V2 (metric indoor) wrapped to: BGR uint8 -> depth (H,W) m.

    Lazy-loads the model on first use so importing this module stays cheap.
    Runs in fp16 on CUDA when available; the output is metric depth in metres.
    """

    depth_kind = "metric_m"

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = device
        self._proc = None
        self._model = None
        self._dtype = None

    def load(self):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self.device == "cuda" else torch.float32

        self._proc = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(
            self.model_id, dtype=self._dtype).to(self.device).eval()
        print(f"Loaded {self.model_id} on {self.device} ({self._dtype}).")
        return self

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """One BGR image -> float32 depth (H,W) in metres, at the input size."""
        import torch

        if self._model is None:
            self.load()
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self._proc(images=rgb, return_tensors="pt").to(self.device)
        if self._dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.inference_mode():
            outputs = self._model(pixel_values=inputs["pixel_values"])

        # post_process knows how the model output maps back to the input size.
        post = self._proc.post_process_depth_estimation(
            outputs, target_sizes=[(h, w)])
        return post[0]["predicted_depth"].float().cpu().numpy()


# --- Debug + validation -----------------------------------------------------

def _colorize(depth: np.ndarray) -> np.ndarray:
    """Turbo-colourised depth (2-98 percentile stretch) for eyeballing."""
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros((*depth.shape, 3), np.uint8)
    lo, hi = np.percentile(depth[finite], [2, 98])
    dn = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def lidar_depth_check(cam: common.Camera, depth: np.ndarray,
                      dataset: common.SessionDataset):
    """Preview the B2->B3 handoff: predicted vs LiDAR depth at projected points.

    Uses the RAW camera pose + raw-odom anchor scan (self-consistent, and so
    independent of whether B1 ran). Returns (median LiDAR/pred ratio ~ B3's
    global scale, Pearson r, sample count) or None if too few samples.
    """
    sc = cam.anchor_scan(dataset)
    if sc is None:
        return None
    pts = np.asarray(sc.read_cloud().points)
    if pts.size == 0:
        return None
    cam_T_odom = np.linalg.inv(cam.odom_T_camera)        # raw pose
    P = (cam_T_odom @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3]
    z = P[:, 2]
    uv = (cam.K @ P.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    u, v = uv[:, 0], uv[:, 1]
    inb = (z > 1e-3) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    u, v, zl = u[inb].astype(int), v[inb].astype(int), z[inb]
    zp = depth[v, u]
    good = np.isfinite(zp) & (zp > 1e-3)
    if good.sum() < 20:
        return None
    zp, zl = zp[good], zl[good]
    return float(np.median(zl / zp)), float(np.corrcoef(zp, zl)[0, 1]), int(good.sum())


# --- Driver -----------------------------------------------------------------

def run(data_root: str, model_id: str = MODEL_ID, device: str | None = None,
        overlay: bool = True, overlay_every: int = 50,
        stride: int = 1, limit: int | None = None):
    cams = common.load_camera_manifest(data_root)
    cams = cams[::stride]
    if limit:
        cams = cams[:limit]

    backend = DepthAnythingV2Metric(model_id, device).load()
    dataset = common.SessionDataset(data_root) if overlay else None

    out_dir = os.path.join(os.path.abspath(data_root), "output", "depth")
    debug_dir = os.path.join(out_dir, "debug")
    os.makedirs(out_dir, exist_ok=True)
    if overlay:
        os.makedirs(debug_dir, exist_ok=True)

    images, ratios, corrs = [], [], []
    t0 = time.time()
    for n, cam in enumerate(cams):
        bgr = cam.read_image()
        depth = backend.infer(bgr)
        stem = os.path.splitext(os.path.basename(cam.source_image))[0]
        depth_path = os.path.join(out_dir, f"{stem}.npy")
        np.save(depth_path, depth.astype(np.float16))

        images.append({
            "source_image": cam.source_image,
            "source": cam.source,
            "camera": cam.camera,
            "depth_path": depth_path,
            "height": int(depth.shape[0]),
            "width": int(depth.shape[1]),
            "min_m": float(np.nanmin(depth)),
            "max_m": float(np.nanmax(depth)),
        })

        if overlay and n % overlay_every == 0:
            vis = _colorize(depth)
            chk = lidar_depth_check(cam, depth, dataset)
            if chk:
                ratios.append(chk[0])
                corrs.append(chk[1])
                cv2.putText(vis, f"x{chk[0]:.2f} r={chk[1]:.2f} n={chk[2]}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
            cv2.imwrite(os.path.join(debug_dir, f"{stem}_depth.jpg"),
                        np.hstack([bgr, vis]))

        if n % 50 == 0:
            print(f"[{n + 1:04d}/{len(cams)}] {stem} "
                  f"depth {depth.min():.2f}-{depth.max():.2f} m")

    manifest = {
        "session": os.path.basename(os.path.abspath(data_root)),
        "model_id": model_id,
        "depth_kind": backend.depth_kind,
        "dtype": "float16",
        "count": len(images),
        "images": images,
    }
    out_path = os.path.join(out_dir, "depth_b2.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    dt = time.time() - t0
    print(f"\nB2: depth for {len(images)} images in {dt:.1f}s "
          f"({dt / max(len(images), 1) * 1e3:.0f} ms/img)")
    if ratios:
        print(f"    LiDAR check (sampled): scale x{np.median(ratios):.2f}, "
              f"pred-vs-LiDAR r median {np.median(corrs):.3f} "
              f"(>0.9 => B3 scale+shift will be well-conditioned)")
    print(f"Saved -> {out_path}")
    if overlay:
        print(f"        depth previews in {debug_dir}")
    return images


def main():
    data_root = "./captures/session_05_20260624"

    model_id = MODEL_ID
    device = None           # None -> CUDA if available, else CPU

    overlay = True          # write colourised depth + LiDAR cross-check
    overlay_every = 50      # sample 1 in N images for the preview
    stride = 1              # process every Nth camera (1 = all 1060)
    limit = None            # set to an int for a quick first-N check

    run(data_root, model_id=model_id, device=device, overlay=overlay,
        overlay_every=overlay_every, stride=stride, limit=limit)


if __name__ == "__main__":
    main()
