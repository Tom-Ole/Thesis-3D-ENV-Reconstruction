"""
STEP B1 - Image pose sync & refinement  (image branch, after B0)
================================================================

B0 (step0_camera_prep.py) emitted output/cameras/manifest.json: for every image
an upright pinhole image, a matching K, and ``odom_T_camera`` taken from Spot's
RAW per-image fused odometry.  That raw pose drifts independently of the LiDAR
SLAM solution (Steps 1-3).  B1 closes the two remaining gaps the TODO lists for
the image branch:

  1. TEMPORAL SYNC.  Images are 5 Hz, LiDAR ~3 Hz on one shared robot clock, so
     no image lands exactly on a scan.  For each image we find the time-nearest
     LiDAR scan; B3 projects *that* scan into the image as the sparse metric
     depth anchor.  We record the scan index and the |image - scan| gap.

  2. POSE REFINEMENT.  We make each camera pose drift-consistent with the LiDAR
     SLAM trajectory.  Per scan i the SLAM solution implies an odom-frame
     correction

         C_i = ref_i @ inv(spot_i)

     where ``spot_i`` is Spot's raw odom_T_sensor and ``ref_i`` is the refined
     pose (Step 1 dense trajectory, or Step 3 pose graph).  C is small and
     temporally smooth, so for an image at time t we SE(3)-interpolate C between
     the bracketing scans and left-apply it to the raw camera pose:

         odom_T_camera_refined = C(t) @ odom_T_camera_raw

     This is exact: the static body_T_camera extrinsic is untouched and C acts
     on the shared odom frame, so correcting the body's drift corrects the
     camera identically.

On session_05 (a near-stationary spin, tiny drift) the correction is small by
design - the real deliverable here is the sync plus a method that is ready for a
translating capture.

Run (from src/odometry/, after step1 [+ optional step3]):
    python step_b1_image_poses.py

Output (under <session>/output/cameras/):
    poses_b1.json              one record per image (refined pose, anchor scan,
                               dt, correction magnitude) - merged transparently
                               by common.load_camera_manifest()
    debug/<stem>_b1.jpg        raw-vs-refined LiDAR reprojection (when enabled)
"""

import os
import json

import numpy as np
import cv2
from scipy.spatial.transform import Rotation, Slerp

import common


# --- SE(3) helpers ----------------------------------------------------------

def _interp_se3(Ta: np.ndarray, Tb: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate two 4x4 transforms: slerp rotation, lerp translation."""
    slerp = Slerp([0.0, 1.0],
                  Rotation.from_matrix(np.stack([Ta[:3, :3], Tb[:3, :3]])))
    T = np.eye(4)
    T[:3, :3] = slerp([alpha])[0].as_matrix()
    T[:3, 3] = (1.0 - alpha) * Ta[:3, 3] + alpha * Tb[:3, 3]
    return T


def _magnitude(T: np.ndarray):
    """(translation metres, rotation degrees) of a 4x4 transform."""
    tr = float(np.linalg.norm(T[:3, 3]))
    cos = np.clip((np.trace(T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return tr, float(np.degrees(np.arccos(cos)))


# --- Correction track (refined-vs-raw SLAM drift over time) -----------------

def build_correction_track(dataset: common.SessionDataset, use_pose_graph: bool):
    """Time-sorted (times_nsec, corrections, scan_idx) sampling SLAM drift.

    Dense Step 1 trajectory samples every scan; the Step 3 pose graph samples
    only keyframes.  Either way we interpolate C over these samples at image
    time, so the rest of B1 is agnostic to the choice.
    """
    out_dir = dataset.output_dir
    spot = dataset.spot_trajectory()                       # (N,4,4) raw
    times_all = np.array([f.t_nsec for f in dataset.frames], dtype=np.int64)

    if use_pose_graph:
        ref = np.load(os.path.join(out_dir, "pose_graph_poses.npy"))
        idx = np.load(os.path.join(out_dir, "pose_graph_keyframes.npy")).astype(int)
        source = "pose_graph_poses.npy"
    else:
        ref = np.load(os.path.join(out_dir, "trajectory_poses_world.npy"))
        idx = np.arange(len(dataset.frames))
        source = "trajectory_poses_world.npy"

    if len(ref) != len(idx):
        raise RuntimeError(
            f"{source}: {len(ref)} poses but {len(idx)} scan indices")

    C = ref @ np.linalg.inv(spot[idx])                     # (M,4,4) corrections
    times = times_all[idx]

    order = np.argsort(times)                              # ensure monotonic
    return times[order], C[order], idx[order], times_all, source


def interp_correction(t_nsec: int, times: np.ndarray, C: np.ndarray):
    """C(t): interpolate the drift correction at image time t.

    Returns (C_t, pose_source, (idx_a, idx_b), alpha).  Images outside the scan
    time span clamp to the nearest endpoint.
    """
    j = int(np.searchsorted(times, t_nsec))
    if j <= 0:
        return C[0].copy(), "clamped", (0, 0), 0.0
    if j >= len(times):
        last = len(times) - 1
        return C[last].copy(), "clamped", (last, last), 0.0
    a, b = j - 1, j
    span = times[b] - times[a]
    alpha = 0.0 if span <= 0 else float((t_nsec - times[a]) / span)
    return _interp_se3(C[a], C[b], alpha), "interpolated", (a, b), alpha


def nearest_scan(t_nsec: int, times_all: np.ndarray):
    """(scan_index, dt_seconds) for the time-nearest LiDAR scan (over all scans)."""
    j = int(np.searchsorted(times_all, t_nsec))
    cands = [k for k in (j - 1, j) if 0 <= k < len(times_all)]
    k = min(cands, key=lambda i: abs(int(times_all[i]) - t_nsec))
    return int(k), abs(int(times_all[k]) - t_nsec) * 1e-9


# --- Validation overlay -----------------------------------------------------

def _draw_projection(img, K, w, h, cam_T_odom, pts_odom, max_depth=15.0):
    """Draw odom-frame points onto img (near=red -> far=blue). In-place copy."""
    out = img.copy()
    if len(pts_odom) == 0:
        return out
    cam = (cam_T_odom @ np.hstack(
        [pts_odom, np.ones((len(pts_odom), 1))]).T).T[:, :3]
    z = cam[:, 2]
    front = z > 1e-3
    cam, z = cam[front], z[front]
    uv = (K @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    u, v = uv[:, 0], uv[:, 1]
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    for ui, vi, zi in zip(u[inb].astype(int), v[inb].astype(int), z[inb]):
        t = min(zi, max_depth) / max_depth
        cv2.circle(out, (ui, vi), 1, (int(255 * t), 0, int(255 * (1 - t))), -1)
    return out


def _overlay_compare(cam, odom_T_ref, lidar_frame, ref_sensor_pose):
    """raw pose + raw-odom scan  |  refined pose + refined-odom scan.

    Each panel keeps camera and LiDAR in the SAME frame, so a correct refinement
    leaves both panels equally well aligned (it does not improve a single
    self-consistent pair - it makes ALL pairs share one global frame). Mismatch
    on the right would flag a bad correction / convention error.
    """
    img = cam.read_image()
    pts_raw = np.asarray(lidar_frame.read_cloud().points)          # raw odom
    pts_ref = np.asarray(lidar_frame.read_cloud_sensor()           # refined odom
                         .transform(ref_sensor_pose).points)

    left = _draw_projection(img, cam.K, cam.width, cam.height,
                            np.linalg.inv(cam.odom_T_camera), pts_raw)
    right = _draw_projection(img, cam.K, cam.width, cam.height,
                             np.linalg.inv(odom_T_ref), pts_ref)
    cv2.putText(left, "raw", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    cv2.putText(right, "refined", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    return np.hstack([left, right])


# --- Driver -----------------------------------------------------------------

def run(data_root: str, use_pose_graph: bool = False, overlay: bool = True,
        overlay_every: int = 50, limit: int | None = None):
    cams = common.load_camera_manifest(data_root)
    if limit:
        cams = cams[:limit]
    dataset = common.SessionDataset(data_root)

    times, C, scan_idx, times_all, source = build_correction_track(
        dataset, use_pose_graph)

    out_dir = os.path.join(os.path.abspath(data_root), "output", "cameras")
    debug_dir = os.path.join(out_dir, "debug")
    if overlay:
        os.makedirs(debug_dir, exist_ok=True)

    records, dts, shifts, corr_r, clamped = [], [], [], [], 0
    for n, cam in enumerate(cams):
        C_t, pose_source, (a, b), alpha = interp_correction(
            cam.t_nsec, times, C)
        odom_T_ref = C_t @ cam.odom_T_camera
        li, dt = nearest_scan(cam.t_nsec, times_all)

        # Camera optical-centre shift (interpretable) + applied rotation.
        shift = float(np.linalg.norm(
            odom_T_ref[:3, 3] - cam.odom_T_camera[:3, 3]))
        _, rot = _magnitude(C_t)

        clamped += pose_source == "clamped"
        dts.append(dt)
        shifts.append(shift)
        corr_r.append(rot)

        records.append({
            "source_image": cam.source_image,
            "source": cam.source,
            "camera": cam.camera,
            "t_nsec": cam.t_nsec,
            "odom_T_camera_refined": odom_T_ref.tolist(),
            "lidar_index": li,
            "lidar_t_nsec": int(times_all[li]),
            "dt_sec": dt,
            "pose_source": pose_source,
            "bracket_scan_idx": [int(scan_idx[a]), int(scan_idx[b])],
            "alpha": alpha,
            "correction_center_shift_m": shift,
            "correction_rot_deg": rot,
        })

        if overlay and n % overlay_every == 0:
            # Refined sensor pose of the anchor scan, via the same correction
            # track (mode-agnostic): C(t_scan) @ raw odom_T_sensor.
            C_scan, *_ = interp_correction(int(times_all[li]), times, C)
            ref_sensor = C_scan @ dataset.frames[li].odom_T_sensor
            ov = _overlay_compare(cam, odom_T_ref, dataset.frames[li],
                                  ref_sensor)
            stem = os.path.splitext(os.path.basename(cam.source_image))[0]
            cv2.imwrite(os.path.join(debug_dir, f"{stem}_b1.jpg"), ov)

    manifest = {
        "session": os.path.basename(os.path.abspath(data_root)),
        "trajectory_source": source,
        "count": len(records),
        "cameras": records,
    }
    out_path = os.path.join(out_dir, "poses_b1.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    dts, shifts, corr_r = np.array(dts), np.array(shifts), np.array(corr_r)
    print(f"B1: posed {len(records)} images against {source}")
    print(f"    sync   dt: median {np.median(dts) * 1e3:.1f} ms, "
          f"max {dts.max() * 1e3:.1f} ms")
    print(f"    pose shift vs raw: median {np.median(shifts) * 1e3:.1f} mm "
          f"/ max {shifts.max() * 1e3:.1f} mm, "
          f"rot max {corr_r.max():.2f} deg")
    print(f"    {clamped} image(s) outside scan time span (clamped)")
    print(f"Saved -> {out_path}")
    if overlay:
        print(f"        raw-vs-refined overlays in {debug_dir}")
    return records


def main():
    data_root = "./captures/session_05_20260624"

    # False -> Step 1 dense per-scan trajectory (recommended: no keyframe gaps).
    # True  -> Step 3 pose graph (loop-closure corrected; interpolated across
    # keyframes). session_05's loop closure is degenerate, so dense is fine.
    use_pose_graph = False

    overlay = True          # write raw-vs-refined reprojection checks
    overlay_every = 50      # sample 1 in N images for the overlay
    limit = None            # set to an int for a quick first-N check

    run(data_root, use_pose_graph=use_pose_graph, overlay=overlay,
        overlay_every=overlay_every, limit=limit)


if __name__ == "__main__":
    main()
