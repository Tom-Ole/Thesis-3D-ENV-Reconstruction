"""
STEP 0 - Camera Prep  (prerequisite for the image branch, B0)
=============================================================

Turns Spot's raw fisheye captures into a clean, consistent input for the image
branch (B1/B3/E): for every image it emits an UPRIGHT image, a pinhole K that
matches those pixels, and the camera's pose in the odom frame.

Two things have to be reconciled (see TODO_updated.md, Key data facts):

  1. ROTATION (always on, lossless).  frontleft / frontright are stored already
     rotated 90deg clockwise (480x640) but their intrinsics still describe the
     raw 640x480 sensor.  We rotate the *intrinsics* (and the optical-frame
     pose) to match the pixels - no image resampling needed.  back / left /
     right are stored landscape and pass through untouched.

  2. FISHEYE UNDISTORTION (optional, OFF by default - easy to skip).  Spot ships
     a *pinhole* model with NO distortion coefficients, and the residual fisheye
     distortion is weak.  So the default treats every image as a plain pinhole
     (distortion_mode="none") and the rest of the pipeline never has to know.
     Pass --undistort fisheye (+ calibrated --dist coeffs) only if you decide
     the distortion is worth correcting; the output schema is identical either
     way, so downstream code is agnostic to the choice.

Decide whether you can skip undistortion with the built-in validation overlay:
it projects a time-matched LiDAR scan into each prepared image.  If LiDAR edges
line up with image edges under distortion_mode="none", undistortion buys you
nothing - leave it off.

Run (from src/odometry/):
    python step0_camera_prep.py

Configure the run via the constants in main() below (which session, whether to
undistort, etc.) - the rest of the pipeline (step1..step6) follows the same
hardcoded-main convention.

Outputs (under <session>/output/cameras/):
    manifest.json              one record per image (K, size, pose, mode, ...)
    debug/<stem>_overlay.jpg   LiDAR-reprojection check (when overlay enabled)
"""

import os
import json
from dataclasses import dataclass, asdict

import numpy as np
import cv2

import common


# Default cv2.fisheye distortion (k1..k4). Zeros == no-op: Spot provides no
# coefficients, so undistortion is only meaningful once these are calibrated.
DEFAULT_FISHEYE_DIST = (0.0, 0.0, 0.0, 0.0)

# Per-camera upright correction (clockwise degrees) that the capture metadata
# does NOT record. The 'right' camera is mounted so its image needs a 180deg
# turn. The INTRINSICS + pose flip for this is ALWAYS applied (Spot always
# reports raw-sensor intrinsics), independent of what the pixels look like.
EXTRA_ROTATION_DEG = {
    "right": 180,
}

# Whether to ALSO physically rotate the stored pixels for the cameras above.
# Needed only for sessions captured BEFORE the capture code was fixed to rotate
# at capture time. Newer captures already arrive upright, so leave this False
# (default) and only the intrinsics are reconciled.
ROTATE_LEGACY_PIXELS_DEFAULT = False


@dataclass
class CameraPrep:
    """Prepared camera: upright image + matching pinhole K + odom pose.

    This is the single contract the image branch consumes; it is identical
    whether or not undistortion ran, so skipping fisheye costs nothing
    downstream.
    """
    source: str                 # "frontleft_fisheye_image"
    camera: str                 # "frontleft"
    t_nsec: int
    image_path: str             # prepared image on disk (== source when no-op)
    width: int
    height: int
    K: list                     # 3x3 pinhole intrinsics for the prepared image
    odom_T_camera: list         # 4x4 optical-frame pose in odom
    distortion_mode: str        # "none" | "fisheye"
    rotation_applied: str | None  # rotation baked in by capture (metadata)
    extra_rotation_deg: int     # extra upright fix Step 0 applied (e.g. right=180)
    mask_path: str | None       # valid-pixel mask (only for fisheye remap)
    source_image: str           # original capture this was derived from


def _reconcile_rotation(frame: common.ImageFrame, rotate_legacy_pixels=False):
    """Reconcile orientation -> (K, (w,h), odom_T_optical, pixel_ops, extra_deg).

    Intrinsics and pose are always made consistent with the upright image (the
    optical frame is turned about its z axis to match). Pixels are only touched
    when they are NOT already upright on disk:

      * metadata ``rotation_applied`` -- the capture already rotated the pixels
        (front cams, and future 'right' captures), so K/pose are adjusted but no
        pixel op is emitted.
      * EXTRA_ROTATION_DEG with no such metadata -- the intrinsics/pose are still
        turned (always), and a pixel op is emitted *only* when
        ``rotate_legacy_pixels`` is set (old sessions captured upside down).
    """
    K = frame.K_raw.copy()
    w, h = frame.raw_size
    pose = frame.odom_T_sensor.copy()
    pixel_ops = []

    meta_rot = frame.rotation_applied

    # (1) rotation the capture already baked into the stored pixels.
    if meta_rot == "ROTATE_90_CLOCKWISE":
        K, (w, h) = common.rotate_intrinsics_90cw(K, w, h)
        # Image turned +90deg about the optical axis -> turn the frame the same
        # way (right-multiply by the inverse rotation).
        pose = pose @ np.linalg.inv(common.ROT_Z_90)
    elif meta_rot == "ROTATE_180":
        K, (w, h) = common.rotate_intrinsics_180(K, w, h)
        pose = pose @ common.ROT_Z_180     # self-inverse
    elif meta_rot:
        raise NotImplementedError(
            f"Unhandled rotation_applied={meta_rot!r} for {frame.source}")

    # (2) correction not recorded in metadata. Always fix intrinsics/pose; only
    # rotate pixels for legacy captures that are still upside down on disk.
    extra_deg = 0 if meta_rot else EXTRA_ROTATION_DEG.get(frame.camera_name, 0)
    if extra_deg == 180:
        K, (w, h) = common.rotate_intrinsics_180(K, w, h)
        pose = pose @ common.ROT_Z_180     # self-inverse
        if rotate_legacy_pixels:
            pixel_ops.append(cv2.ROTATE_180)
    elif extra_deg:
        raise NotImplementedError(
            f"Unhandled EXTRA_ROTATION_DEG={extra_deg} for {frame.camera_name} "
            f"(only 180 implemented)")

    return K, (w, h), pose, pixel_ops, extra_deg


def prepare_camera(frame: common.ImageFrame, out_dir: str,
                   undistort: bool = False,
                   dist_coeffs=DEFAULT_FISHEYE_DIST,
                   rotate_legacy_pixels: bool = False) -> CameraPrep:
    """Produce a CameraPrep for one image.

    undistort=False (default) -> rotation only; the image passes through by
    reference unless a legacy pixel fix is requested (then it is rewritten
    upright). undistort=True -> additionally remap through a cv2.fisheye model
    and write the corrected image + valid mask.
    """
    K, (w, h), odom_T_cam, pixel_ops, extra_deg = _reconcile_rotation(
        frame, rotate_legacy_pixels=rotate_legacy_pixels)
    mode = "none"
    image_path = frame.image_path
    mask_path = None
    stem = os.path.splitext(os.path.basename(frame.image_path))[0]

    # Read + rewrite only when we actually change pixels (extra rotation and/or
    # undistortion). Pure rotation-of-intrinsics cases pass through untouched.
    if pixel_ops or undistort:
        img = frame.read_image()
        for op in pixel_ops:
            img = cv2.rotate(img, op)

        if undistort:
            mode = "fisheye"
            D = np.asarray(dist_coeffs, dtype=np.float64).reshape(4, 1)
            # New K: keep the same intrinsics so depth stays metric-comparable;
            # estimateNewCameraMatrixForUndistortRectify could re-frame the FOV
            # but we keep it 1:1 for predictable downstream geometry.
            new_K = K.copy()
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
            img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
            # Valid mask: where a source pixel mapped in (borders go black).
            ones = np.full((h, w), 255, np.uint8)
            mask = cv2.remap(ones, map1, map2, interpolation=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT)
            mask_path = os.path.join(out_dir, f"{stem}_mask.png")
            cv2.imwrite(mask_path, mask)
            K = new_K

        image_path = os.path.join(out_dir, f"{stem}_prep.jpg")
        cv2.imwrite(image_path, img)

    return CameraPrep(
        source=frame.source,
        camera=frame.camera_name,
        t_nsec=frame.t_nsec,
        image_path=image_path,
        width=int(w),
        height=int(h),
        K=K.tolist(),
        odom_T_camera=odom_T_cam.tolist(),
        distortion_mode=mode,
        rotation_applied=frame.rotation_applied,
        extra_rotation_deg=int(extra_deg),
        mask_path=mask_path,
        source_image=frame.image_path,
    )


def _nearest_lidar(lidar, t_nsec: int):
    """The LiDAR frame whose timestamp is closest to t_nsec."""
    times = np.array([f.t_nsec for f in lidar.frames])
    return lidar.frames[int(np.argmin(np.abs(times - t_nsec)))]


def overlay_lidar(prep: CameraPrep, lidar_frame, max_depth=15.0):
    """Project a LiDAR scan into a prepared image (rotation/extrinsics check).

    Returns a BGR image with LiDAR points drawn, coloured near=red -> far=blue.
    A correct prep makes the points land on the matching scene geometry.
    """
    img = cv2.imread(prep.image_path, cv2.IMREAD_COLOR)
    K = np.array(prep.K)
    odom_T_cam = np.array(prep.odom_T_camera)
    cam_T_odom = np.linalg.inv(odom_T_cam)

    pts = np.asarray(lidar_frame.read_cloud().points)  # odom frame
    if pts.size == 0:
        return img
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    cam = (cam_T_odom @ pts_h.T).T[:, :3]               # optical frame

    z = cam[:, 2]
    front = z > 1e-3
    cam, z = cam[front], z[front]
    uv = (K @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    u, v = uv[:, 0], uv[:, 1]

    inb = (u >= 0) & (u < prep.width) & (v >= 0) & (v < prep.height)
    u, v, z = u[inb].astype(int), v[inb].astype(int), z[inb]
    for ui, vi, zi in zip(u, v, z):
        t = min(zi, max_depth) / max_depth          # 0 near -> 1 far
        color = (int(255 * t), 0, int(255 * (1 - t)))  # BGR: red->blue
        cv2.circle(img, (ui, vi), 1, color, -1)
    return img


def run(data_root: str, undistort: bool = False,
        dist_coeffs=DEFAULT_FISHEYE_DIST, overlay: bool = True,
        rotate_legacy_pixels: bool = False, limit: int | None = None):
    frames = common.load_image_frames(data_root)
    if limit:
        frames = frames[:limit]
    if not frames:
        raise RuntimeError(f"No image frames found under {data_root}")

    out_dir = os.path.join(os.path.abspath(data_root), "output", "cameras")
    debug_dir = os.path.join(out_dir, "debug")
    os.makedirs(out_dir, exist_ok=True)
    if overlay:
        os.makedirs(debug_dir, exist_ok=True)
        lidar = common.SessionDataset(data_root)

    if undistort and tuple(dist_coeffs) == DEFAULT_FISHEYE_DIST:
        print("WARNING: --undistort fisheye with zero coefficients is a no-op. "
              "Pass calibrated --dist k1,k2,k3,k4 or leave undistortion off.")

    preps = []
    for fr in frames:
        prep = prepare_camera(fr, out_dir, undistort=undistort,
                              dist_coeffs=dist_coeffs,
                              rotate_legacy_pixels=rotate_legacy_pixels)
        preps.append(prep)
        if overlay:
            ov = overlay_lidar(prep, _nearest_lidar(lidar, fr.t_nsec))
            stem = os.path.splitext(os.path.basename(fr.image_path))[0]
            cv2.imwrite(os.path.join(debug_dir, f"{stem}_overlay.jpg"), ov)

    manifest = {
        "session": os.path.basename(os.path.abspath(data_root)),
        "distortion_mode": "fisheye" if undistort else "none",
        "count": len(preps),
        "cameras": [asdict(p) for p in preps],
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Prepared {len(preps)} images (mode="
          f"{'fisheye' if undistort else 'none'}).")
    print(f"Saved -> {manifest_path}")
    if overlay:
        print(f"        overlays in {debug_dir}")
    return preps


def main():
    data_root = "./captures/session_05_20260624"

    # Fisheye undistortion is OFF by default - Spot gives no distortion coeffs
    # and the distortion is weak, so we treat the images as plain pinhole.
    # Flip this to True (and set DIST_COEFFS to a calibrated k1,k2,k3,k4) only
    # if the overlay shows undistortion is actually needed.
    undistort = False
    dist_coeffs = DEFAULT_FISHEYE_DIST

    overlay = True       # write LiDAR-reprojection validation images
    limit = None         # set to an int for a quick first-N check

    # Physically rotate legacy upside-down pixels (e.g. 'right' captured before
    # the capture code was fixed). The intrinsics are reconciled either way;
    # only set this True for old sessions whose images are still upside down.
    rotate_legacy_pixels = True

    run(data_root, undistort=undistort, dist_coeffs=dist_coeffs,
        overlay=overlay, rotate_legacy_pixels=rotate_legacy_pixels, limit=limit)


if __name__ == "__main__":
    main()
