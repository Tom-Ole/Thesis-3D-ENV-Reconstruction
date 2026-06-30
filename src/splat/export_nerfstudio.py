"""
Stage 3 adapter: our pipeline -> Nerfstudio dataset for DN-Splatter
===================================================================

Exports the metric, posed, depth-bearing data we already produce (B1 refined
poses + B3 LiDAR-anchored metric depth + the prepared RGB) into the Nerfstudio
`transforms.json` layout that DN-Splatter's `normal-nerfstudio` dataparser reads.
DN-Splatter then trains a depth+normal-regularised 3DGS and exports a mesh -
the "hyper-realistic" stage.

Why this is a good fit: DN-Splatter is built exactly for posed-RGB + depth (+
normals) indoor scenes, which is precisely what stages 1-2 give us. We feed our
LiDAR-anchored METRIC depth as the depth supervision, so the splat geometry
inherits the right dimensions instead of guessing scale.

Conventions:
  * Our `Camera.pose` is odom_T_camera in the computer-vision frame (x right,
    y down, z forward). Nerfstudio/`transforms.json` wants camera-to-world in the
    OpenGL/Blender frame (x right, y up, z back), so we right-multiply by
    diag(1,-1,-1,1).
  * Depth is written as float32 .npy in METRES. Train with the dataparser's
    depth-unit scale = 1.0 and auto-orient/scale/center DISABLED to stay metric
    (see print-out at the end for the exact ns-train flags).

NOTE: depth field name / scale expectations are confirmed against the installed
dn-splatter `normal_nerfstudio` dataparser before the first real run; adjust
DEPTH_DIRNAME / the per-frame depth key if that parser differs.

Run (in the MAIN pipeline venv, after stages 1-2):
    python src/splat/export_nerfstudio.py
"""

import os
import json
import shutil

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "odometry"))
import common


# OpenGL <- OpenCV camera-frame flip (y, z).
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])

MIN_QUALITY = 0.3          # skip frames B3 flagged as low-quality
DEPTH_DIRNAME = "depth"    # nerfstudio depth subfolder


def export(data_root: str, out_dir: str | None = None,
           min_quality: float = MIN_QUALITY):
    cams = common.load_camera_manifest(data_root)
    if out_dir is None:
        out_dir = os.path.join(os.path.abspath(data_root), "output", "splat",
                               "nerfstudio")
    img_dir = os.path.join(out_dir, "images")
    dep_dir = os.path.join(out_dir, DEPTH_DIRNAME)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(dep_dir, exist_ok=True)

    frames = []
    used = 0
    for cam in cams:
        if cam.align_a is None:
            continue
        if cam.align_quality is not None and cam.align_quality < min_quality:
            continue
        depth = cam.read_depth_aligned()
        if depth is None:
            continue

        stem = f"frame_{used:05d}"
        # RGB: copy the prepared (upright, K-consistent) image in.
        ext = os.path.splitext(cam.image_path)[1] or ".jpg"
        img_name = f"{stem}{ext}"
        shutil.copyfile(cam.image_path, os.path.join(img_dir, img_name))
        # Depth: metric metres, float32 .npy (invalid -> 0).
        dep_name = f"{stem}.npy"
        np.save(os.path.join(dep_dir, dep_name),
                np.nan_to_num(depth, nan=0.0).astype(np.float32))

        c2w = cam.pose @ CV_TO_GL
        K = cam.K
        frames.append({
            "file_path": f"images/{img_name}",
            "depth_file_path": f"{DEPTH_DIRNAME}/{dep_name}",
            "transform_matrix": c2w.tolist(),
            "w": int(cam.width), "h": int(cam.height),
            "fl_x": float(K[0, 0]), "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        })
        used += 1

    transforms = {
        "camera_model": "OPENCV",
        "frames": frames,
    }

    # Seed the Gaussians with our LiDAR-anchored metric cloud (much better than
    # random init, and keeps the right dimensions). Same odom world frame as the
    # poses; train with orientation/center/scale disabled so it stays aligned.
    try:
        import open3d as o3d
        pcd, _ = common.load_fused_cloud(data_root)
        pcd = pcd.voxel_down_sample(0.03)
        o3d.io.write_point_cloud(os.path.join(out_dir, "points3D.ply"), pcd)
        transforms["ply_file_path"] = "points3D.ply"
        print(f"  init cloud: points3D.ply ({len(pcd.points)} pts from fused_cloud)")
    except Exception as e:
        print(f"  (no init cloud: {e}; DN-Splatter will use random init)")

    with open(os.path.join(out_dir, "transforms.json"), "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"Exported {used}/{len(cams)} frames -> {out_dir}")
    print(f"  images/ + {DEPTH_DIRNAME}/ (metric metres) + transforms.json")
    print("\nTrain (in the splat venv), keeping metric scale:")
    print(f"  ns-train dn-splatter --pipeline.model.use-depth-loss True "
          f"--pipeline.model.depth-lambda 0.2 --pipeline.model.use-normal-loss "
          f"True --pipeline.model.normal-supervision depth "
          f"normal-nerfstudio --data {out_dir} "
          f"--orientation-method none --center-method none "
          f"--auto-scale-poses False")
    print("Mesh:  gs-mesh o3dtsdf --load-config <run>/config.yml "
          f"--output-dir {out_dir}/mesh")
    return out_dir


def main():
    export("./captures/session_05_20260624")


if __name__ == "__main__":
    main()
