"""
PHASE D - Surface + Mesh  (the first actually-good mesh)
========================================================

Meshes the image-branch + LiDAR geometry into a clean triangle mesh.  Two paths:

  TSDF  (default, recommended) - volumetrically integrate the B3 aligned depth
        maps into a truncated signed-distance field, then Marching Cubes.  Each
        surface voxel averages MANY confidence-gated depth observations (the
        near-stationary spin gives huge overlap), so walls come out flat and the
        surface is watertight with RGB baked in.  This is dramatically cleaner
        than meshing the back-projected points directly.

  POISSON (baseline) - Poisson on Phase C's fused_cloud.  Kept for comparison;
        it meshes ~1.2 M individually-noisy monocular-depth points, so the
        surface is bumpy ("tinfoil").  See step6_mesh.py for the LiDAR-only one.

Why TSDF wins here: the noise was per-view depth disagreement + grazing-angle
smear baked into points.  TSDF resolves it in the SDF by weighted averaging
before a surface ever exists, instead of fitting a surface through the noise.

Inputs are all wired: `Camera.read_depth_aligned()` (B3 metric depth, LiDAR-
anchored), `.confidence()` (per-pixel weight: quality x edge x range-taper),
`.read_image()`, refined `.pose`/`.cam_T_odom`, `.K`.

Run (from repo root, after step_c_fuse for the Poisson path / step_b3 for TSDF):
    python src/odometry/step_d_mesh.py

Output (under <session>/output/fusion/):
    mesh_fused.ply        cleaned triangle mesh (Phase E will texture it)
    mesh_fused_preview.png shaded render for a quick look
"""

import os

import numpy as np
import open3d as o3d

import common
import step6_mesh as step6


# --- TSDF tunables ----------------------------------------------------------
TSDF_VOXEL = 0.03            # SDF voxel (m): smaller=more detail+noise
TSDF_TRUNC = 0.12            # truncation distance (~4x voxel)
DEPTH_TRUNC = 8.0            # ignore depth beyond this (m)
MIN_QUALITY = 0.35           # skip frames with B3 fit_quality below this
CONF_THR = 0.45              # zero out pixels below this confidence (cuts fringe)

# --- shared cleanup ---------------------------------------------------------
MIN_COMPONENT_TRIS = 2000    # drop connected components smaller than this
TAUBIN_ITERS = 8             # volume-preserving smoothing passes
TARGET_TRIANGLES = 0         # quadric-decimation target (0 = keep full detail)

# --- Poisson (baseline) tunables --------------------------------------------
POISSON_DEPTH = 10
DENSITY_QUANTILE = 0.035
CROP_MARGIN = 0.20


def tsdf_mesh(data_root, voxel=TSDF_VOXEL, trunc=TSDF_TRUNC,
              depth_trunc=DEPTH_TRUNC, min_quality=MIN_QUALITY,
              conf_thr=CONF_THR):
    """Volumetric TSDF integration of the confidence-gated aligned depth maps."""
    cams = common.load_camera_manifest(data_root)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    used = 0
    for cam in cams:
        if cam.align_a is None or (cam.align_quality or 0.0) < min_quality:
            continue
        z = cam.read_depth_aligned()
        if z is None:
            continue
        conf = cam.confidence(z=z)
        z = z.copy()
        z[(conf < conf_thr) | ~np.isfinite(z)] = 0.0     # confidence-gate
        if (z > 0).sum() < 1000:
            continue
        rgb = np.ascontiguousarray(cam.read_image()[:, :, ::-1])  # BGR->RGB
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(np.ascontiguousarray(z.astype(np.float32))),
            depth_scale=1.0, depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False)
        K = cam.K
        intr = o3d.camera.PinholeCameraIntrinsic(
            cam.width, cam.height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        vol.integrate(rgbd, intr, cam.cam_T_odom)        # extrinsic = world->cam
        used += 1

    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print(f"TSDF: integrated {used}/{len(cams)} frames -> "
          f"{len(mesh.triangles)} tris")
    return mesh


def poisson_mesh(data_root):
    """Baseline: Poisson on Phase C's fused point cloud."""
    pcd, attrs = common.load_fused_cloud(data_root)
    mesh = step6.poisson_mesh(pcd, depth=POISSON_DEPTH,
                              density_quantile=DENSITY_QUANTILE,
                              crop_margin=CROP_MARGIN)
    print(f"Poisson: {len(mesh.triangles)} tris (before cleanup)")
    return mesh


def postprocess(mesh, min_component_tris=MIN_COMPONENT_TRIS,
                taubin_iters=TAUBIN_ITERS, target_triangles=TARGET_TRIANGLES):
    """Shared cleanup: drop floaters, hygiene, smooth, (optional) simplify."""
    labels, n_per, _ = mesh.cluster_connected_triangles()
    labels, n_per = np.asarray(labels), np.asarray(n_per)
    mesh.remove_triangles_by_mask(n_per[labels] < min_component_tris)
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    if taubin_iters:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=taubin_iters)
    if target_triangles and len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    mesh.compute_vertex_normals()
    return mesh


def render_preview(mesh, path, front=(0.4, -0.8, 0.45), zoom=0.62):
    """Save a shaded offscreen render (legacy GL visualizer works headless)."""
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=960)
        vis.add_geometry(mesh)
        vis.get_render_option().mesh_show_back_face = True
        ctr = vis.get_view_control()
        ctr.set_front(list(front)); ctr.set_up([0, 0, 1])
        ctr.set_lookat(mesh.get_center()); ctr.set_zoom(zoom)
        vis.poll_events(); vis.update_renderer()
        vis.capture_screen_image(path, do_render=True)
        vis.destroy_window()
        print(f"        preview -> {path}")
    except Exception as e:
        print(f"        (preview render skipped: {e})")


def run(data_root: str, method: str = "tsdf"):
    if method == "tsdf":
        mesh = tsdf_mesh(data_root)
    elif method == "poisson":
        mesh = poisson_mesh(data_root)
    else:
        raise ValueError(f"unknown method {method!r} (tsdf|poisson)")

    mesh = postprocess(mesh)

    out_dir = os.path.join(os.path.abspath(data_root), "output", "fusion")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "mesh_fused.ply")
    o3d.io.write_triangle_mesh(path, mesh)

    bb = mesh.get_axis_aligned_bounding_box()
    print(f"\nPhase D ({method}): {len(mesh.vertices)} verts, "
          f"{len(mesh.triangles)} tris (colored: {mesh.has_vertex_colors()})")
    print(f"    bbox min {np.round(bb.min_bound, 2)} max {np.round(bb.max_bound, 2)}")
    print(f"Saved -> {path}")
    render_preview(mesh, os.path.join(out_dir, "mesh_fused_preview.png"))
    return mesh


def main():
    data_root = "./captures/session_05_20260624"
    method = "tsdf"          # "tsdf" (clean, recommended) | "poisson" (baseline)
    run(data_root, method=method)


if __name__ == "__main__":
    main()
