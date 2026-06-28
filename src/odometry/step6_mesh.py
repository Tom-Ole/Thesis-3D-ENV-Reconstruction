"""
STEP 6 - Surface Meshing   (produces the END-GOAL mesh)
=======================================================

Turns the Step 4 surfel cloud into a triangle mesh via Poisson surface
reconstruction, then trims and (optionally) simplifies it.

    surfels.ply (oriented points + colour)
        -> Poisson reconstruction        (watertight-ish surface)
        -> density-quantile trim          (cut low-support "balloon")
        -> crop to input bounding box     (drop extrapolated regions)
        -> optional quadric simplification
        -> mesh.ply

Colour: carries Step 4's placeholder (confidence) colours for now; real
texture/vertex colour comes once Step 5 (multi-camera colour) is in.

TODO:
    - TSDF + Marching Cubes path once depth images are captured (often cleaner
      indoors than Poisson)
    - cleanup: remove small components, fill holes, Taubin smoothing
    - texturing: bake Step 5 colour to vertices / UV atlas

Run Step 4 first (writes surfels.ply).
"""

import numpy as np
import open3d as o3d

import common


# --- Tunables ---------------------------------------------------------------
POISSON_DEPTH = 10           # octree depth; higher = finer + slower
DENSITY_QUANTILE = 0.04      # drop this fraction of lowest-support vertices
TARGET_TRIANGLES = 300_000   # quadric-decimation target; 0 to skip
CROP_MARGIN = 0.20           # m, expand input bbox before cropping the mesh


def load_surfels(dataset, name="surfels.ply"):
    path = f"{dataset.output_dir}/{name}"
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.12, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(15)
    return pcd


def poisson_mesh(pcd, depth=POISSON_DEPTH, density_quantile=DENSITY_QUANTILE,
                 crop_margin=CROP_MARGIN):
    """Poisson reconstruction + density trim + bbox crop."""
    mesh, densities = \
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth)
    densities = np.asarray(densities)

    # Poisson balloons into empty space where there's no support -> trim the
    # lowest-density vertices.
    thr = np.quantile(densities, density_quantile)
    mesh.remove_vertices_by_mask(densities < thr)

    # Drop anything Poisson extrapolated beyond the actual scanned volume.
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        bbox.min_bound - crop_margin, bbox.max_bound + crop_margin)
    mesh = mesh.crop(bbox)

    mesh.compute_vertex_normals()
    return mesh


def simplify(mesh, target=TARGET_TRIANGLES):
    if target and len(mesh.triangles) > target:
        mesh = mesh.simplify_quadric_decimation(target)
        mesh.compute_vertex_normals()
    return mesh


def save_mesh(dataset, mesh, name="mesh.ply"):
    out = dataset.ensure_output_dir()
    path = f"{out}/{name}"
    o3d.io.write_triangle_mesh(path, mesh)
    print(f"Saved mesh -> {path}")
    return path


def visualize(mesh):
    common.draw([mesh], title="Step 6 - surface mesh")


def main():
    data_root = "./captures/session_05_20260624"
    dataset = common.SessionDataset(data_root)

    pcd = load_surfels(dataset)
    print(f"Input surfels: {len(pcd.points)}")

    mesh = poisson_mesh(pcd)
    print(f"Poisson mesh: {len(mesh.vertices)} verts, "
          f"{len(mesh.triangles)} tris")

    mesh = simplify(mesh)
    print(f"After simplify: {len(mesh.triangles)} tris")

    save_mesh(dataset, mesh)
    visualize(mesh)


if __name__ == "__main__":
    main()
