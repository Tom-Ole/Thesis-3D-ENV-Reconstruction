"""
STEP 3 - Pose Graph Optimization (Global Consistency)
=====================================================

Removes the drift that Step 1's frame-to-frame odometry accumulates by building
a pose graph over keyframes and optimizing it with loop-closure constraints.

Graph (Open3D conventions):
    nodes  = keyframe poses (node.pose = local->world), initialized from Step 1
    edges  = relative transforms T_{target<-source} with an information matrix
        - sequential (odometry) edges between consecutive keyframes  -> certain
        - loop-closure edges between revisited keyframes             -> uncertain

Loop closures (where global registration / RANSAC belongs, not Step 1):
    candidates proposed by proximity in the current estimate, cheap-screened
    with evaluate_registration, then verified by ICP. For scenes with NO good
    prior (large-scale / kidnapped robot) swap the proposer for Scan Context or
    FPFH+RANSAC global registration -- left as a TODO.

Optimization: Levenberg-Marquardt with Open3D's line process, which
down-weights/prunes inconsistent loop edges (robust to false positives).

Outputs: output/pose_graph_poses.npy (optimized keyframe poses),
         output/pose_graph_keyframes.npy (their frame indices),
         output/global_map_optimized.ply

NOT implemented (TODO): Manhattan-world / structural constraints; Scan Context.

Run Step 1 first (writes trajectory_poses_world.npy).
"""

import numpy as np
import open3d as o3d

import common
import step2_mapping as step2

reg = o3d.pipelines.registration


# --- Tunables ---------------------------------------------------------------
KF_VOXEL = 0.10                  # keyframe downsample for registration
SEQ_MAX_CORR = 0.30              # correspondence distance, sequential edges
LOOP_MAX_CORR = 0.30            # correspondence distance, loop edges

MIN_LOOP_GAP = 5                 # min keyframe index gap to count as a loop
LOOP_RADIUS = 3.0                # candidate proximity in current estimate (m)
PRESCREEN_FITNESS = 0.30         # cheap reject before running full ICP
LOOP_MIN_FITNESS = 0.60         # accept a verified loop edge above this

# Global optimization
PRUNE_THRESHOLD = 0.25
PREFERENCE_LOOP = 2.0


def prep_keyframe_clouds(dataset, keyframes, voxel=KF_VOXEL):
    """Downsampled sensor-frame clouds with normals, one per keyframe."""
    clouds = []
    for i in keyframes:
        c = dataset[i].read_cloud_sensor().voxel_down_sample(voxel)
        c.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
        clouds.append(c)
    return clouds


def register(source, target, init, max_corr):
    """Point-to-plane ICP + information matrix. Returns (T, info, fitness)."""
    res = reg.registration_icp(
        source, target, max_corr, init,
        reg.TransformationEstimationPointToPlane())
    info = reg.get_information_matrix_from_point_clouds(
        source, target, max_corr, res.transformation)
    return res.transformation, info, res.fitness


def build_pose_graph(kf_poses, clouds):
    """
    Build the pose graph. Returns (graph, loop_pairs).
    kf_poses[k] is keyframe k's absolute pose (local->world).
    """
    graph = reg.PoseGraph()
    for T in kf_poses:
        graph.nodes.append(reg.PoseGraphNode(T.copy()))

    K = len(kf_poses)

    # Sequential (odometry) edges.
    for a in range(K - 1):
        b = a + 1
        init = np.linalg.inv(kf_poses[b]) @ kf_poses[a]   # T_{b<-a} prior
        T, info, _ = register(clouds[a], clouds[b], init, SEQ_MAX_CORR)
        graph.edges.append(reg.PoseGraphEdge(a, b, T, info, uncertain=False))

    # Loop-closure edges.
    loop_pairs = []
    pos = np.array([T[:3, 3] for T in kf_poses])
    for a in range(K):
        for b in range(a + MIN_LOOP_GAP, K):
            if np.linalg.norm(pos[a] - pos[b]) > LOOP_RADIUS:
                continue
            init = np.linalg.inv(kf_poses[b]) @ kf_poses[a]
            # Cheap screen before the expensive ICP + information matrix.
            e = reg.evaluate_registration(
                clouds[a], clouds[b], LOOP_MAX_CORR, init)
            if e.fitness < PRESCREEN_FITNESS:
                continue
            T, info, fit = register(clouds[a], clouds[b], init, LOOP_MAX_CORR)
            if fit >= LOOP_MIN_FITNESS:
                graph.edges.append(
                    reg.PoseGraphEdge(a, b, T, info, uncertain=True))
                loop_pairs.append((a, b))

    print(f"Pose graph: {K} nodes, {K-1} sequential edges, "
          f"{len(loop_pairs)} loop-closure edges")
    return graph, loop_pairs


def optimize(graph, max_corr=SEQ_MAX_CORR):
    """Robust global optimization in place. Returns optimized poses (K,4,4)."""
    option = reg.GlobalOptimizationOption(
        max_correspondence_distance=max_corr,
        edge_prune_threshold=PRUNE_THRESHOLD,
        preference_loop_closure=PREFERENCE_LOOP,
        reference_node=0)
    reg.global_optimization(
        graph,
        reg.GlobalOptimizationLevenbergMarquardt(),
        reg.GlobalOptimizationConvergenceCriteria(),
        option)
    return np.stack([n.pose for n in graph.nodes])


def loop_lineset(poses, loop_pairs):
    """Red lines between keyframes joined by a loop-closure edge."""
    if not loop_pairs:
        return None
    pts = poses[:, :3, 3]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(loop_pairs))
    ls.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(loop_pairs))
    return ls


def visualize(before_poses, after_poses, loop_pairs, optimized_map=None):
    geoms = [
        common.trajectory_lineset(before_poses, color=(1.0, 0.5, 0.0)),  # before
        common.trajectory_lineset(after_poses, color=(0.1, 0.9, 0.1)),   # after
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
    ]
    ls = loop_lineset(after_poses, loop_pairs)
    if ls is not None:
        geoms.append(ls)
    if optimized_map is not None:
        geoms.insert(0, optimized_map)
    common.draw(geoms, title="Step 3 - pose graph (orange=before, green=after, red=loops)")


def main():
    data_root = "./captures/session_05_20260624"
    dataset = common.SessionDataset(data_root)
    poses_world = step2.load_poses(dataset)

    keyframes = step2.select_keyframes(poses_world)
    kf_poses = poses_world[keyframes]
    print(f"Keyframes: {len(keyframes)}/{len(poses_world)}")

    clouds = prep_keyframe_clouds(dataset, keyframes)
    graph, loop_pairs = build_pose_graph(kf_poses, clouds)

    optimized = optimize(graph)

    moved = np.linalg.norm(optimized[:, :3, 3] - kf_poses[:, :3, 3], axis=1)
    print(f"Optimization moved keyframes: mean {moved.mean():.3f} m, "
          f"max {moved.max():.3f} m")

    # Save optimized keyframe poses + indices.
    out = dataset.ensure_output_dir()
    np.save(f"{out}/pose_graph_poses.npy", optimized)
    np.save(f"{out}/pose_graph_keyframes.npy", np.array(keyframes))
    print(f"Saved optimized poses -> {out}/pose_graph_poses.npy")

    # Re-fuse the map with optimized poses (build_global_map indexes keyframes).
    poses_opt_full = poses_world.copy()
    for k, idx in enumerate(keyframes):
        poses_opt_full[idx] = optimized[k]
    optimized_map = step2.build_global_map(dataset, poses_opt_full, keyframes)
    step2.save_map(dataset, optimized_map, name="global_map_optimized.ply")

    visualize(kf_poses, optimized, loop_pairs, optimized_map)


if __name__ == "__main__":
    main()
