"""Step 2 -- Pose Graph Construction and Optimization.

Reads Step 1's keyframes (``captures/<session>/output/step1/keyframes.json``)
and builds a pose graph: sequential edges from Step 1's already odometry-seeded,
quality-gated ICP chain, plus loop-closure edges wherever the trajectory
revisits a place it has already seen. Frame-to-frame ICP (Step 1) has no way
to notice a revisit, so drift only ever accumulates; a loop closure gives the
optimizer a direct constraint tying the two visits together, pulling the whole
graph back into global consistency.

**Loop-closure candidates are proximity-based**, not appearance-based: no
place-recognition/descriptor system exists in this project, so "has the robot
been near here before" is answered by comparing Step 1's own position
estimates, then verified (or rejected) with the same multi-scale ICP Step 1
uses. This is the todo2-appropriate scope for a bachelor thesis Step 2, not a
full SLAM place-recognition pipeline.

**Optimizer:** todo2.md names g2o/Ceres, but neither has a workable
pip-installable Windows binding (g2opy needs a manual SWIG/C++ build with no
official wheels). This uses Open3D's own pose-graph optimization instead --
already installed, zero new dependencies --
``o3d.pipelines.registration.global_optimization`` with
``GlobalOptimizationLevenbergMarquardt``. Its robustness mechanism (the edge
``uncertain`` flag + ``edge_prune_threshold``, a switchable-constraints/
line-process scheme from Choi/Zhou/Koltun 2015) serves the same purpose as
todo2's "Huber/Cauchy kernel" ask -- reject bad loop closures -- via a
different mechanism than a literal M-estimator kernel.

Produces, under ``captures/<session>/output/step2/``:
- ``pose_graph_poses.npy`` / ``pose_graph_tum.txt`` -- optimized keyframe poses
- ``loop_closures.json`` -- accepted loop-closure edges + how well the
  optimizer honored each one
- ``debug/*.png`` -- three-trajectory map view, 2D pose-graph diagram, one
  loop-closure before/after example

Runnable standalone: edit the variables at the bottom of this file and run
either ``python -m src.reconstruction.pose_graph`` or
``python src/reconstruction/pose_graph.py`` directly from the project root.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    # Allow `python src/reconstruction/pose_graph.py` (no -m) by putting the
    # project root on sys.path so the `src.*` absolute imports below resolve.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from src.reconstruction.common import (
    MAX_CORR_DISTANCES,
    LidarFrame,
    build_map,
    coordinate_frame,
    crop_to_trajectory,
    draw,
    ensure_dir,
    load_lidar_frames,
    multiscale_icp,
    pose_disagreement,
    pyramid,
    trajectory_lineset,
)
from src.reconstruction.lidar_icp import draw_registration_result

# --- Loop-closure candidate search (proximity-based place recognition) -----
MIN_KEYFRAME_GAP = 15
MAX_LOOP_DISTANCE_M = 1.0
LOOP_MIN_FITNESS = 0.4  # stricter than Step 1's sequential gate -- a bad loop
LOOP_MAX_TRANS_MULTIPLE = 2.0  # closure is far more damaging than a bad sequential edge

# --- Global optimization -----------------------------------------------------
EDGE_PRUNE_THRESHOLD = 0.25
REFERENCE_NODE = 0
MAP_VOXEL_SIZE = 0.05


@dataclass
class Keyframe:
    scan_index: int
    t_nsec: int
    pose_initial: np.ndarray
    pose_refined: np.ndarray


def load_keyframes(step1_dir: Path) -> list[Keyframe]:
    entries = json.loads((step1_dir / "keyframes.json").read_text())
    return [
        Keyframe(
            scan_index=e["scan_index"],
            t_nsec=e["t_nsec"],
            pose_initial=np.array(e["pose_initial"]),
            pose_refined=np.array(e["pose_refined"]),
        )
        for e in entries
    ]


@dataclass
class LoopClosure:
    """An accepted loop-closure edge between keyframes ``i`` and ``j`` (i < j).

    ``transformation`` maps keyframe ``i``'s local sensor-frame points into
    keyframe ``j``'s local sensor frame (Open3D's edge/registration
    convention: transform(source) aligns onto target, source=i, target=j).
    """

    i: int
    j: int
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float


def find_loop_closure_candidates(
    keyframes: list[Keyframe], min_gap: int = MIN_KEYFRAME_GAP, max_distance: float = MAX_LOOP_DISTANCE_M
) -> list[tuple[int, int]]:
    """Non-adjacent keyframe pairs whose Step-1 positions are close together."""
    positions = np.array([kf.pose_refined[:3, 3] for kf in keyframes])
    candidates = []
    n = len(keyframes)
    for i in range(n):
        for j in range(i + min_gap, n):
            if float(np.linalg.norm(positions[i] - positions[j])) < max_distance:
                candidates.append((i, j))
    return candidates


def verify_loop_closure(
    i: int, j: int, keyframes: list[Keyframe], frame_by_index: dict[int, LidarFrame]
) -> Optional[LoopClosure]:
    """Multi-scale ICP between two spatially-close, non-adjacent keyframes.

    Seeded by the current (Step 1) relative pose estimate; rejects results
    with weak fitness or an implausibly large correction, since a false loop
    closure is much more damaging to the graph than a missed one.
    """
    cloud_i = frame_by_index[keyframes[i].scan_index].read_cloud_sensor()
    cloud_j = frame_by_index[keyframes[j].scan_index].read_cloud_sensor()
    init = np.linalg.inv(keyframes[j].pose_refined) @ keyframes[i].pose_refined
    relative, result = multiscale_icp(pyramid(cloud_i), pyramid(cloud_j), init)

    translation_m, _ = pose_disagreement(relative, init)
    if result.fitness < LOOP_MIN_FITNESS or translation_m > LOOP_MAX_TRANS_MULTIPLE * MAX_LOOP_DISTANCE_M:
        return None
    return LoopClosure(i, j, relative, result.fitness, result.inlier_rmse)


def find_and_verify_loop_closures(
    keyframes: list[Keyframe], frame_by_index: dict[int, LidarFrame]
) -> list[LoopClosure]:
    candidates = find_loop_closure_candidates(keyframes)
    print(f"[Step 2] {len(candidates)} loop-closure candidate pair(s) within {MAX_LOOP_DISTANCE_M} m")

    accepted = []
    for i, j in candidates:
        loop_closure = verify_loop_closure(i, j, keyframes, frame_by_index)
        if loop_closure is not None:
            accepted.append(loop_closure)
            print(
                f"[Step 2]   accepted keyframe {i} <-> {j}: "
                f"fitness={loop_closure.fitness:.3f} rmse={loop_closure.inlier_rmse:.4f}"
            )
    print(f"[Step 2] {len(accepted)}/{len(candidates)} candidate(s) verified as loop closures")
    return accepted


# ---------------------------------------------------------------------------
# Pose graph
# ---------------------------------------------------------------------------


def build_pose_graph(
    keyframes: list[Keyframe], frame_by_index: dict[int, LidarFrame], loop_closures: list[LoopClosure]
) -> o3d.pipelines.registration.PoseGraph:
    graph = o3d.pipelines.registration.PoseGraph()
    for kf in keyframes:
        graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(kf.pose_refined))

    max_corr = MAX_CORR_DISTANCES[-1]

    for idx in range(1, len(keyframes)):
        a, b = idx - 1, idx
        transformation = np.linalg.inv(keyframes[b].pose_refined) @ keyframes[a].pose_refined
        cloud_a = pyramid(frame_by_index[keyframes[a].scan_index].read_cloud_sensor())[-1]
        cloud_b = pyramid(frame_by_index[keyframes[b].scan_index].read_cloud_sensor())[-1]
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            cloud_a, cloud_b, max_corr, transformation
        )
        graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(a, b, transformation, information, uncertain=False)
        )

    for lc in loop_closures:
        cloud_i = pyramid(frame_by_index[keyframes[lc.i].scan_index].read_cloud_sensor())[-1]
        cloud_j = pyramid(frame_by_index[keyframes[lc.j].scan_index].read_cloud_sensor())[-1]
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            cloud_i, cloud_j, max_corr, lc.transformation
        )
        graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(lc.i, lc.j, lc.transformation, information, uncertain=True)
        )

    return graph


def optimize_pose_graph(graph: o3d.pipelines.registration.PoseGraph) -> np.ndarray:
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=MAX_CORR_DISTANCES[-1],
        edge_prune_threshold=EDGE_PRUNE_THRESHOLD,
        reference_node=REFERENCE_NODE,
    )
    o3d.pipelines.registration.global_optimization(
        graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )
    return np.stack([node.pose for node in graph.nodes])


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_keyframe_trajectory(keyframes: list[Keyframe], poses: np.ndarray, output_dir: Path, prefix: str) -> None:
    """Save keyframe poses as 4x4 matrices (.npy) and in TUM format with real timestamps."""
    np.save(output_dir / f"{prefix}_poses.npy", poses)
    with open(output_dir / f"{prefix}_tum.txt", "w") as handle:
        for kf, pose in zip(keyframes, poses):
            tx, ty, tz = pose[:3, 3]
            qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
            handle.write(
                f"{kf.t_nsec * 1e-9:.9f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )


def save_loop_closures_json(
    keyframes: list[Keyframe], loop_closures: list[LoopClosure], optimized_poses: np.ndarray, path: Path
) -> None:
    entries = []
    for lc in loop_closures:
        optimized_relative = np.linalg.inv(optimized_poses[lc.j]) @ optimized_poses[lc.i]
        residual_m, residual_deg = pose_disagreement(optimized_relative, lc.transformation)
        entries.append(
            {
                "keyframe_i": keyframes[lc.i].scan_index,
                "keyframe_j": keyframes[lc.j].scan_index,
                "fitness": lc.fitness,
                "inlier_rmse": lc.inlier_rmse,
                "transformation": lc.transformation.tolist(),
                "residual_after_optimization_m": residual_m,
                "residual_after_optimization_deg": residual_deg,
            }
        )
    path.write_text(json.dumps(entries, indent=2))


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def plot_map_with_trajectories(
    keyframes: list[Keyframe],
    frame_by_index: dict[int, LidarFrame],
    poses_raw: np.ndarray,
    poses_step1: np.ndarray,
    poses_step2: np.ndarray,
    debug_dir: Path,
    show: bool,
) -> None:
    """Fused map (Step-2 poses) with all three trajectories, viewed from directly above.

    Reuses Step 1's top-down camera + trajectory-height map cropping
    (``common.crop_to_trajectory`` / ``common.draw(top_down=True)``) so the
    loop-closure correction is directly comparable to Step 1's figure.
    """
    keyframe_frames = [frame_by_index[kf.scan_index] for kf in keyframes]
    fused_map = build_map(keyframe_frames, poses_step2, voxel_size=MAP_VOXEL_SIZE, every=1)
    fused_map = crop_to_trajectory(fused_map, poses_step2)
    fused_map.paint_uniform_color([0.75, 0.75, 0.75])
    geometries = [
        fused_map,
        trajectory_lineset(poses_raw, color=(0.0, 0.4, 1.0)),
        trajectory_lineset(poses_step1, color=(1.0, 0.5, 0.0)),
        trajectory_lineset(poses_step2, color=(0.0, 0.7, 0.1)),
        coordinate_frame(size=0.5),
    ]
    draw(
        geometries,
        title="Step 2 - pose graph (blue=raw, orange=Step1 ICP, green=Step2 optimized)",
        screenshot_path=debug_dir / "map_and_trajectory.png",
        show=show,
        point_size=1.5,
        top_down=True,
    )


def plot_pose_graph_2d(
    poses_step2: np.ndarray, loop_closures: list[LoopClosure], path: Path
) -> None:
    """Classic SLAM pose-graph diagram: nodes + sequential edges + loop closures."""
    xy = poses_step2[:, :2, 3]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xy[:, 0], xy[:, 1], "-o", color="tab:green", linewidth=1.2, markersize=3, label="keyframe / sequential edge")
    for lc in loop_closures:
        ax.plot(
            [xy[lc.i, 0], xy[lc.j, 0]],
            [xy[lc.i, 1], xy[lc.j, 1]],
            "--",
            color="tab:red",
            linewidth=1.2,
            label="loop closure" if lc is loop_closures[0] else None,
        )
    ax.scatter(*xy[0], color="green", zorder=5, label="start")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Step 2 - pose graph ({len(loop_closures)} loop closure(s))")
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loop_closure_example(
    keyframes: list[Keyframe],
    frame_by_index: dict[int, LidarFrame],
    loop_closures: list[LoopClosure],
    debug_dir: Path,
    show: bool,
) -> None:
    """Before/after for the best-fitness accepted loop closure, Open3D-tutorial style."""
    if not loop_closures:
        return
    best = max(loop_closures, key=lambda lc: lc.fitness)
    source = frame_by_index[keyframes[best.i].scan_index].read_cloud_sensor().voxel_down_sample(0.05)
    target = frame_by_index[keyframes[best.j].scan_index].read_cloud_sensor().voxel_down_sample(0.05)
    init = np.linalg.inv(keyframes[best.j].pose_refined) @ keyframes[best.i].pose_refined

    draw(
        draw_registration_result(source, target, init),
        title="Step 2 - loop closure: before ICP (Step 1 estimate)",
        screenshot_path=debug_dir / "loop_closure_example_before.png",
        show=show,
    )
    draw(
        draw_registration_result(source, target, best.transformation),
        title=f"Step 2 - loop closure: after ICP (fitness={best.fitness:.2f})",
        screenshot_path=debug_dir / "loop_closure_example_after.png",
        show=show,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_step2(session_dir: Path, output_dir: Optional[Path] = None, show: bool = True) -> Path:
    session_dir = Path(session_dir)
    step1_dir = session_dir / "output" / "step1"
    if not (step1_dir / "keyframes.json").exists():
        raise FileNotFoundError(f"No {step1_dir / 'keyframes.json'} found; run Step 1 first.")
    output_dir = Path(output_dir) if output_dir else session_dir / "output" / "step2"
    debug_dir = ensure_dir(output_dir / "debug")

    keyframes = load_keyframes(step1_dir)
    print(f"[Step 2] Loaded {len(keyframes)} keyframes from {step1_dir}")
    frame_by_index = {f.index: f for f in load_lidar_frames(session_dir)}

    loop_closures = find_and_verify_loop_closures(keyframes, frame_by_index)

    graph = build_pose_graph(keyframes, frame_by_index, loop_closures)
    poses_step2 = optimize_pose_graph(graph)

    poses_raw = np.stack([kf.pose_initial for kf in keyframes])
    poses_step1 = np.stack([kf.pose_refined for kf in keyframes])

    save_keyframe_trajectory(keyframes, poses_step2, output_dir, "pose_graph")
    save_loop_closures_json(keyframes, loop_closures, poses_step2, output_dir / "loop_closures.json")

    step1_end_gap = float(np.linalg.norm(poses_step1[-1][:3, 3] - poses_raw[-1][:3, 3]))
    step2_end_gap = float(np.linalg.norm(poses_step2[-1][:3, 3] - poses_step1[-1][:3, 3]))
    print(f"[Step 2] End-pose shift, Step1-ICP vs. raw odometry: {step1_end_gap:.3f} m")
    print(f"[Step 2] End-pose shift, Step2-optimized vs. Step1-ICP: {step2_end_gap:.3f} m")

    print("[Step 2] Rendering visualizations ...")
    plot_map_with_trajectories(keyframes, frame_by_index, poses_raw, poses_step1, poses_step2, debug_dir, show)
    plot_pose_graph_2d(poses_step2, loop_closures, debug_dir / "pose_graph_2d.png")
    plot_loop_closure_example(keyframes, frame_by_index, loop_closures, debug_dir, False)

    print(f"[Step 2] Done. Output in {output_dir}")
    return output_dir


def _default_session_dir() -> Optional[Path]:
    captures_root = Path("captures")
    if not captures_root.exists():
        return None
    sessions = sorted(p for p in captures_root.iterdir() if p.is_dir() and p.name.startswith("session_"))
    return sessions[-1] if sessions else None


if __name__ == "__main__":
    # Edit these to point at a different session / tune Step 2, then re-run this file.
    SESSION_DIR = _default_session_dir()
    OUTPUT_DIR = None  # defaults to captures/<session>/output/step2
    SHOW = True

    if SESSION_DIR is None:
        raise SystemExit("No captures/session_* directory found; set SESSION_DIR explicitly.")

    run_step2(SESSION_DIR, OUTPUT_DIR, show=SHOW)
