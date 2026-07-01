"""Step 1 -- LiDAR Odometry Refinement (frame-to-frame multi-scale ICP).

Reads a raw capture session directly (no separate preprocessing stage) and
refines Spot's onboard odometry with coarse-to-fine point-to-plane ICP run on
**every consecutive scan**, seeded by the odometry prior and gated by a
quality check that falls back to the prior when a registration is untrustworthy.
Keyframes are then selected from the resulting dense, robust trajectory purely
to keep the reported output and fused map small -- keyframe spacing never
gates which scans get registered against each other.

Produces, under ``captures/<session>/output/step1/``:
- ``odometry_raw_poses.npy`` / ``odometry_refined_poses.npy`` -- (N, 4, 4) dense trajectories
- ``odometry_raw_tum.txt`` / ``odometry_refined_tum.txt`` -- TUM-format trajectories
- ``keyframes.json``   -- reduced pose set + relative transform between keyframes
- ``debug/*.png``      -- combined map+trajectory view, pairwise ICP examples, top-down plot

Runnable standalone: edit the variables at the bottom of this file and run
either ``python -m src.reconstruction.lidar_icp`` or
``python src/reconstruction/lidar_icp.py`` directly from the project root.
"""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    # Allow `python src/reconstruction/lidar_icp.py` (no -m) by putting the
    # project root on sys.path so the `src.*` absolute imports below resolve.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from src.reconstruction.common import (
    VOXEL_SIZES,
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
    rotation_angle_deg,
    trajectory_lineset,
)

# --- Quality gate: reject a registration and trust the odometry prior when it
# is weak or disagrees with the prior by too much. Without this, one bad
# frame-to-frame registration (e.g. a feature-poor corridor) gets chained into
# every pose after it. ---------------------------------------------------
MIN_FITNESS = 0.30
MAX_TRANS_DISAGREE_M = 0.50
MAX_ROT_DISAGREE_DEG = 20.0

# --- Keyframing is a post-hoc reduction of the (already good) dense
# trajectory, not a gate on which scans get registered. ---------------------
KEYFRAME_TRANSLATION_THRESH_M = 0.2
KEYFRAME_ROTATION_THRESH_DEG = 10.0
MAP_VOXEL_SIZE = 0.05


@dataclass
class PairResult:
    """Diagnostics for one frame-to-frame registration (scan ``index`` onto ``index - 1``)."""

    index: int
    fitness: float
    inlier_rmse: float
    prior: np.ndarray
    relative: np.ndarray
    rejected: bool


def run_lidar_odometry(
    session_dir: Path,
) -> tuple[list[LidarFrame], np.ndarray, np.ndarray, list[PairResult]]:
    """Dense frame-to-frame multi-scale ICP, seeded by Spot's odometry prior.

    Returns (frames, poses_refined, poses_raw, pair_results); both pose
    arrays are in the odom frame so ICP and raw odometry are directly
    comparable.
    """
    frames = load_lidar_frames(session_dir)
    n = len(frames)
    if n < 2:
        raise ValueError(f"Need >= 2 LiDAR scans, found {n} in {session_dir}")
    print(f"[Step 1] Loaded {n} LiDAR scans from {session_dir}")

    poses_raw = np.stack([f.odom_T_sensor for f in frames])
    poses_refined = [poses_raw[0].copy()]
    pair_results: list[PairResult] = []

    target_pyramid = pyramid(frames[0].read_cloud_sensor())
    rejected = 0
    for i in range(1, n):
        source_pyramid = pyramid(frames[i].read_cloud_sensor())

        prior = np.linalg.inv(poses_raw[i - 1]) @ poses_raw[i]
        relative, result = multiscale_icp(source_pyramid, target_pyramid, prior)

        trans_disagree, rot_disagree = pose_disagreement(relative, prior)
        is_rejected = (
            result.fitness < MIN_FITNESS
            or trans_disagree > MAX_TRANS_DISAGREE_M
            or rot_disagree > MAX_ROT_DISAGREE_DEG
        )
        if is_rejected:
            relative = prior
            rejected += 1

        pair_results.append(
            PairResult(i, result.fitness, result.inlier_rmse, prior, relative, is_rejected)
        )
        poses_refined.append(poses_refined[-1] @ relative)
        target_pyramid = source_pyramid

        flag = "  REJECTED -> odometry prior" if is_rejected else ""
        print(
            f"[Step 1]   [{i:04d}/{n - 1}] fitness={result.fitness:.3f} "
            f"rmse={result.inlier_rmse:.4f} (vs prior: {trans_disagree:.3f}m/{rot_disagree:.1f}deg){flag}"
        )

    print(f"[Step 1] Done. {rejected}/{n - 1} pairs fell back to the odometry prior.")
    return frames, np.stack(poses_refined), poses_raw, pair_results


def select_keyframes(
    poses: np.ndarray,
    translation_thresh_m: float = KEYFRAME_TRANSLATION_THRESH_M,
    rotation_thresh_deg: float = KEYFRAME_ROTATION_THRESH_DEG,
) -> list[int]:
    """Indices of ``poses`` to keep, gated by translation/rotation since the last keyframe.

    Purely a density reduction of an already-refined trajectory -- unrelated
    to which pairs were registered against each other in `run_lidar_odometry`.
    """
    keyframes = [0]
    last_pose = poses[0]
    for i in range(1, len(poses)):
        delta = np.linalg.inv(last_pose) @ poses[i]
        translation_m = float(np.linalg.norm(delta[:3, 3]))
        if translation_m >= translation_thresh_m or rotation_angle_deg(delta) >= rotation_thresh_deg:
            keyframes.append(i)
            last_pose = poses[i]
    if keyframes[-1] != len(poses) - 1:
        keyframes.append(len(poses) - 1)
    return keyframes


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_trajectory(frames: list[LidarFrame], poses: np.ndarray, output_dir: Path, prefix: str) -> None:
    """Save poses as 4x4 matrices (.npy) and in TUM format with real timestamps."""
    np.save(output_dir / f"{prefix}_poses.npy", poses)
    times = np.array([f.t_nsec for f in frames]) * 1e-9
    with open(output_dir / f"{prefix}_tum.txt", "w") as handle:
        for t, pose in zip(times, poses):
            tx, ty, tz = pose[:3, 3]
            qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
            handle.write(f"{t:.9f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


def save_keyframes_json(
    frames: list[LidarFrame],
    poses_raw: np.ndarray,
    poses_refined: np.ndarray,
    keyframe_indices: list[int],
    path: Path,
) -> None:
    entries = []
    prev_idx = None
    for idx in keyframe_indices:
        relative_to_prev = (
            (np.linalg.inv(poses_refined[prev_idx]) @ poses_refined[idx]).tolist()
            if prev_idx is not None
            else None
        )
        entries.append(
            {
                "scan_index": frames[idx].index,
                "t_nsec": frames[idx].t_nsec,
                "pose_initial": poses_raw[idx].tolist(),
                "pose_refined": poses_refined[idx].tolist(),
                "relative_to_prev_keyframe": relative_to_prev,
            }
        )
        prev_idx = idx
    path.write_text(json.dumps(entries, indent=2))


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def plot_map_with_trajectory(
    frames: list[LidarFrame], poses_raw: np.ndarray, poses_refined: np.ndarray, debug_dir: Path, show: bool
) -> None:
    """Fused map (refined poses) with both trajectories drawn on top, viewed from directly above.

    An oblique 3/4 angle looks nicer for showing wall height, but from that
    angle a corridor's walls occlude a trajectory running through its open
    middle. Top-down avoids that, and cropping the map to a height slab
    around the trajectory (see ``common.crop_to_trajectory``) removes the
    ceiling that would otherwise occlude it from directly above -- so the
    path is drawn at its real, unmodified height throughout.
    """
    fused_map = build_map(frames, poses_refined, voxel_size=MAP_VOXEL_SIZE, every=2)
    fused_map = crop_to_trajectory(fused_map, poses_refined)
    fused_map.paint_uniform_color([0.75, 0.75, 0.75])  # neutral gray so the trajectories stand out
    geometries = [
        fused_map,
        trajectory_lineset(poses_raw, color=(0.0, 0.4, 1.0)),
        trajectory_lineset(poses_refined, color=(1.0, 0.5, 0.0)),
        coordinate_frame(size=0.5),
    ]
    draw(
        geometries,
        title="Step 1 - fused map + trajectory (blue=raw odometry, orange=ICP-refined)",
        screenshot_path=debug_dir / "map_and_trajectory.png",
        show=show,
        point_size=1.5,
        top_down=True,
    )


def draw_registration_result(
    source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, transformation: np.ndarray
) -> list:
    """Classic Open3D ICP-tutorial look: yellow source aligned onto cyan target."""
    source_vis = copy.deepcopy(source)
    target_vis = copy.deepcopy(target)
    source_vis.paint_uniform_color([1.0, 0.706, 0.0])
    target_vis.paint_uniform_color([0.0, 0.651, 0.929])
    source_vis.transform(transformation)
    return [source_vis, target_vis, coordinate_frame(size=0.3)]


def _correction_size(pair: PairResult) -> float:
    """How much ICP moved the result away from the odometry prior (m, normalized with deg/90)."""
    translation_m, rotation_deg = pose_disagreement(pair.relative, pair.prior)
    return translation_m + rotation_deg / 90.0


def plot_pairwise_examples(
    frames: list[LidarFrame], pair_results: list[PairResult], debug_dir: Path, show: bool
) -> None:
    """Before/after alignment for the worst-fitness and most-corrected pair."""
    non_rejected = [p for p in pair_results if not p.rejected] or pair_results
    worst_fitness = min(non_rejected, key=lambda p: p.fitness)
    largest_correction = max(pair_results, key=_correction_size)

    picks = {"worst_fitness": worst_fitness, "largest_correction": largest_correction}
    for tag, pair in picks.items():
        source = frames[pair.index].read_cloud_sensor().voxel_down_sample(VOXEL_SIZES[-1])
        target = frames[pair.index - 1].read_cloud_sensor().voxel_down_sample(VOXEL_SIZES[-1])
        draw(
            draw_registration_result(source, target, pair.prior),
            title=f"Step 1 - {tag}: before ICP (odometry prior)",
            screenshot_path=debug_dir / f"pairwise_{tag}_before.png",
            show=show,
        )
        draw(
            draw_registration_result(source, target, pair.relative),
            title=f"Step 1 - {tag}: after ICP (fitness={pair.fitness:.2f})",
            screenshot_path=debug_dir / f"pairwise_{tag}_after.png",
            show=show,
        )


def plot_trajectory_topdown(poses_raw: np.ndarray, poses_refined: np.ndarray, path: Path) -> None:
    raw_xy = poses_raw[:, :2, 3]
    refined_xy = poses_refined[:, :2, 3]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(raw_xy[:, 0], raw_xy[:, 1], "-", color="tab:blue", linewidth=1.5, label="raw odometry")
    ax.plot(refined_xy[:, 0], refined_xy[:, 1], "-", color="tab:orange", linewidth=1.5, label="ICP-refined")
    ax.scatter(*raw_xy[0], color="green", zorder=5, label="start")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Step 1 - trajectory: raw odometry vs. ICP-refined")
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_step1(session_dir: Path, output_dir: Optional[Path] = None, show: bool = True) -> Path:
    session_dir = Path(session_dir)
    output_dir = Path(output_dir) if output_dir else session_dir / "output" / "step1"
    debug_dir = ensure_dir(output_dir / "debug")

    frames, poses_refined, poses_raw, pair_results = run_lidar_odometry(session_dir)

    save_trajectory(frames, poses_raw, output_dir, "odometry_raw")
    save_trajectory(frames, poses_refined, output_dir, "odometry_refined")

    keyframe_indices = select_keyframes(poses_refined)
    print(f"[Step 1] {len(keyframe_indices)}/{len(frames)} scans selected as keyframes")
    save_keyframes_json(frames, poses_raw, poses_refined, keyframe_indices, output_dir / "keyframes.json")

    end_gap_m = float(np.linalg.norm(poses_refined[-1][:3, 3] - poses_raw[-1][:3, 3]))
    print(f"[Step 1] End-pose gap, ICP vs. raw odometry: {end_gap_m:.3f} m")

    print("[Step 1] Rendering visualizations ...")
    plot_map_with_trajectory(frames, poses_raw, poses_refined, debug_dir, show)
    plot_pairwise_examples(frames, pair_results, debug_dir, False)
    plot_trajectory_topdown(poses_raw, poses_refined, debug_dir / "trajectory_topdown.png")

    print(f"[Step 1] Done. Output in {output_dir}")
    return output_dir


def _default_session_dir() -> Optional[Path]:
    captures_root = Path("captures")
    if not captures_root.exists():
        return None
    sessions = sorted(p for p in captures_root.iterdir() if p.is_dir() and p.name.startswith("session_"))
    return sessions[-1] if sessions else None


if __name__ == "__main__":
    # Edit these to point at a different session / tune Step 1, then re-run this file.
    SESSION_DIR = _default_session_dir()
    OUTPUT_DIR = None  # defaults to captures/<session>/output/step1
    SHOW = True

    if SESSION_DIR is None:
        raise SystemExit("No captures/session_* directory found; set SESSION_DIR explicitly.")

    run_step1(SESSION_DIR, OUTPUT_DIR, show=SHOW)
