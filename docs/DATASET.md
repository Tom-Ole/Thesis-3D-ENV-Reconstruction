# Capture dataset format (Step 0 readiness)

This document describes what the capture tab records and how it maps to the
**Step 0 — Data Preparation** requirements. The capture app records the raw data
and metadata; the Step 0 algorithms (calibration, undistortion, deskewing,
interpolation) run downstream on this dataset.

## On-disk layout

```
captures/
  session_01_YYYYMMDD/
    session.json                 # calibration manifest (per session, see below)
    images/
      0001_frontleft_fisheye_image.jpg
      0002_frontright_fisheye_image.jpg
      ...
    pointclouds/
      0001_velodyne-point-cloud.ply   # binary little-endian PLY, XYZ float32
      ...
    metadata/
      0001_frontleft_fisheye_image.json
      0001_velodyne-point-cloud.json
      ...
    imu/
      state_log.jsonl            # continuous IMU / odometry stream (one JSON per line)
```

A *session* is one recording run (Start → Stop). Capture is **continuous**: while
you walk Spot, images and point clouds are recorded at independent configurable
rates (`config.image_sample_rate`, `config.lidar_sample_rate`), and the IMU /
odometry stream is logged the whole time.

## How each Step 0 requirement is supported

### 1. Time synchronisation
- **Single common clock.** Spot stamps every sensor reading (image, point cloud,
  IMU/state) in the **robot clock**. Each asset's metadata carries
  `acquisition_time_robot_nsec` (robot-clock nanoseconds) — interpolate on this
  field directly. Because all sensors share one clock, sync is fundamentally an
  interpolation problem, not a clock-alignment problem.
- **Client-clock mapping / constant offset.** `clock_skew_sec` (robot − client)
  is recorded per asset and in `session.json`. Subtract it to map robot
  timestamps to wall/client time, or to estimate/correct a constant offset
  against any external reference.
- **Interpolation over nearest-neighbour.** The `imu/state_log.jsonl` stream is
  recorded continuously at a higher rate than the cameras/LiDAR, so every image
  and scan timestamp is bracketed by state samples — interpolate pose/IMU to the
  exact sensor timestamp rather than snapping to the nearest sample.

### 2. Camera calibration (intrinsics) + undistortion
- Each image's metadata and `session.json` record the robot-reported
  `camera_model`: type (`pinhole` or `pinhole_brown_conrady`), `intrinsics`
  (focal length, principal point, skew) and, when available, Brown-Conrady
  `distortion` (k1,k2,k3,p1,p2). This is the nominal/factory calibration to seed
  or validate your own OpenCV calibration.
- **Images are saved RAW (unrotated) by default** (`apply_camera_rotation=False`)
  so pixels match the reported intrinsics. Rotation is display-only; if ever
  enabled, `rotation_applied` is recorded so it can be undone. **Keep rotation
  off for calibration.**

### 3. Extrinsic calibration (LiDAR↔IMU, LiDAR↔camera)
- Every image and point cloud stores its `transforms_snapshot` (the frame tree at
  capture time) and its sensor frame name (`frame_name_image_sensor` /
  `frame_name_sensor`). `session.json` also stores a `frame_tree_snapshot`.
- Compose transforms through the common `body` / `odom` frame to obtain
  LiDAR↔camera and LiDAR↔body(IMU) rigid transforms. These are the nominal
  extrinsics to validate (via LiDAR→image projection) and refine.

### 4. LiDAR motion-distortion correction (deskewing)
- The continuous `imu/state_log.jsonl` provides the high-rate pose/IMU needed to
  interpolate sensor motion across a scan period.
- **Limitation (honest):** Spot's PointCloud API exposes **one acquisition
  timestamp per scan**, not per-point timestamps. Deskew by interpolating pose
  across the scan period assuming a uniform angular sweep (Velodyne ~10 Hz ⇒
  ~100 ms/revolution). This is documented in `session.json` (`deskew_note`).

## IMU source (important)

The standard `get_robot_state` RPC does **not** expose raw IMU packets; it gives
the fused `kinematic_state` (odometry pose in `odom`/`vision` + body velocities),
which is the practical, reliable signal for pose interpolation and deskewing.

True high-rate raw IMU comes from the optional `robot-state-streaming` service.
When present, it is used automatically (`config.prefer_state_streaming=True`) and
each line of `state_log.jsonl` is the full streamed message; otherwise the app
falls back to polling at `config.state_sample_rate` Hz. `session.json.imu_source`
records which path was used.

## Notes / assumptions
- "LiDAR" = the EAP payload's Velodyne via `PointCloudClient`. If the payload is
  absent, image + IMU capture still work and `session.json` notes no LiDAR.
- Point clouds are saved as binary PLY (XYZ float32) — widely readable by Open3D,
  CloudCompare, COLMAP. Intensity is not provided by the `XYZ_32F` encoding.
- Pure data capture needs auth + time-sync only; **no lease/E-Stop** is acquired.
