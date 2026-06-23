# STEP 0 - Data Preparation (Critical Foundation)

## Inputs:
- Velodyne LiDAR point clouds (timestamped)
- IMU data from Spot API (timestamped)
- 5 camera streams (timestamped)

## Required preprocessing:

### 1. Time synchronization
- Synchronize LiDAR, IMU, and cameras
- Prefer interpolation over nearest-neighbor matching
- Estimate and correct constant time offsets between sensors if present

### 2. Camera calibration
- Intrinsics calibration for each of the 5 cameras (OpenCV)
- Undistort all images before processing

### 3. Extrinsic calibration
- Compute rigid transforms:
  - LiDAR → IMU
  - LiDAR → each camera
- Validate via projection consistency (LiDAR points projected into images should align with edges)

### 4. LiDAR motion distortion correction (deskewing)
- Use IMU to correct motion during LiDAR scan acquisition
- This is essential for Velodyne accuracy indoors

## Output:
- Fully synchronized and calibrated dataset:
  - LiDAR scans (deskewed)
  - IMU measurements
  - 5 undistorted, synchronized camera images
  - Known extrinsics for all sensors

---

# STEP 1 - LiDAR Odometry

Best option:
- Open3D ICP-based SLAM (Python)

Output:
- trajectory estimates T_i
- aligned point clouds

---

# STEP 2 - Keyframe Selection and Global Point Cloud Map

## Keyframe strategy:
Select LiDAR frames based on:
- Distance traveled threshold
- Rotation change threshold
- Overlap reduction criteria

## Mapping process:
For each keyframe:
- Transform point cloud into global frame:
  P_global = T_i * P_lidar_i

## Fusion:
- Voxel grid downsampling (2–5 cm recommended for indoor scenes)
- Statistical outlier removal
- Radius-based filtering

## Output:
- Coarse but globally consistent point cloud map
- Reduced redundancy compared to raw scans

---

# STEP 3 - Pose Graph Optimization (Global Consistency)

## Graph structure:

### Nodes:
- Keyframe poses from FAST-LIO2

### Edges:
1. Sequential constraints (temporal continuity)
2. Loop closure constraints (revisiting places)
3. Structural constraints (indoor geometry priors)

## Loop closure detection:
- Scan Context descriptors (recommended)
- Global LiDAR feature matching
- ICP verification of candidate loop closures

## Optional but highly effective:
- Manhattan-world constraints (indoor orthogonality assumptions)

## Optimization:
- g2o or Ceres Solver
- Or Open3D pose graph optimization

## Output:
- Globally consistent trajectory
- Reduced drift over long indoor sequences

---

# STEP 4 - Surfel-Based Surface Reconstruction

## Representation:
Each surfel stores:
- Position
- Normal
- Radius
- Confidence score
- Color (from cameras)

## Update process:
1. Associate new LiDAR points with existing surfels
2. Update position via weighted averaging
3. Estimate normals using local PCA
4. Update confidence over time
5. Remove or decay unstable surfels

## Important requirement:
- Surfel lifecycle management:
  - confidence increases with repeated observations
  - unstable surfels decay or are removed

## Output:
- Dense, smooth, and stable surface representation
- Better indoor detail than voxel-only TSDF methods

---

# STEP 5 - Multi-Camera Color Fusion

## For each surfel:

1. Project surfel into all 5 camera views
2. Check visibility:
   - Occlusion filtering
   - Viewing angle constraints
   - Distance thresholds
   - Image quality (blur detection)

3. Compute color using weighted blending:
   - Higher weight for:
     - frontal viewing angles
     - closer distances
     - sharper images

## Important design principle:
- Do NOT assign color from a single “best” camera
- Maintain a running multi-view color estimate per surfel

## Output:
- Stable and consistent colored 3D reconstruction
- Reduced texture seams across views

---

# STEP 6 - Neural Refinement (3D Gaussian Splatting)

## Input requirements:
- Optimized poses from pose graph (Step 3)
- Clean surfel or point cloud representation (Step 4–5)
- Synchronized multi-view images

## Process:
- Fit Gaussian primitives to reconstructed geometry
- Optimize radiance field using image reconstruction loss
- Use LiDAR geometry as structural prior to prevent collapse

## Benefits:
- Handles low-parallax robot trajectories
- Recovers fine geometric and visual detail
- Works even when SfM methods (e.g., COLMAP) fail
- Produces high-quality photorealistic reconstructions

## Output:
- High-density Gaussian representation of the scene
- Optional conversion to mesh or point cloud for downstream use