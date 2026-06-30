# Hybrid Indoor 3D Reconstruction Pipeline (Spot Robot + LiDAR + RGB + Odometry)

This pipeline is designed for robust indoor 3D reconstruction using multi-sensor robot data. It combines geometric fusion, probabilistic mapping, and structure-aware mesh refinement to produce sharp, object-preserving meshes.

---

# 0. Sensor Preprocessing (Calibration + Synchronization)

## Inputs:
- RGB cameras (Spot)
- Velodyne LiDAR
- Robot odometry / IMU

## Steps:
- Time synchronization of all sensor streams
- Camera intrinsic calibration
- Extrinsic calibration:
  - LiDAR ↔ robot base frame
  - Cameras ↔ robot base frame
- LiDAR motion compensation (deskewing during movement)
- Image undistortion

## Output:
Synchronized dataset:
- (RGB_t, LiDAR_t, Pose_t_initial)

---

# 1. Odometry Refinement (LiDAR ICP)

## Goal:
Improve raw odometry using scan matching

## Steps:
- Initialize poses from robot odometry
- Apply frame-to-frame LiDAR ICP:
  - Prefer point-to-plane ICP
- Downsample scans (voxel grid filtering)
- Keyframe selection based on:
  - translation threshold
  - rotation threshold

## Output:
Refined relative poses between keyframes

---

# 2. Pose Graph Construction and Optimization

## Goal:
Global trajectory consistency

## Graph Structure:
- Nodes: keyframes
- Edges:
  - Odometry constraints
  - ICP constraints
  - Loop closure constraints (place recognition)

## Optimization:
- Non-linear optimization (g2o / Ceres Solver)
- Robust loss function (Huber / Cauchy kernel)

## Output:
Globally consistent trajectory

---

# 3. Multi-Sensor Fusion Mapping

## 3.1 Dense Point Cloud Fusion
- Transform all LiDAR scans into global frame
- Merge into global point cloud
- Apply:
  - voxel grid downsampling
  - statistical outlier removal

## Output:
Clean global point cloud

---

## 3.2 TSDF Volume Integration (Coarse Geometry)
- Convert LiDAR / depth into depth maps
- Fuse into TSDF volume

### Recommended Parameters:
- voxel size: 0.02 – 0.05 m (indoor)
- truncation distance: 2–3 voxels
- weighted integration based on viewing angle + distance

## Output:
Watertight coarse geometry representation

---

## 3.3 Surface Normals and Confidence Estimation
- Compute normals via PCA or depth gradients
- Estimate confidence per observation:
  - viewing angle
  - sensor distance
  - multi-view consistency

## Output:
Normal + confidence field for surface reliability

---

# 4. Scene Structure Decomposition

## Goal:
Separate structure (walls) from objects (furniture, equipment)

## Steps:

### 4.1 Plane Extraction
- RANSAC plane detection:
  - floor
  - ceiling
  - walls

### 4.2 Object Clustering
- Euclidean clustering on remaining points
- Identify:
  - chairs
  - tables
  - PC cases
  - small objects

## Output:
Scene graph:
- Static structure (planes)
- Dynamic/object-level clusters

---

# 5. Geometry Refinement and Sharp Surface Reconstruction

## 5.1 Plane-Constrained TSDF Refinement
- Snap TSDF surfaces to detected planes
- Enforce flatness constraints for:
  - walls
  - floors
  - ceilings

---

## 5.2 Edge-Preserving Surface Extraction
- Apply bilateral filtering to TSDF
- Preserve discontinuities during smoothing
- Extract mesh via Marching Cubes (refined TSDF)

---

## 5.3 Object-Level Reconstruction
For each clustered object:
- Reconstruct independently using:
  - TSDF OR Poisson OR local Gaussian reconstruction
- Prevent merging of nearby objects

## Output:
Separated, high-fidelity object meshes

---

# 6. Gaussian Splatting Refinement Layer (Optional)

If Gaussian Splatting is used:

## Recommended strategy:
- Render depth maps from Gaussian splats
- Fuse rendered depth into TSDF volume
- Reconstruct mesh from TSDF instead of direct splat meshing

## Benefits:
- Reduced noise
- Improved geometric stability
- Better edge consistency than direct splat-to-mesh conversion

---

# 7. Final Mesh Post-Processing

## Steps:

### 7.1 Mesh Cleanup
- Remove small disconnected components
- Keep largest consistent surfaces

### 7.2 Edge Enhancement
- Curvature-based edge detection
- Vertex adjustment near high curvature regions

### 7.3 Bilateral Mesh Filtering
- Smooth surfaces while preserving edges

### 7.4 Optional Remeshing
- Improve triangle quality and uniformity

---

# Final Outputs

## 1. Global Architectural Mesh
- Walls, floors, ceilings

## 2. Object-Level Meshes
- Furniture
- Equipment (e.g., PC cases)
- Small objects

## 3. Semantic Scene Representation
- Structured environment graph
- Object + structure separation

---

# Key Insight

High-quality indoor reconstruction is not achieved by a single algorithm, but by a layered hybrid pipeline:

- Geometric fusion (TSDF / point cloud)
- Structural decomposition (planes + objects)
- Edge-preserving refinement (bilateral + constraints)
- Object-level reconstruction (separate processing per cluster)