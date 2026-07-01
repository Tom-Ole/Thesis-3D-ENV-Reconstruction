"""
Handles the Reconstruction from the (preprocessed) data (Images and/or LiDAR, and Spots Odometry/IMU)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


class Reconstruction:

    def __init__(self, session_dir: Path, output_dir: Optional[Path] = None, show: bool = True):
        self.session_dir = Path(session_dir)
        self.output_dir = Path(output_dir) if output_dir else self.session_dir / "output"
        self.show = show
        self.step1_dir: Optional[Path] = None
        self.step2_dir: Optional[Path] = None
        self.step3_dir: Optional[Path] = None

    # Optional
    def data_preprocess(self):
        """
        **STEP 0:** Sensor Preprocessing (Calibration + Synchronization)

        Not a separate stage here: Step 1 reads the raw capture session
        directly (see src/reconstruction/common.py) and resolves each scan's
        pose straight from its own capture metadata, so there is nothing to
        precompute. Kept as a no-op for interface compatibility with the
        original step list.
        """
        pass

    def lidar_icp(self):
        """
        **STEP 1:** Odometry Refinement (Lidar ICP)... \n
        *Output:* \\
        Refined relative poses between keyframes
        """
        from src.reconstruction.lidar_icp import run_step1

        self.step1_dir = run_step1(self.session_dir, self.output_dir / "step1", show=self.show)
        return self.step1_dir

    def pose_graph(self):
        """
        **STEP 2:** Pose Graph Construction and Optimization... \n
        *Output:* \\
        Globally consistent trajectory
        """
        from src.reconstruction.pose_graph import run_step2

        self.step2_dir = run_step2(self.session_dir, self.output_dir / "step2", show=self.show)
        return self.step2_dir

    def fusion_mapping(self):
        """
        **STEP 3:** Multi-Sensor Fusion Mapping... \n
        **STEP 3.1:** Dense Point Cloud Fusion... 
        *Output:* \\
        Clean global point cloud \n
        **STEP 3.2:** TSDF Volume Integration (Coarse Geometry)... 
        *Output:* \\
        Watertight coarse geometry representation \n
        **STEP 3.2:** TSDF Volume Integration (Coarse Geometry)... 
        *Output:* \\
        Watertight coarse geometry representation \n
        **STEP 3.3:** Surface Normals and Confidence Estimation...
        *Output:* \\
        Normal + confidence field for surface reliability
        """
        from src.reconstruction.fusion import run_step3

        self.step3_dir = run_step3(self.session_dir, self.output_dir / "step3", show=self.show)
        return self.step3_dir

    def structure_decomposition(self):
        """
        **STEP 4:** Scene Structure Decomposition... \n
        **STEP 4.1:** Plane Extraction... \n
        **STEP 4.2:** Object Clustering... \n
        *Output:* \\
        Scene graph:
        - Static structure (planes)
        - Dynamic/object-level clusters
        """
        pass

    def geometry_refinement(self):
        """
        **STEP 5:** Geometry Refinement and Sharp Surface Reconstruction... \n
        **STEP 5.1:** Plane-Constrained TSDF Refinement... \n
        **STEP 5.2:** Edge-Preserving Surface Extraction... \n
        **STEP 5.3:** Object-Level Reconstruction... \n
        *Output:* \\
        Separated, high-fidelity object meshes
        """
        pass

    # Optional
    def gs_refinement(self):
        """
        If Gaussian Splatting is used:

        Recommended strategy:
        - Render depth maps from Gaussian splats
        - Fuse rendered depth into TSDF volume
        - Reconstruct mesh from TSDF instead of direct splat meshing

        Benefits:
        - Reduced noise
        - Improved geometric stability
        - Better edge consistency than direct splat-to-mesh conversion
        """
        pass

    def mesh_post_processing(self):
        """
        **STEP 7:** Final Mesh Post-Processing... \n
        **STEP 7.1:** Mesh Cleanup... \n
        **STEP 7.2:** Edge Enhancement... \n
        **STEP 7.3:** Bilateral Mesh Filtering... \n
        **STEP 7.4:** (Optional) Remeshing... \n
        *Final Output:* 
        1. Global Architectural Mesh
          - Walls, floors, ceilings
        2. Object-Level Meshes
          - Furniture
          - Equipment (e.g., PC cases)
          - Small objects
        3. Semantic Scene Representation
          - Structured environment graph
          - Object + structure separation
        """