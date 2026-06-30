"""
Handles the Reconstruction from the (preprocessed) data (Images and/or LiDAR, and Spots Odometry/IMU)
"""

class Reconstruction:

    def __init__(self):
        pass

    # Optional
    def data_preprocess(self):
        """
        Handles the preprocessesing stepfrom `../preprocessing/preprocessing.py` 
        if its not already done in the preprocess tab.
        """
        pass

    def lidar_icp(self):
        """ 
        **STEP 0:** Sensor Preprocessing (Lidar ICP)... \n
        *Output:* \\
        Synchronized dataset:
        - (RGB_t, LiDAR_t, Pose_t_initial) \n

        **STEP 1:** Odometry Refinement (Lidar ICP)... \n
        *Output:* \\
        Refined relative poses between keyframes
        """
        pass

    def pose_graph(self):
        """
        **STEP 2:** Pose Graph Construction and Optimization... \n
        *Output:* \\
        Globally consistent trajectory
        """
        pass

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
        pass

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