# HW1 Habitat 3D Scene Reconstruction
* Before executing, you must download apartment_0 from Habitat-sim.
## Structure
* data_collector
Take RGB and depth pictures from habitat-sim
* pcd_creator.ipynb & pcd_creator-v2.ipynb
Convert RGB and depth images to point cloud files.
* registration.py
SVD local registration (ICP)
* hw1.ipynb
Reconstruct with open3D global and local registration.
* hw1_myICP.ipynb
Reconstruct with open3D global registration and self-made ICP function.

