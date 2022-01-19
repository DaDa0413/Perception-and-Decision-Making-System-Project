# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import open3d as o3d
import numpy as np
import copy
from pcd_creator_v2 import *

first_image = 0
last_image_number = 66
file_path = 'Data_collection/apartment0_test_data/'
color_folder_name = 'color/'
pcd_folder_name = 'color_pcd/'
# Ground truth
semantic_folder_name = 'semantic/'
semantic_pcd_folder_name = 'semantic_pcd/'
# # Predicted same distrobution
# semantic_folder_name = 'same_distro_predicted_semantic/'
# semantic_pcd_folder_name = 'same_distro_predicted_semantic_pcd/'
# # Predicted different distrobution
# semantic_folder_name = 'diff_distro_predicted_semantic/'
# semantic_pcd_folder_name = 'diff_distro_predicted_semantic_pcd/'

print(o3d.__version__)

voxel_size = 0.002
threshold = 0.002


# %%
# Create point cloud for reconstruction
create_pcds(file_path + color_folder_name, file_path + 'depth/', last_image_number, file_path + pcd_folder_name)


# %%
def preprocess_point_cloud(pcd, voxel_size):
#     print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
#     print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
#     print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


# %%
def prepare_dataset(image, voxel_size):
    # Create target point cloud
    file_name = '{:04d}'.format(image)
    pcd = o3d.io.read_point_cloud(file_path + pcd_folder_name + file_name + ".pcd")

#     print(":: Load target point cloud and disturb initial pose.")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
#     source.transform(trans_init)

    # Extract fpfh
    pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size)
    return pcd, pcd_down, pcd_fpfh


# %%
# Global ICP
# RANSAC registration
def execute_global_registration(source, target, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
#     print(":: RANSAC registration on downsampled point clouds.")
#     print("   Since the downsampling voxel size is %.3f," % voxel_size)
#     print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


# %%
# Create target point cloud

target, target_down, target_fpfh = prepare_dataset(first_image, voxel_size)
aggr_pcd = target


# %%
# Transform all the pic
transformations = []
transformations.append(np.identity(4))
trajectories = []
trajectories.append([0,0,0,1])

for index in range(first_image + 1, last_image_number + 1):
    print("Pic: {0:04d}".format(index))
    # Load point cloud and preprocessing
    source, source_down, source_fpfh = prepare_dataset(index, voxel_size)
    
    # Global registration
    result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
    trans_init = result_ransac.transformation
    # Local registration
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # Update the transformations list and transform the current pcd to G_0
    t = np.dot(transformations[-1], reg_p2p.transformation)
    transformations.append(t)
    # transofrm the source to G_0
    s = copy.deepcopy(source)
    s.transform(t)
    # Trajectory
    transition = np.append(reg_p2p.transformation[0:3,3], 1)
    trajectories.append(np.dot(t, transition))
    # Aggregate the result
    aggr_pcd += s
    
    target = source
    target_down = source_down
    target_fpfh = source_fpfh


# %%
# Flip it, otherwise the pointcloud will be upside down
aggr_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
aggr_pcd_down = aggr_pcd.voxel_down_sample(voxel_size / 5)


# %%
# Remove the roof
no_roof_pcd = copy.deepcopy(aggr_pcd_down)
pts = np.asarray(no_roof_pcd.points)
crs = np.asarray(no_roof_pcd.colors)
valid = pts[:,1] < 0

no_roof_pcd.points = o3d.utility.Vector3dVector(pts[valid])
no_roof_pcd.colors = o3d.utility.Vector3dVector(crs[valid])
# o3d.visualization.draw_geometries([no_roof_pcd])


# %%
# Create semantic point cloud
create_pcds(file_path + semantic_folder_name, file_path + 'depth/', last_image_number, file_path + semantic_pcd_folder_name)
semantic_pcd = o3d.geometry.PointCloud()

for index in range(last_image_number + 1):
    file_name = '{:04d}'.format(index)
    tmp_pcd = o3d.io.read_point_cloud(file_path + semantic_pcd_folder_name + file_name + ".pcd")
    tmp_pcd.transform(transformations[index])
    semantic_pcd += tmp_pcd
# o3d.visualization.draw_geometries([semantic_pcd])


# %%
# Remove the roof
no_roof_pcd = copy.deepcopy(semantic_pcd)
no_roof_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# no_roof_pcd = copy.deepcopy(my_voxel_down_pcd)
pts = np.asarray(no_roof_pcd.points)    
crs = np.asarray(no_roof_pcd.colors)
valid = pts[:,1] < 0

no_roof_pcd.points = o3d.utility.Vector3dVector(pts[valid])
no_roof_pcd.colors = o3d.utility.Vector3dVector(crs[valid])
# o3d.visualization.draw_geometries([no_roof_pcd])


# %%
from scipy import stats
def voxel_filter(pts, scores, grid_size, use_3d=False, min_num_pts=4):
    mins = pts.min(axis=0) - grid_size
    maxs = pts.max(axis=0) + grid_size
    bins = [np.arange(mins[i], maxs[i], grid_size) for i in range(len(mins))]   # Create bins array of each dimension

    si = 2
    if use_3d:
        si = 3

    counts, edges, binnumbers = stats.binned_statistic_dd(  # counts of each bin, edges of the bins, data-bin mapping
        pts[:, :si],
        values=None,
        statistic="count",
        bins=bins[:si],
        range=None,
        expand_binnumbers=False
    )

    ub = np.unique(binnumbers)
    pts_ds = []
    scores_ds = []
    for b in ub:
        if len(np.where(binnumbers == b)[0]) >= min_num_pts:
            pts_ds.append(pts[np.where(binnumbers == b)[0]].mean(axis=0))
            u, c = np.unique(scores[np.where(binnumbers == b)[0]], return_counts=True, axis=0)
            scores_ds.append(u[np.argmax(c)])

    pts_ds = np.vstack(pts_ds)
    scores_ds = np.vstack(scores_ds)
    return pts_ds, scores_ds


# %%
pts = np.asarray(semantic_pcd.points)
crs = np.asarray(semantic_pcd.colors)

pts_ds, crs_ds =voxel_filter(pts, crs, voxel_size, True, 1)


# %%
my_voxel_down_pcd = o3d.geometry.PointCloud()
my_voxel_down_pcd.points = o3d.utility.Vector3dVector(pts_ds)
my_voxel_down_pcd.colors = o3d.utility.Vector3dVector(crs_ds)
o3d.visualization.draw_geometries([my_voxel_down_pcd])


# %%
# Remove the roof
no_roof_pcd = copy.deepcopy(my_voxel_down_pcd)
no_roof_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pts = np.asarray(no_roof_pcd.points)    
crs = np.asarray(no_roof_pcd.colors)
valid = pts[:,1] < 0

no_roof_pcd.points = o3d.utility.Vector3dVector(pts[valid])
no_roof_pcd.colors = o3d.utility.Vector3dVector(crs[valid])
o3d.visualization.draw_geometries([no_roof_pcd])


