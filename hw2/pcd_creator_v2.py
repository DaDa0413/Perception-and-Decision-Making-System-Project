# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import open3d as o3d
import numpy as np
import cv2
import copy

# file_path = 'Data_collection/first_floor/'
# image_number = 189
# print(o3d.__version__)
# intrinsic matrix

def depth_image_to_point_cloud(rgb, depth, K):
    # Image plane
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])
    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)
    # Image plane to object plane
    depth_scale = 1000.0
    Z = depth.astype(float) / depth_scale
    X = (u - K[0, 2]) * Z / K[0, 0] # (u-cx) * Z / fx
    Y = (v - K[1, 2]) * Z / K[1, 1] # (v-cy) * Z / fy
    # Flatten and remove invalid point
    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)
    valid = Z  > 0
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    position = np.vstack((X, Y, Z,))

    R = np.ravel(rgb[:, :, 0])[valid]/255.
    G = np.ravel(rgb[:, :, 1])[valid]/255.
    B = np.ravel(rgb[:, :, 2])[valid]/255.
    points = np.transpose(position)
    colors = np.transpose(np.vstack((R, G, B)))
    return(points, colors)


def create_pcds(color_path, depth_path, last_image_number, pcd_storing_path):
    K = np.array([[256, 0, 255],
            [0, 256, 255],
            [0,   0,  1]])
    for index in range(0, last_image_number + 1):
        file_name = "{:04d}".format(index)
        color_raw = cv2.imread(color_path + file_name + ".jpg", cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(depth_path + file_name + ".png", cv2.IMREAD_UNCHANGED)
        points, colors = depth_image_to_point_cloud(color_raw, depth_raw, K)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud(pcd_storing_path + file_name + '.pcd', pcd) 


