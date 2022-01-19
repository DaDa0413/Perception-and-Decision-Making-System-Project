# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import open3d as o3d
import numpy as np
import copy

target_furniture = "refrigerator"
# target_furniture = "lamp"
# target_furniture = "cooktop"
# target_furniture = "cushion"
# target_furniture = "rack"

pcd_path = 'semantic_3d_pointcloud/'
voxel_size = 0.002
threshold = 0.002
obstacle_threshold = 0.2

# %%
# Load the npy to point cloud
points = np.load(pcd_path + "point.npy")
points = points * 10000. / 255
colors = np.load(pcd_path + "color01.npy")

# Crop the ceiling
valid = points[:,1] < -0.1
points = points[valid]
colors = colors[valid]

# Crop the floor
valid = points[:,1] > -1.2
points = points[valid]
colors = colors[valid]

reconstructed_pcd = o3d.geometry.PointCloud()
reconstructed_pcd.points = o3d.utility.Vector3dVector(points)
reconstructed_pcd.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([reconstructed_pcd])


# %%
print(reconstructed_pcd.get_min_bound())
print(reconstructed_pcd.get_max_bound())


# %%

# Draw the point cloud on scatterplot
import matplotlib.pyplot as plt 
x = points[:, 2]
y = points[:, 0]

plt.scatter(x,y,c=colors, marker=".")
x_init = (7.5, 3)  # starting location
def onclick(event):
    global x_init
    x_init = (event.xdata, event.ydata)
    print("x_init: {}".format(x_init))


plt.connect('button_press_event', onclick)
plt.savefig("map.png", bbox_inches='tight')
