# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import open3d as o3d
import numpy as np
import copy

# target_furniture = "refrigerator"
# target_furniture = "lamp"
# target_furniture = "cooktop"
# target_furniture = "cushion"
target_furniture = "rack"

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
plt.show()


# %%
# Majority voxel size down
from scipy import stats
def voxel_filter(pts, scores, grid_size, target_crs, min_num_pts=4):
    mins = pts.min(axis=0) - grid_size
    maxs = pts.max(axis=0) + grid_size
    bins = []
    bins.append(np.arange(mins[0], maxs[0], grid_size))
    bins.append(np.arange(mins[2], maxs[2], grid_size))
    # bins = [np.arange(mins[i], maxs[i], grid_size) for i in range(len(mins))]   # Create bins array of each dimension

    # print(bins)
    counts, edges, binnumbers = stats.binned_statistic_dd(  # counts of each bin, edges of the bins, data-bin mapping
        pts[:,[0,2]],
        values=None,
        statistic="count",
        bins=bins,
        range=None,
        expand_binnumbers=False
    )
    ub = np.unique(binnumbers)
    pts_ds = []
    scores_ds = []
    target_coord = {}
    for b in ub:
        if len(np.where(binnumbers == b)[0]) >= min_num_pts:
            # Get the coordinate and color in this voxel
            pt = pts[np.where(binnumbers == b)[0]].mean(axis=0)
            u, c = np.unique(scores[np.where(binnumbers == b)[0]], return_counts=True, axis=0)
            cr = u[np.argmax(c)]

            scores_ds.append(cr)
            pts_ds.append(pt)
            tuple_cr = tuple(cr * 255)
            # if (tuple_cr in target_crs):
            if (tuple_cr in target_crs and not tuple_cr in target_coord):
                target_coord[target_crs[tuple_cr]] = tuple(pt[[2, 0]])    # Reverse x and z, crop y
                # target_coord[target_crs[tuple_cr]] = pt    # Reverse x and z, crop y
    pts_ds = np.vstack(pts_ds)
    scores_ds = np.vstack(scores_ds)
    return pts_ds, scores_ds, target_coord


# %%
# Voxel down size and find target
target_crs = {(255, 0, 0): "refrigerator", (0, 255, 133): "rack", (255, 9, 92): "cushion", 
                    (160, 150, 20): "lamp", (7, 255, 224): "cooktop"}
pts_ds, crs_ds, target_coord =voxel_filter(points, colors, 0.1, target_crs, 1)


# %%
# Move the coordinate
def tupleAdd(a, b):
    return tuple([sum(i) for i in zip (a, b)])
target_coord["refrigerator"] = tupleAdd(target_coord["refrigerator"], (-0.2, -0.8))
target_coord["lamp"] = tupleAdd(target_coord["lamp"], (-0.8, -0.5))
target_coord["cooktop"] = tupleAdd(target_coord["cooktop"], (0.5, -0.1)) 
target_coord["cushion"] = tupleAdd(target_coord["cushion"], (0, -1))
target_coord["rack"] = tupleAdd(target_coord["rack"], (-0.6, -0.6))


# %%
# Create bounding box for each obstacles
pts_ds[:,[0,2]]
Obstacles = np.array([(pt[1] - obstacle_threshold, pt[0] - obstacle_threshold, pt[1] + obstacle_threshold, pt[0] + obstacle_threshold) for pt in pts_ds[:,[0, 2]]])
pts_ds[:,[0,2]]


# %%
## Implement RRT

from rrt_lib.rrt import RRT
from rrt_lib.search_space import SearchSpace
from random import uniform

X_dimensions = np.array([(-4.9, 9.9), (-3, 6.2)])  # dimensions of Search Space
# x_init = (7.5, 3)  # starting location
x_goal = target_coord[target_furniture]  # goal location


Q = np.linspace(0.1, 1, 10)  # length of tree edges
r = 0.001  # length of smallest edge to check for intersection with obstacles
max_samples = 1024  # max number of samples to take before timing out
prc = 0.1  # probability of checking for a connection to goal
X = SearchSpace(X_dimensions, Obstacles)    # create search space
# Find a free goal point
# x_goal_free = tuple([x_goal[0] + 0.15 * uniform(0, 1), x_goal[1] + 0.15 * uniform(0, 1)])
# offset = 0.15
# while not X.obstacle_free(x_goal_free):
#     x_goal_free = tuple([x_goal[0] + offset * uniform(-1, 1), x_goal[1] + offset * uniform(-1, 1)])
#     offset += 0.01
# print(offset)
rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
path, rrt_pts = rrt.rrt_search()


# %%
# Plot the scatter plot
x = pts_ds[:, 2]
y = pts_ds[:, 0]
x2 = rrt_pts[:, 0]
y2 = rrt_pts[:, 1]
x3 = path[:, 0]
y3 = path[:, 1]

# x = pts_ds[:, 0]
# y = pts_ds[:, 2]
# x2 = rrt_pts[:, 1]
# y2 = rrt_pts[:, 0]
# x3 = path[:, 1]
# y3 = path[:, 0]



plt.scatter(x,y,c=crs_ds, marker=".")   # draw the scene
plt.scatter(x2,y2,c='black', marker=".")    # draw the sampled points
plt.scatter(x_goal[0],x_goal[1],c='red', marker="v")    # draw the target with triangle
plt.plot(x3,y3 , "r")   # draw the path

plt.savefig("route.png", bbox_inches='tight')

# %%
from numpy import linalg as LA
# Calculate the rotation angle and magnitude
path_3d = [[p[0],p[1]] for p in path]
# path_3d = [[p[0],p[1]] for p in path]
path_3d
vec = []
for i in range(0, len(path_3d) - 1):
    vec.append([path_3d[i + 1][0] - path_3d[i][0], path_3d[i + 1][1] - path_3d[i][1]])
vec = np.array(vec)
# rotation angle
x = vec[:,0]
y = vec[:,1]
ang = ((np.arctan2(y, x) * 180 / np.pi)) % 360.0
# magnitude
mag = LA.norm(vec, axis=1)
# %%
# Run in simulator
furniture_to_label = {"refrigerator": 67, "lamp": 47, "cooktop": 32, "cushion": 29, "rack":66}
from simulate import simulate
agent_start_pos = [x_init[1], 0, x_init[0]]
simulate(ang, mag, agent_start_pos, furniture_to_label[target_furniture])