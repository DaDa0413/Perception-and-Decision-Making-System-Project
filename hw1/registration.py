''' 
                                    *********************
                                    *      ICP-SVD      *
                                    *                   *
                                    *                   *
                                    *      Kyriakos     *
                                    *      Lite         *
                                    *                   *
                                    *********************
Functions:

cmdparser():
• --fix, filename of fixed point point cloud. Default = 'point_cloud_a.txt'
• --mov, filename of moving point point cloud. Default = 'point_cloud_b.txt'
• --thres, error threshold. Default = 0.0001
• --iter, maximum Iterations. Default = 100
• --errplt, Boolean value that controls the generation of the error convergence to “0”. Default = True
• --plt, Boolean value that controls the generation of images of the cloud points alignment during execution.
The pictures are saved at the working directory. Default = False ~~~SLOW~~~

pointparser(filename): a function that parses the 3D points from a text files and returns a np.array of dimension (N,3)

indxtMean(index,arrays): calculates the centroid of the corresponded points in the fixed point-set.

Indxtfixed(index,arrays): returns an array of the corresponding from the fixed point cloud.

plotter(fixed,moving,i): used only when the user wants to save the images of the two point-clouds aligning to one another.
    It is set by default as False dude to the computational deficiency.

errplotter(errStorage): plots the error evolvement at the end of execution.

ICPSVD(fixed,moving,thres,maxIter,pltornot): returns the final homogeneous matrix that describes the necessary Roto – Translation
    to align the moving points with the fixed ones, a vector containing the error convergence and the transformed moving points.

'''


import numpy as np
import argparse
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree as KDTree
import open3d as o3d
import copy

# def pointParser(fileName):
#         temp = []
#         with open(fileName) as f:
#                 for l in f:
#                         x, y, z = l.split()
#                         temp.append([float(x), float(y), float(z)])
#         return np.asarray(temp)

def indxtMean(index,arrays):
    indxSum = np.array([0.0, 0.0 ,0.0])
    for i in range(np.size(index,0)):
        indxSum = np.add(indxSum, np.array(arrays[index[i]]), out = indxSum ,casting = 'unsafe')
    return indxSum/np.size(index,0)

def indxtfixed(index,arrays):
    T = []
    for i in index:
        T.append(arrays[i])
    return np.asanyarray(T)


def ICPSVD(target_pcd, source_pcd, trans_init, threshold, mat_iteration):
    
    # Apply global registration
    source_pcd_transform = copy.deepcopy(source_pcd)
    source_pcd_transform.transform(trans_init)
    target = np.asarray(target_pcd.points)
    source = np.asarray(source_pcd.points)

    # The final transformation matrix
    transfromation = np.identity(4)
    rotation = np.identity(3)
    trainsition = [0.0, 0.0, 0.0]
    TREE = KDTree(target)
    err = 999999

    for i in range(mat_iteration):
        preverr = err
        """Conduct a tree search"""
        distance, index = TREE.query(source)
        """ Eliminate outlier"""
        q3 = np.percentile(distance, [75], interpolation='nearest')[0]
        valid = distance < q3
        distance = distance[valid]
        index = index[valid]
        source_inlier = source[valid]
        n = np.size(source_inlier, 0)

        """Calculate and store the Error"""
        # print(distance)
        err = np.mean(distance**2)
        # errStorage.append(err)
        """Calculate the Centroid of source and target point clouds (Corresponded points)"""
        source_centroid = np.mean(source_inlier, 0)
        target_centroid = indxtMean(index, target)
        """Form the W matrix to calculate the necessary Rot Matrix"""
        H = np.dot(np.transpose(source_inlier), indxtfixed(index, target)) - n*np.outer(source_centroid, target_centroid)   
        H = np.nan_to_num(H)
        U , _ , VT  = np.linalg.svd(H, full_matrices = False) 
        tempR = np.dot(VT.T,U.T)
        """Calculate the Needed Translation"""
        tempT = target_centroid - np.dot(tempR, source_centroid)
        """Apply the Computed Rotation and Translation to the Moving Points"""
        source = (tempR.dot(source.T)).T
        source = np.add(source, tempT)
        """Store the RotoTranslation"""
        rotation = np.dot(tempR, rotation)
        trainsition = np.add(np.dot(tempR, trainsition), tempT)
        # print('{} Cycle the MSE is equal to {}'.format(i+1,err))

        """Error Check """
        if abs(preverr-err) < threshold:
            """Create a Homogeneous Matrix of the Results and plot"""
            transfromation[0:3,0:3] = rotation[0:,0:]
            transfromation[0:3,3] = trainsition[:]
            # print('\nThe Algorithm has exited on the {}th iteration with Error: {}\n'.format(i+1,err))
            # print('The Homogeneous Transformation matrix =\n \n {}'.format(finhom))
            break
    
    return transfromation
                        