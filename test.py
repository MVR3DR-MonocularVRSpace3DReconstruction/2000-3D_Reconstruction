from itertools import count
import os
import numpy as np
import open3d as o3d
from camera_pose import read_trajectory
from depth2pcd import generate_point_cloud
from depth2pcd import generate_point_cloud_with_matrix
# from core.deep_global_registration import DeepGlobalRegistration
from core.deep_global_registration import DeepGlobalRegistration
from config import get_config

# generate pointcloud from RGB-D
from depth2pcd import generate_point_cloud
from depth2pcd import fusion_pcds
from depth2pcd import concate_pcds

from pcd2mesh import pcd_to_trianglemesh

pcd0 = generate_point_cloud('./data/redwood-livingroom/image/00000.jpg','./data/redwood-livingroom/depth/00000.png')
pcd1 = generate_point_cloud('./data/redwood-livingroom/image/00100.jpg','./data/redwood-livingroom/depth/00100.png')

config = get_config()
if config.weights is None:
    config.weights = "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"

# registration
dgr = DeepGlobalRegistration(config)
# preprocessing
pcd0.estimate_normals()
pcd1.estimate_normals()

T, wsum = dgr.register(pcd1, pcd0)
# pcd1.transform(T)

pcd1.transform(T)

# delete points which has smaller distance than origin point cloud

ori_min_d = min(pcd0.compute_nearest_neighbor_distance())
print("min distance:",ori_min_d)

pcd = concate_pcds(pcd0, pcd1)
conc_min_d = min(pcd.compute_nearest_neighbor_distance())
print("min distance:",conc_min_d)



# print("division:", ori_min_d / conc_min_d)
D = pcd.compute_nearest_neighbor_distance()
# print(D)
mask = []
for d in range(len(D)):
    if D[d] < ori_min_d:
        mask.append(d)

# print(mask)

pcd_tree = pcd_tree = o3d.geometry.KDTreeFlann(pcd)

pcd = pcd.select_by_index(mask,invert = True)

pcd.estimate_normals()

o3d.visualization.draw_geometries([pcd])

