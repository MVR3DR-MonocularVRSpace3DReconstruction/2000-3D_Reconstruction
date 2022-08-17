from itertools import count
import os
import numpy as np
import open3d as o3d
from camera_pose import read_trajectory

from utils import *
# from core.deep_global_registration import DeepGlobalRegistration
from core.deep_global_registration import DeepGlobalRegistration
from config import get_config

import random
import math
# generate pointcloud from RGB-D

from pcd2mesh import pcd_to_trianglemesh

pcd0 = generate_point_cloud('./data/redwood-livingroom/image/00500.jpg','./data/redwood-livingroom/depth/00500.png')
pcd1 = generate_point_cloud('./data/redwood-livingroom/image/00550.jpg','./data/redwood-livingroom/depth/00550.png')

config = get_config()
if config.weights is None:
    config.weights = "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"

# registration
dgr = DeepGlobalRegistration(config)
# preprocessing
pcd0.estimate_normals()
pcd1.estimate_normals()

# pcd0.paint_uniform_color([0.1,0.2,0.8])
# pcd1.paint_uniform_color([0.5,0,0.5])

T, wsum = dgr.register(pcd1, pcd0)
# pcd1.transform(T)

pcd1.transform(T)




box_0 = pcd0.get_axis_aligned_bounding_box()
box_0.color = [0,1,0]
box_1 = pcd1.get_axis_aligned_bounding_box()
box_1.color = [1,0,0]
bp_0 = box_0.get_box_points()
bp_1 = box_1.get_box_points()
# print(np.asarray(bp_0),np.asarray(bp_1))
# print(np.asarray(bp_0)[:,1])
pcd = concate_pcds([pcd0,pcd1])
box = pcd.get_axis_aligned_bounding_box()


# box_center, box_extent = get_max_box_center_extent(np.asarray(bp_0),np.asarray(bp_1))

# box = o3d.geometry.OrientedBoundingBox(box_center,np.eye(3),box_extent)

box.color = [0,0,1]

# pcd0 = pcd0.select_by_index(mask,invert = True)
# pcd1 = pcd1.select_by_index(mask,invert = True)
print(box.get_min_bound(), box.get_max_bound())
def slice_grid_pcds(pcd0, pcd1, box, step):
    box_points = np.asarray(box.get_box_points())
    start_point = [min(box_points[:,0]), min(box_points[:,1]),min(box_points[:,2]) ]
    # print(start_point)
    extent = box.get_extent()
    grid_list = []
    px, py, pz = start_point
    for x in range(1, math.floor((extent[0] * 2 )/step) - 1):
        for y in range(1, math.floor((extent[1] * 2 )/step) - 1):
            for z in range(1, math.floor((extent[1] * 2 )/step) - 1):
                cx = start_point[0] + step * x
                cy = start_point[1] + step * y
                cz = start_point[2] + step * z
                grid_box = o3d.geometry.AxisAlignedBoundingBox([px, py, pz],[cx, cy, cz])
                # print('px py pz cx cy cz\n',[px, py, pz],'\n',[cx, cy, cz],'\n')
                if random.choice([True, False]):
                    grid = pcd0.crop(grid_box)
                else:
                    grid = pcd1.crop(grid_box)
                grid_list.append(grid)
                pz = cz
            pz = start_point[2]
            py = cy
        pz = start_point[2]
        py = start_point[1]
        px = cx
    pcd = concate_pcds(grid_list)
    return pcd

pcd = slice_grid_pcds(pcd0, pcd1, box, 0.3)

# pcd = pcd.crop(box)
o3d.visualization.draw_geometries([pcd,box_0,box_1,box]) #,pcd1,box_0,box_1


