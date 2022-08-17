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
pcd1 = generate_point_cloud('./data/redwood-livingroom/image/00510.jpg','./data/redwood-livingroom/depth/00510.png')

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

pcd2 = generate_point_cloud('./data/redwood-livingroom/image/00520.jpg','./data/redwood-livingroom/depth/00520.png')
pcd2.estimate_normals()

T, wsum = dgr.register(pcd2, pcd1)

pcd2.transform(T)

o3d.visualization.draw_geometries([pcd0,pcd1,pcd2])