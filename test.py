
import os
import re
import numpy as np
import pandas as pd
import open3d as o3d
import open3d.core as o3c
import matplotlib
import matplotlib.pyplot as plt

from core.deep_global_registration import DeepGlobalRegistration
from config import get_config

from plyfile import PlyData, PlyElement

from utils import *
import pcd2mesh
from colored_icp import *


'''
color icp fusion

'''

DATA_DIR = "./data/redwood-livingroom/"
COLOR_LIST = sorted(os.listdir(DATA_DIR+'image/'))
DEPTH_LIST = sorted(os.listdir(DATA_DIR+'depth/'))
STEP = 10


source = generate_point_cloud(
	DATA_DIR+'image/'+COLOR_LIST[255],
	DATA_DIR+'depth/'+DEPTH_LIST[255]
	)
target = generate_point_cloud(
	DATA_DIR+'image/'+COLOR_LIST[286],
	DATA_DIR+'depth/'+DEPTH_LIST[286]
	)
# preprocessing
source.estimate_normals()
target.estimate_normals()


o3d.visualization.draw_geometries([source,target])

config = get_config()
if config.weights is None:
	config.weights = "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
# registration
dgr = DeepGlobalRegistration(config)
print("* Total "+str(len(COLOR_LIST))+" RGB-D with "+str(STEP)+" per step, needs "+str(int(len(COLOR_LIST)/STEP))+" steps")

# registration
T, isGoodReg = dgr.register(target, source)
print(T)

target.transform(T)

o3d.visualization.draw_geometries([source,target])

# color icp

# T = colored_icp(target,source)

# target.transform(T)

# o3d.visualization.draw_geometries([source,target])

# color icp in cpp

T = color_icp_cpp(target,source)

target.transform(T)

o3d.visualization.draw_geometries([source,target])