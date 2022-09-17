
import os
import re
import numpy as np
import pandas as pd
import open3d as o3d
import open3d.core as o3c
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from core.deep_global_registration import DeepGlobalRegistration
from config import get_config

from plyfile import PlyData, PlyElement

from utils import *
import pcd2mesh
from colored_icp import *
from pathlib import Path

# imgPath = Path("./data/midas/depth/")
# depthList = sorted(os.listdir(imgPath))
# for f in depthList:
# 	if ".png" in f:
# 		print(f)
# 		img = cv2.imread(str(imgPath / f))
# 		w, h, c = img.shape[0:3]
# 		for i in range(w):
# 			for j in range(h):
# 				img[i][j] = (255 - img[i][j]) // 10
# 		cv2.imwrite(str(imgPath / f), img)


data_path = "./data/redwood-livingroom/"
image_files = sorted(os.listdir(data_path+'image/'))
depth_files = sorted(os.listdir(data_path+'depth/'))

idx = 50
color_image_path = data_path+'image/'+image_files[idx]
depth_image_path = data_path+'depth/'+depth_files[idx]

color_sample = Image.open(color_image_path)
depth_sample = Image.open(depth_image_path)
color_sample = color_sample.resize(depth_sample.size)
color_sample.save(color_image_path)

color_raw = o3d.io.read_image(color_image_path)
depth_raw = o3d.io.read_image(depth_image_path)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
	rgbd_image,
	o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
)
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

print(np.array(pcd.points))
# o3d.io.write_point_cloud("room2.ply",pcd)
o3d.visualization.draw_geometries([pcd])







# config = get_config()
# if config.weights is None:
# 	config.weights = "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
# # registration
# dgr = DeepGlobalRegistration(config)
# print("* Total "+str(len(image_files))+" RGB-D with "+str(STEP)+" per step, needs "+str(int(len(image_files)/STEP))+" steps")

# # registration
# T, isGoodReg = dgr.register(target, source)
# print(T)

# target.transform(T)

# o3d.visualization.draw_geometries([source,target])












# color icp

# T = colored_icp(target,source)

# target.transform(T)

# o3d.visualization.draw_geometries([source,target])

# color icp in cpp







# T = colored_icp(target,source)

# target.transform(T)

# o3d.visualization.draw_geometries([source,target])