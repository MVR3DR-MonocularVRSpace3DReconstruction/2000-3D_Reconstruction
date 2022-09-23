
from ast import main
from genericpath import isfile
import os
from turtle import right
from urllib.request import urlretrieve
import numpy as np

import open3d as o3d
from core.deep_global_registration import DeepGlobalRegistration
from config import get_config

# generate pointcloud from RGB-D
import matplotlib.pyplot as plt
import matplotlib
from utils import *
from colored_icp import *

DATA_DIR = "./data/redwood-livingroom/"
COLOR_LIST = sorted(os.listdir(DATA_DIR+'image/'))
DEPTH_LIST = sorted(os.listdir(DATA_DIR+'depth/'))
PCD_LIST = []
STEP = 1

if __name__ == '__main__':

	config = get_config()
	if config.weights is None:
		config.weights = "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
	# registration
	dgr = DeepGlobalRegistration(config)

	for pcd in sorted(os.listdir(DATA_DIR+'fragments/')):
		if ".ply" in pcd:
			temp_pcd = o3d.io.read_point_cloud(DATA_DIR+'fragments/'+pcd)
			temp_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
			PCD_LIST.append(temp_pcd)
	print("* Total "+str(len(PCD_LIST))+" point clouds loaded. ")

	# REG_PCD_LIST = [PCD_LIST[0]]
	merged_pcd = merge_pcds([PCD_LIST[0]])
	count = 0
	for pcd in PCD_LIST[1:]:
		print("==> Phase: {}// with {} times reg in total".format(count,len(PCD_LIST)))
		count += 1
		# get base from registrated pcd list
		# pcd_base = REG_PCD_LIST[-1]
		pcd_trans = pcd
		# preprocessing
		# pcd_base.estimate_normals()
		pcd_trans.estimate_normals()
		print('=> Registration..')
		# registration
		T, isGoodReg = dgr.register(pcd_trans, merged_pcd) # pcd_base
		pcd_trans.transform(T)
		# color registration
		T = colored_icp(pcd_trans, merged_pcd) # pcd_base
		pcd_trans.transform(T)
		# stored pcd
		# REG_PCD_LIST.append(pcd_trans)
		# if isGoodReg:
		# 	print('=> Good Registration [v]')
		# else:
		# 	print('=> Bad Registration [x]')
		merged_pcd = merge_pcds([merged_pcd, pcd_trans])
		# o3d.visualization.draw_geometries([merged_pcd])
		o3d.io.write_point_cloud("./outputs/method-static-duration.ply", merged_pcd)
	o3d.visualization.draw_geometries([merged_pcd])	
