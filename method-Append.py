
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

DATA_DIR = "./data/apartment/"
COLOR_LIST = sorted(os.listdir(DATA_DIR+'image/'))
DEPTH_LIST = sorted(os.listdir(DATA_DIR+'depth/'))
STEP = 1
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

if not os.path.exists(DATA_DIR):
  print('No such DIR - "'+DATA_DIR+'" !')

if __name__ == '__main__':

	config = get_config()
	if config.weights is None:
		config.weights = "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
	# registration
	dgr = DeepGlobalRegistration(config)
	print("* Total "+str(len(COLOR_LIST))+" RGB-D with "+str(STEP)+" per step, needs "+str(int(len(COLOR_LIST)/STEP))+" steps")

	PCD_LIST = [generate_point_cloud(
			DATA_DIR+'image/'+COLOR_LIST[i],
			DATA_DIR+'depth/'+DEPTH_LIST[i]
			)  for i in range(0,len(COLOR_LIST),STEP)]
	print("=> PCD_LIST generated")

	REG_PCD_LIST = [PCD_LIST[0]]
	count = 0
	for pcd in PCD_LIST:
		print("==> Phase: {}// with {} times reg in total".format(count,len(PCD_LIST)))
		count += 1
		# get base from registrated pcd list
		pcd_base = REG_PCD_LIST[-1]
		pcd_trans = pcd
		# preprocessing
		# pcd_base.estimate_normals()
		pcd_trans.estimate_normals()
		print('=> Registration..')
		# registration
		T, isGoodReg = dgr.register(pcd_trans, pcd_base)
		pcd_trans.transform(T)
		# color registration
		T = colored_icp(pcd_trans,pcd_base)
		pcd_trans.transform(T)
		# stored pcd
		REG_PCD_LIST.append(pcd_trans)
		if isGoodReg:
			print('=> Good Registration [v]')
		else:
			print('=> Bad Registration [x]')
		merged_pcd = merge_pcds(REG_PCD_LIST)
		o3d.visualization.draw_geometries([merged_pcd])
		o3d.io.write_point_cloud("./tmp/method-static-duration.ply", merged_pcd)
	o3d.visualization.draw_geometries(REG_PCD_LIST)	
