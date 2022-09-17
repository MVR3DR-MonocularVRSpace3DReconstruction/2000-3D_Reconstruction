
import os
import numpy as np

from time import time
import datetime

import math
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt

from core.deep_global_registration import DeepGlobalRegistration
from config import get_config

from utils import *
from colored_icp import *

DATA_DIR = "./data/redwood-livingroom/"
COLOR_LIST = sorted(os.listdir(DATA_DIR+'image/'))
DEPTH_LIST = sorted(os.listdir(DATA_DIR+'depth/'))
STEP = 10


# merge pcds DFS

def pcd_fusion_dfs(_pcd_list,dgr):
	print("=> Current list: ",_pcd_list)
	# return single pcd
	if len(_pcd_list) < 2:
		return generate_point_cloud(
			DATA_DIR+'image/'+COLOR_LIST[_pcd_list[0]],
			DATA_DIR+'depth/'+DEPTH_LIST[_pcd_list[0]]
			) 
	# get half of merged pcds
	left_pcd = pcd_fusion_dfs(_pcd_list[:len(_pcd_list)//2],dgr)
	right_pcd = pcd_fusion_dfs(_pcd_list[len(_pcd_list)//2:],dgr)

	# preprocessing
	left_pcd.estimate_normals()
	right_pcd.estimate_normals()

	print('=> Registration..')
	# registration
	T, isGoodReg = dgr.register(left_pcd, right_pcd)
	left_pcd.transform(T)

	if not isGoodReg:
		# o3d.visualization.draw_geometries([left_pcd,right_pcd])
		return left_pcd
	
	T = colored_icp(left_pcd, right_pcd)
	left_pcd.transform(T)

	print('=> merge pcds')
	merged_pcd = merge_pcds([left_pcd,right_pcd])

	# storage temp
	timestamp = int(round(time() * 1000))
	o3d.io.write_point_cloud("./tmp/dfs/{}.ply".format(timestamp), merged_pcd)

	# o3d.visualization.draw_geometries([merged_pcd])

	return merged_pcd



# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

#draw graph
plt.ion()

def plot_durations(i, y):
    plt.figure(1)
#     plt.clf() 此时不能调用此函数，不然之前的点将被清空。
    plt.subplot(111)
    plt.plot(i, y, '.')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


if not os.path.exists(DATA_DIR):
  print('No such DIR - "'+DATA_DIR+'" !')

if __name__ == '__main__':
	config = get_config()
	if config.weights is None:
		config.weights = "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
	# registration
	dgr = DeepGlobalRegistration(config)

	print("* Total "+str(len(COLOR_LIST))+" RGB-D with "+str(STEP)+" per step, needs "+str(int(len(COLOR_LIST)/STEP))+" steps")

	# METHOD-1 fusion pcds DFS

	FUSION_LIST = [i for i in range(0,len(COLOR_LIST),STEP) ]
	main_pcd =  pcd_fusion_dfs(FUSION_LIST,dgr)

	o3d.visualization.draw_geometries([main_pcd])