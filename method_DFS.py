
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
from overlap import overlap_predator

data_dir = "./data/redwood-livingroom/"

def pcd_fusion_dfs(_pcd_list, depth):
	
	# return single pcd
	if len(_pcd_list) < 2:
		print("="*20)
		print("=> Single Point Cloud [Returned]")
		print("="*20)
		return _pcd_list[0]
	# get half of merged pcds
	left_pcd = pcd_fusion_dfs(_pcd_list[:len(_pcd_list)//2], depth+1)
	right_pcd = pcd_fusion_dfs(_pcd_list[len(_pcd_list)//2:], depth+1)
	# o3d.visualization.draw_geometries([left_pcd, right_pcd])
	# preprocessing
	left_pcd.estimate_normals()
	right_pcd.estimate_normals()

	print('=> Registration..')
	# registration
	if len(_pcd_list) < 4:
		T = overlap_predator(left_pcd, right_pcd)
	else:
		T, _ = DGR.register(left_pcd, right_pcd)
	left_pcd.transform(T)
	T = colored_icp(left_pcd, right_pcd)
	left_pcd.transform(T)

	print('=> Merge pcds')
	merged_pcd = merge_pcds([left_pcd,right_pcd])
	# storage temp
	timestamp = int(round(time() * 1000))
	o3d.io.write_point_cloud("./outputs/dfs/{}.ply".format(timestamp), merged_pcd)
	# o3d.visualization.draw_geometries([merged_pcd])

	print("="*10*(10-depth))
	print("=> List length: {} Stack Depth: {} [Merged Complete]".format(len(_pcd_list), depth))
	print("="*10*(10-depth))
	return merged_pcd

if __name__ == '__main__':

	start_time = time()
	config = get_config()
	config.weights = "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
	DGR = DeepGlobalRegistration(config)

	pcds = read_point_clouds(data_dir = "./data/redwood-livingroom/",down_sample=1)


	pause_time = time()
	o3d.visualization.draw_geometries(pcds)
	pause_time = time() - pause_time


	main_pcd =  pcd_fusion_dfs(pcds, 0)


	end_time = time()
	time_cost = end_time-start_time-pause_time
	print("\n## Total cost {}s = {}m{}s.".format(
		time_cost, int((time_cost)//60), int(time_cost - (time_cost)//60*60)))
	o3d.io.write_point_cloud("./outputs/dfs/DFS-outputs.ply", main_pcd)
	o3d.visualization.draw_geometries([main_pcd])