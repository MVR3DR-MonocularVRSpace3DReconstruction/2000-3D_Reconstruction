
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

data_dir = "./data/redwood-livingroom/"
def read_point_clouds(data_dir = "./data/redwood-livingroom/",down_sample=0.1):
    pcds = []
    count = 0
    for pcd in sorted(os.listdir(data_dir+'fragments/')):
        if ".ply" in pcd:
            temp_pcd = o3d.io.read_point_cloud(data_dir+'fragments/'+pcd)
            temp_pcd = temp_pcd.random_down_sample(down_sample)
            temp_pcd.estimate_normals()
            temp_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            
            pcds.append(temp_pcd)
            print("=> Load point cloud <{0}> [{1}/{2}]".format(pcd,count,"?"))
            count += 1
        # if count == 5:
        #     break
    print("="*50)
    print("=> Total "+str(len(pcds))+" point clouds loaded. ")
    print("="*50)
    return pcds
# merge pcds DFS

def append_point_clouds(pcds):
	reged_pcds = [pcds[0]]
	count = 0
	for pcd in pcds[1:]:
		print("==> Phase: {}// with {} times reg in total".format(count,len(pcds)))
		count += 1
		# get base from registrated pcd list
		pcd_base = reged_pcds[-1]
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
		reged_pcds.append(pcd_trans)
		if isGoodReg:
			print('=> Good Registration [v]')
		else:
			print('=> Bad Registration [x]')
	pcd = merge_pcds(reged_pcds)
	return pcd

def execute_global_registration(source_down, target_down):
    distance_threshold = voxel_size * 1.5

    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    print("=> voxel size: {} // distance threshold: {}".format(voxel_size, distance_threshold))
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result.transformation

def pcd_fusion_dfs(_pcd_list, depth):
	print("="*50)
	print("=> Current list length: {} Current Stack Depth: {}".format(len(_pcd_list), depth))
	print("="*50)
	# return single pcd
	if len(_pcd_list) < 2:
		return _pcd_list[0]
	# get half of merged pcds
	left_pcd = pcd_fusion_dfs(_pcd_list[:int(len(_pcd_list)*2/3)], depth+1)
	right_pcd = pcd_fusion_dfs(_pcd_list[int(len(_pcd_list)*2/3):], depth+1)

	# preprocessing
	left_pcd.estimate_normals()
	right_pcd.estimate_normals()

	print('=> Registration..')
	# registration
	T, isGoodReg = dgr.register(left_pcd, right_pcd)
	# T = execute_global_registration(left_pcd,right_pcd)
	left_pcd.transform(T)

	# if not isGoodReg:
	# 	o3d.visualization.draw_geometries([left_pcd,right_pcd])
	# 	return left_pcd if len(left_pcd.points) > len(right_pcd.points) else right_pcd
	
	T = colored_icp(left_pcd, right_pcd)
	left_pcd.transform(T)

	print('=> Merge pcds')
	merged_pcd = merge_pcds([left_pcd,right_pcd])
	# storage temp
	timestamp = int(round(time() * 1000))
	o3d.io.write_point_cloud("./outputs/dfs/{}.ply".format(timestamp), merged_pcd)
	o3d.visualization.draw_geometries([merged_pcd])
	return merged_pcd.random_down_sample(0.6)

if __name__ == '__main__':
	config = get_config()
	if config.weights is None:
		config.weights = "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
	# registration
	dgr = DeepGlobalRegistration(config)
	voxel_size = 0.05
	PCD_LIST = read_point_clouds(data_dir = "./data/redwood-livingroom/",down_sample=0.5)
	o3d.visualization.draw_geometries(PCD_LIST)
	# METHOD-1 fusion pcds DFS
	main_pcd =  pcd_fusion_dfs(PCD_LIST, 0)
	o3d.visualization.draw_geometries([main_pcd])