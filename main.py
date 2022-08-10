
# Cite from:
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019


from ast import main
import os
from turtle import right
from urllib.request import urlretrieve
import numpy as np

import open3d as o3d
from core.deep_global_registration import DeepGlobalRegistration
from config import get_config

# generate pointcloud from RGB-D
from utils import *



DATA_DIR = "./data/redwood-livingroom/"
COLOR_LIST = sorted(os.listdir(DATA_DIR+'image/'))
DEPTH_LIST = sorted(os.listdir(DATA_DIR+'depth/'))
STEP = 20

# concate pcds DFS

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
	T, wsum = dgr.register(left_pcd, right_pcd)
	# print(T)
	left_pcd.transform(T)

	print('=> Concate pcds')
	# concate pcds
	# cur_pcd = cur_pcd.voxel_down_sample(voxel_size=0.03)
	concated_pcd = concate_pcds([left_pcd,right_pcd])
	mean, cov = concated_pcd.compute_mean_and_covariance()
	print(mean, cov)

	down_sample = 1e6/len(concated_pcd.points)
	print('# pcd size:',len(concated_pcd.points), 'down sample:',down_sample)
	concated_pcd = concated_pcd.random_down_sample(0.5)
	# concated_pcd = concated_pcd.normalize_normals()
	if wsum < 200:
		o3d.visualization.draw_geometries([concated_pcd])
		return left_pcd
	
	# storage temp
	o3d.io.write_point_cloud("./tmp/tmp.ply", concated_pcd)

	return concated_pcd


# concate pcds volume

def pcd_fusion_vol(_pcd_list):
	print("=> Current list: ",_pcd_list[0],_pcd_list[1])
	pcd0 = generate_point_cloud(
		DATA_DIR+'image/'+_pcd_list[0]+'.jpg',
		DATA_DIR+'depth/'+_pcd_list[0]+'.png'
		)
	pcd1 = generate_point_cloud(
		DATA_DIR+'image/'+_pcd_list[1]+'.jpg',
		DATA_DIR+'depth/'+_pcd_list[1]+'.png'
		)
	# preprocessing
	pcd0.estimate_normals()
	pcd1.estimate_normals()
	print('=> Registration..')
	# registration
	T, wsum = dgr.register(pcd1, pcd0)
	print(T)
	sum_pcd = fusion_pcds(
		_pcd_list[0],
		_pcd_list[1],
		T,
		[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
		)
	o3d.visualization.draw_geometries([sum_pcd])

	# # sum up pcd
	# sum_pcd = generate_point_cloud(
	# 		DATA_DIR+'image/'+_pcd_list[0],
	# 		DATA_DIR+'depth/'+_pcd_list[0]
	# 		)
	# for rgbd_file in _pcd_list:
	# 	# get pcd
	# 	tmp_pcd = generate_point_cloud(
	# 		DATA_DIR+'image/'+rgbd_file,
	# 		DATA_DIR+'depth/'+rgbd_file
	# 		)
	# 	# preprocessing
	# 	sum_pcd.estimate_normals()
	# 	tmp_pcd.estimate_normals()
	# 	print('=> Registration..')
	# 	# registration
	# 	T, wsum = dgr.register(sum_pcd, tmp_pcd)
	# 	# print(T)
	# 	# sum_pcd.transform(T)
	# 	sum_pcd = fusion_pcds(_pcd_list[0],_pcd_list[1],T)




if not os.path.exists(DATA_DIR):
  print('No such DIR - "'+DATA_DIR+'" !')

if __name__ == '__main__':

	config = get_config()
	if config.weights is None:
		config.weights = "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
						#  "./pth/ResUNetBN2C-feat32-kitti-v0.3.pth"
						#  "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
	# registration
	dgr = DeepGlobalRegistration(config)

	print("* Total "+str(len(COLOR_LIST))+" RGB-D with "+str(STEP)+" per step, needs "+str(int(len(COLOR_LIST)/STEP))+" steps")

	# fusion pcds DFS

	FUSION_LIST = [i for i in range(0,len(COLOR_LIST),STEP) ]
	main_pcd =  pcd_fusion_dfs(FUSION_LIST,dgr)

	# fusion pcds VOLUME

	# FUSION_LIST = [COLOR_LIST[i][:-4] for i in range(0,len(COLOR_LIST),STEP) ]
	# # print(FUSION_LIST)
	# main_pcd = pcd_fusion_vol(FUSION_LIST)

	o3d.visualization.draw_geometries([main_pcd])