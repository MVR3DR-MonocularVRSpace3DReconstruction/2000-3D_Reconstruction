
# Cite from:
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019


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



DATA_DIR = "./data/redwood-lobby/"
COLOR_LIST = sorted(os.listdir(DATA_DIR+'image/'))
DEPTH_LIST = sorted(os.listdir(DATA_DIR+'depth/'))
STEP = 10


# reg first 





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
	# print(T)
	left_pcd.transform(T)

	print('=> merge pcds')
	# merge pcds
	# cur_pcd = cur_pcd.voxel_down_sample(voxel_size=0.03)
	merged_pcd = merge_pcds([left_pcd,right_pcd])


	# mean, cov = merged_pcd.compute_mean_and_covariance()
	# print(mean, cov)

	# down_sample = 1e6/len(merged_pcd.points)
	# print('# pcd size:',len(merged_pcd.points), 'down sample:',down_sample)
	# merged_pcd = merged_pcd.random_down_sample(0.5)
	# merged_pcd = merged_pcd.normalize_normals()

	if not isGoodReg:
		o3d.visualization.draw_geometries([merged_pcd])
		return left_pcd
	
	# storage temp
	o3d.io.write_point_cloud("./tmp/tmp.ply", merged_pcd)

	return merged_pcd


# merge pcds volume

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
						#  "./pth/ResUNetBN2C-feat32-kitti-v0.3.pth"
						#  "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
	# registration
	dgr = DeepGlobalRegistration(config)

	print("* Total "+str(len(COLOR_LIST))+" RGB-D with "+str(STEP)+" per step, needs "+str(int(len(COLOR_LIST)/STEP))+" steps")




	# METHOD-5 DFS flex duration













	# METHOD-4 flex duration
	MAX_DURATION = 100
	REDUCE = 0.75

	print("=> PCD_LIST generated")
	
	curr_pcd = 0
	duration = MAX_DURATION
	REG_PCD_LIST = [generate_point_cloud(
			DATA_DIR+'image/'+COLOR_LIST[curr_pcd],
			DATA_DIR+'depth/'+DEPTH_LIST[curr_pcd]
			)]
	#draw 
	X = []
	Y = []

	count = 0
	T = np.eye(4)
	while curr_pcd != len(COLOR_LIST)-1:
		print("==> Current pcd: [{}] \t Reg pcd: [{}] \tTotal pcd: [0-{}]".format(curr_pcd,curr_pcd+duration,len(COLOR_LIST)-1))
		
		#avoid endless loop
		if duration == 0:
			print("## WARNING endless loop in no valid registration")
			o3d.visualization.draw_geometries([pcd_base[-1],pcd_trans])
			cin = input()
			if cin == 'q':
				break
			if cin == 'c':
			# enlarge the scan range may find new fit pcds
				duration = 220
			

		pcd_base = REG_PCD_LIST[-1]
		pcd_base.voxel_down_sample(0.8)
		pcd_trans = generate_point_cloud(
			DATA_DIR+'image/'+COLOR_LIST[curr_pcd+duration],
			DATA_DIR+'depth/'+DEPTH_LIST[curr_pcd+duration]
			)

		# pcd_trans.transform(T)
		
		# preprocessing

		# pcd_base.estimate_normals()
		pcd_trans.estimate_normals()

		print('=> Registration..')
		# registration
		T, isGoodReg = dgr.register(pcd_trans, pcd_base)
		if isGoodReg:
			print('=> Good Registration [v]\n\t success duration <{}> '.format(duration))
			pcd_trans.transform(T)
			REG_PCD_LIST.append(pcd_trans)

			#log
			X.append(count)
			Y.append(duration)
			plot_durations(X, Y)


			curr_pcd += duration
			# avoid range overflow
			if curr_pcd + MAX_DURATION <= len(COLOR_LIST)-1:
				duration = MAX_DURATION
			else:
				duration = len(COLOR_LIST)-1-curr_pcd

			count += 1
			# o3d.visualization.draw_geometries([REG_PCD_LIST[-1],REG_PCD_LIST[-2]])
			
		else:
			duration = int(duration * REDUCE)
			print('=> Bad Registration [x]\n\t set duration into "{}"'.format(duration))

	print("## Registration finished")
	merged_pcd = merge_pcds(REG_PCD_LIST)
	o3d.io.write_point_cloud("./tmp/tmp.ply", merged_pcd)
	o3d.visualization.draw_geometries([merged_pcd])




	# METHOD-3 reg first

	# PCD_LIST = [generate_point_cloud(
	# 		DATA_DIR+'image/'+COLOR_LIST[i],
	# 		DATA_DIR+'depth/'+DEPTH_LIST[i]
	# 		)  for i in range(0,len(COLOR_LIST),STEP)]
	# print("=> PCD_LIST generated")

	# REG_PCD_LIST = []
	# current_reg_list = [PCD_LIST[0]]
	# count = 0
	# for i in PCD_LIST:
	# 	print("==> Phase: {}// with {} times reg in total".format(count,len(PCD_LIST)))
	# 	count += 1

	# 	pcd_base = current_reg_list[-1]
	# 	pcd_trans = i
	# 	# preprocessing
	# 	pcd_base.estimate_normals()
	# 	pcd_trans.estimate_normals()
	# 	print('=> Registration..')
	# 	# registration
	# 	T, isGoodReg = dgr.register(pcd_trans, pcd_base)
	# 	if isGoodReg:
	# 		pcd_trans.transform(T)
	# 		current_reg_list.append(pcd_trans)
	# 		print('=> Good Registration [v]')
	# 	else:
	# 		REG_PCD_LIST.append(current_reg_list)
	# 		current_reg_list = [pcd_trans]

	# 		print('=> Bad Registration [x] \n \t append new pcd array')

	# print("## {} sub list generated with length:\n\t{}".format(len(REG_PCD_LIST),[len(i) for i in REG_PCD_LIST]))


	# for i in range(len(REG_PCD_LIST)):
	# 	if len(REG_PCD_LIST[i]) > 3:
	# 		o3d.visualization.draw_geometries(REG_PCD_LIST[i])
	# 		merged_pcd = merge_pcds(REG_PCD_LIST[i])
	# 		o3d.io.write_point_cloud("./tmp/regF/tmp-{:0>5}.ply".format(i), merged_pcd)


	# METHOD-1 fusion pcds DFS

	# FUSION_LIST = [i for i in range(0,len(COLOR_LIST),STEP) ]
	# main_pcd =  pcd_fusion_dfs(FUSION_LIST,dgr)



	# METHOD-2 fusion pcds VOLUME

	# FUSION_LIST = [COLOR_LIST[i][:-4] for i in range(0,len(COLOR_LIST),STEP) ]
	# # print(FUSION_LIST)
	# main_pcd = pcd_fusion_vol(FUSION_LIST)


	# o3d.visualization.draw_geometries([REG_PCD_LIST])