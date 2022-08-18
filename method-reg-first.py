
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
from colored_icp import *



DATA_DIR = "./data/redwood-livingroom/"
COLOR_LIST = sorted(os.listdir(DATA_DIR+'image/'))
DEPTH_LIST = sorted(os.listdir(DATA_DIR+'depth/'))
STEP = 10

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

	# METHOD-3 reg first

	PCD_LIST = [generate_point_cloud(
			DATA_DIR+'image/'+COLOR_LIST[i],
			DATA_DIR+'depth/'+DEPTH_LIST[i]
			)  for i in range(0,len(COLOR_LIST),STEP)]
	print("=> PCD_LIST generated")

	REG_PCD_LIST = []
	current_reg_list = [PCD_LIST[0]]
	count = 0
	for i in PCD_LIST:
		print("==> Phase: {}// with {} times reg in total".format(count,len(PCD_LIST)))
		count += 1

		pcd_base = current_reg_list[-1]
		pcd_trans = i
		# preprocessing
		pcd_base.estimate_normals()
		pcd_trans.estimate_normals()
		print('=> Registration..')
		# registration
		T, isGoodReg = dgr.register(pcd_trans, pcd_base)
		if isGoodReg:
			pcd_trans.transform(T)

			T = colored_icp(pcd_trans,pcd_base)
			pcd_trans.transform(T)

			current_reg_list.append(pcd_trans)
			
			o3d.visualization.draw_geometries(REG_PCD_LIST[i])

			print('=> Good Registration [v]\n\n')
		else:
			REG_PCD_LIST.append(current_reg_list)
			current_reg_list = [pcd_trans]

			print('=> Bad Registration [x] \n \t append new pcd array')

	print("## {} sub list generated with length:\n\t{}".format(len(REG_PCD_LIST),[len(i) for i in REG_PCD_LIST]))


	for i in range(len(REG_PCD_LIST)):
		if len(REG_PCD_LIST[i]) > 3:
			o3d.visualization.draw_geometries(REG_PCD_LIST[i])
			merged_pcd = merge_pcds(REG_PCD_LIST[i])
			o3d.io.write_point_cloud("./tmp/regF/tmp-{:0>5}.ply".format(i), merged_pcd)
