import os
import numpy as np
import open3d as o3d
from core.deep_global_registration import DeepGlobalRegistration
from config import get_config

from simpleicp import PointCloud, SimpleICP
from utils import generate_point_cloud_with_camera_pose, merge_pcds, color_icp_cpp, read_trajectory
from colored_icp import colored_icp

from time import clock

# boardroom
DATA_DIR = "./data/redwood-livingroom/"
POSE_FILE = "livingroom.log"
COLOR_LIST = sorted(os.listdir(DATA_DIR+'image/'))
DEPTH_LIST = sorted(os.listdir(DATA_DIR+'depth/'))
STEP = 10

camera_poses = read_trajectory("{}{}".format(DATA_DIR, POSE_FILE))
error_log = []


def simpleICP(pcd_base, pcd_trans):
	pc_fix = PointCloud(np.array(pcd_base.points), columns=["x", "y", "z"])
	pc_mov = PointCloud(np.array(pcd_trans.points), columns=["x", "y", "z"])
	# Create simpleICP object, add point clouds, and run algorithm!
	icp = SimpleICP()
	icp.add_point_clouds(pc_fix, pc_mov)
	T, X_mov_transformed, rigid_body_transformation_params = icp.run(max_overlap_distance=1)
	# print(T)
	# o3d.visualization.draw_geometries([pcd_base, pcd_trans])
	# pcd_trans.transform(T)
	return T

def draw_vis(vis, geo):
	vis.update_geometry(geo)
	vis.poll_events()
	vis.update_renderer()

def pcd_fusion_dfs(_pcd_dix,dgr):
	print("="*50)
	print("="*50)
	print("=> Error List: ",error_log)
	print("="*50)
	print("\n\n=> Current list: ",_pcd_dix)
	###########################################################
	# return minimal point cloud
	###########################################################

	if len(_pcd_dix) < 2:
		print("=> Generate {}th point cloud. ".format(_pcd_dix[0]))
		pcd = generate_point_cloud_with_camera_pose(
			DATA_DIR+'image/'+COLOR_LIST[_pcd_dix[0]],
			DATA_DIR+'depth/'+DEPTH_LIST[_pcd_dix[0]],
			camera_poses[_pcd_dix[0]].pose
			)
		pcd.estimate_normals()
		return pcd
	###########################################################
	# get half of merged point clouds
	###########################################################
	left_pcd = pcd_fusion_dfs(_pcd_dix[:len(_pcd_dix)//2],dgr)
	right_pcd = pcd_fusion_dfs(_pcd_dix[len(_pcd_dix)//2:],dgr)

	l_pcd = left_pcd.random_down_sample(0.03)
	r_pcd = right_pcd.random_down_sample(0.03)

	print('=> Registration...')

	###########################################################
	# color icp from CPP
	###########################################################

	# T = color_icp_cpp(l_pcd, r_pcd)
	# left_pcd.transform(T)
	# l_pcd.transform(T)

	###########################################################
	# registration
	###########################################################

	
	T, isGoodReg = dgr.register(l_pcd, r_pcd)
	left_pcd.transform(T)
	l_pcd.transform(T)

	if not isGoodReg:
		print("==> NOT GOOD REGISTRATION !!!\n==> For point cloud: {}-{}.".format(_pcd_dix[0], _pcd_dix[-1]))
		error_log.append(str(_pcd_dix[0])+"-"+str(_pcd_dix[-1]))
		# o3d.visualization.draw_geometries([left_pcd,right_pcd])
		# return left_pcd

	

	# T = simpleICP(right_pcd, left_pcd)
	# left_pcd.transform(T)

	###########################################################
	# colored icp 
	###########################################################

	T = colored_icp(l_pcd, r_pcd)
	left_pcd.transform(T)
	l_pcd.transform(T)
	# merge
	print('=> merge pcds')
	merged_pcd = merge_pcds([left_pcd,right_pcd])
	merged_pcd.random_down_sample(2/((_pcd_dix[-1] - _pcd_dix[0])//STEP))
	merged_pcd.voxel_down_sample(0.05)

	# store cache and return
	o3d.io.write_point_cloud("./tmp/dfs/{}-{}.ply".format(_pcd_dix[0], _pcd_dix[-1]), merged_pcd)
	return merged_pcd




if __name__ == '__main__':

	start_time = clock()
	###########################################################
	# refresh cache
	###########################################################
	os.system("rm -rf ./tmp/dfs")
	os.system("mkdir ./tmp/dfs")
	config = get_config()
	if config.weights is None:
		config.weights = "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
							#ResUNetBN2C-feat32-3dmatch-v0.05.pth
							#ResUNetBN2C-feat32-kitti-v0.3.pth
	dgr = DeepGlobalRegistration(config)

	print("* Total "+str(len(COLOR_LIST))+" RGB-D with "+str(STEP)+" per step, needs "+str(int(len(COLOR_LIST)/STEP))+" steps")

	###########################################################
	# DFS
	###########################################################

	FUSION_IDX = [i for i in range(0,len(COLOR_LIST),STEP)]

	main_pcd =  pcd_fusion_dfs(FUSION_IDX,dgr)

	print("# Error pcd fusion: \n",error_log)
	o3d.visualization.draw_geometries([main_pcd])


if False:

	config = get_config()
	if config.weights is None:
		config.weights = "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
	# registration
	# dgr = DeepGlobalRegistration(config)

	print("* Total "+str(len(COLOR_LIST))+" RGB-D with "+str(STEP)+" per step, needs "+str(int(len(COLOR_LIST)/STEP))+" steps")

	camera_poses = read_trajectory("{}{}".format(DATA_DIR, POSE_FILE))
	pcd_base = generate_point_cloud_with_camera_pose(
		"{}image/{}".format(DATA_DIR, COLOR_LIST[0]),
		"{}depth/{}".format(DATA_DIR, DEPTH_LIST[0]),
		camera_poses[0].pose
	)

	for i in range(0,len(COLOR_LIST),STEP):

		print("=> {} time fusion// with {} in total.".format(i, len(COLOR_LIST) // STEP))

		pcd = generate_point_cloud_with_camera_pose(
			"{}image/{}".format(DATA_DIR, COLOR_LIST[i]),
			"{}depth/{}".format(DATA_DIR, DEPTH_LIST[i]),
			camera_poses[i].pose
		)
		pcd.estimate_normals()
		# o3d.visualization.draw_geometries([pcd_base,pcd])
		print('=> Registration..')
		# registration
		# T, isGoodReg = dgr.register(pcd, pcd_base)
		# pcd.transform(T)
		# color registration
		T = color_icp_cpp(pcd,pcd_base)
		pcd.transform(T)

		# stored pcd
		pcd_base.random_down_sample(0.8)

		pcd_base = merge_pcds([pcd_base,pcd])
		# o3d.visualization.draw_geometries([pcd_base])
		o3d.io.write_point_cloud("./tmp/main.ply", pcd_base)

	o3d.visualization.draw_geometries([pcd_base])