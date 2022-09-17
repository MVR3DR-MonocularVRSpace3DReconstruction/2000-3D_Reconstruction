
import os
import numpy as np
import open3d as o3d
from core.deep_global_registration import DeepGlobalRegistration
from config import get_config

from simpleicp import PointCloud, SimpleICP
from utils import generate_point_cloud_with_camera_pose, merge_pcds, color_icp_cpp, read_trajectory
from colored_icp import colored_icp
import random

# boardroom
DATA_DIR = "./data/redwood-livingroom/"
POSE_FILE = "livingroom.log"
COLOR_LIST = sorted(os.listdir(DATA_DIR+'image/'))
DEPTH_LIST = sorted(os.listdir(DATA_DIR+'depth/'))
STEP = 20


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


if __name__ == '__main__':

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