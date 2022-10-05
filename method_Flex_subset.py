
import os
from pathlib import Path
import glob
import numpy as np
from PIL import Image
import open3d as o3d
from deep_global_registration.core.deep_global_registration import DeepGlobalRegistration
from deep_global_registration.config import get_config
from utils import *
from colored_icp import *

data_dir = Path("./data/classroom/")
image_dir = sorted(glob.glob(str(data_dir/"image/*.jpg")))
depth_dir = sorted(glob.glob(str(data_dir/"depth_out/*.png")))
STEP = 1

def generate_point_cloud(image_dir:str, depth_dir:str):
    color_raw = o3d.io.read_image(image_dir)
    depth_raw = o3d.io.read_image(depth_dir)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(640,480,1050,1050,320,240),
    )
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd

if __name__ == '__main__':

	# config = get_config()
	# if config.weights is None:
	# 	config.weights = "deep_global_registration/pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
	# # registration
	# dgr = DeepGlobalRegistration(config)

	print("* Total "+str(len(image_dir))+" RGB-D with "+str(STEP)+" per step, needs "+str(int(len(image_dir)/STEP))+" steps")
	REG_PCD_LIST = []

	# curr_pcd_list = [generate_point_cloud(
	# 		data_dir+'image/'+image_dir[curr_pcd],
	# 		data_dir+'depth/'+depth_dir[curr_pcd]
	# 		)]
	
	print(np.array(Image.open(depth_dir[0])))
	curr_pcd_list = [generate_point_cloud(str(image_dir[0]), str(depth_dir[0])),]

	# for pcd in curr_pcd_list:
	o3d.visualization.draw_geometries([curr_pcd_list[0]])
	