from simpleicp import PointCloud, SimpleICP
import numpy as np
import random

from tqdm import tqdm
import os
import open3d as o3d
from utils import generate_point_cloud, load_point_clouds, merge_pcds, color_icp_cpp, read_trajectory, read_point_clouds
from colored_icp import colored_icp


# boardroom
data_dir = "./data/predict-livingroom/"
POSE_FILE = "livingroom.log"
COLOR_LIST = sorted(os.listdir(data_dir+'image/'))
DEPTH_LIST = sorted(os.listdir(data_dir+'depth/'))
STEP = 50

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

PCD_LIST = load_point_clouds(data_dir = data_dir, step=50, down_sample=1)
print("=> PCD_LIST generavted")
o3d.visualization.draw_geometries([PCD_LIST[0]])	
# count = 0
merged_pcd = PCD_LIST[0]
for pcd in PCD_LIST:
    # print("==> Phase: {}// with {} times reg in total".format(count,len(PCD_LIST)))
    # count += 1
    # get base from registrated pcd list
    # preprocessing
    # pcd_base.estimate_normals()
    pcd.estimate_normals()
    # print('=> Registration..')
    # registration
    T = simpleICP(merged_pcd, pcd)
    pcd.transform(T)
    # color registration
    T = colored_icp(pcd,merged_pcd)
    pcd.transform(T)
    # stored pcd
    # REG_PCD_LIST.append(pcd)
    merged_pcd = merge_pcds([merged_pcd, pcd])
    merged_pcd.random_down_sample(0.8)
    # o3d.visualization.draw_geometries([merged_pcd])
    o3d.io.write_point_cloud("./outputs/simpleICP.ply", merged_pcd)
o3d.visualization.draw_geometries([merged_pcd])	



