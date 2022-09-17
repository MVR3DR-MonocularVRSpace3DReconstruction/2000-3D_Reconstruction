from simpleicp import PointCloud, SimpleICP
import numpy as np
import random

import os
import open3d as o3d
from utils import generate_point_cloud, merge_pcds, color_icp_cpp, read_trajectory
from colored_icp import colored_icp
# boardroom
DATA_DIR = "./data/redwood-livingroom/"
POSE_FILE = "livingroom.log"
COLOR_LIST = sorted(os.listdir(DATA_DIR+'image/'))
DEPTH_LIST = sorted(os.listdir(DATA_DIR+'depth/'))
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

camera_poses = read_trajectory("{}{}".format(DATA_DIR, POSE_FILE))



volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=4.0 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

color = o3d.io.read_image("{}image/{}".format(DATA_DIR, COLOR_LIST[0]))
depth = o3d.io.read_image("{}depth/{}".format(DATA_DIR, DEPTH_LIST[0]))
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)

volume.integrate(
    rgbd,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
    np.linalg.inv(camera_poses[0].pose))
pcd = volume.extract_point_cloud()



PCD_LIST = [pcd]
for i in range(0,len(COLOR_LIST),STEP):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    color = o3d.io.read_image("{}image/{}".format(DATA_DIR, COLOR_LIST[i]))
    depth = o3d.io.read_image("{}depth/{}".format(DATA_DIR, DEPTH_LIST[i]))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
    
    # random noise
    print(camera_poses[i].pose)
    random.seed(i)
    camera_poses[i].pose[0][3] += random.randint(-5000,5000) / 10000
    camera_poses[i].pose[1][3] += random.randint(-5000,5000) / 10000
    camera_poses[i].pose[2][3] += random.randint(-5000,5000) / 10000
    print(camera_poses[i].pose)

    volume.integrate(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        np.linalg.inv(camera_poses[i].pose))
    pcd = volume.extract_point_cloud()
    # PCD_LIST.append(pcd)
    tmp = PCD_LIST
    tmp.append(pcd)
    o3d.visualization.draw_geometries(tmp)

    pcd_base = PCD_LIST[-1]
    pcd.estimate_normals()
    print('=> Registration..')
    # registration
    T = simpleICP(pcd_base, pcd)
    pcd.transform(T)
    # color registration
    T = colored_icp(pcd,pcd_base)
    pcd.transform(T)
    # stored pcd
    pcd_base.random_down_sample(0.8)
    # o3d.visualization.draw_geometries([pcd_base])
    PCD_LIST.append(pcd)
    o3d.visualization.draw_geometries(PCD_LIST)
    o3d.io.write_point_cloud("./tmp/simpleICP.ply", pcd_base)


PCD_LIST = [generate_point_cloud(
                DATA_DIR+'image/'+COLOR_LIST[i],
                DATA_DIR+'depth/'+DEPTH_LIST[i]
			)  for i in range(0,len(COLOR_LIST),STEP)]
print("=> PCD_LIST generated")

# REG_PCD_LIST = [PCD_LIST[0]]

count = 0



merged_pcd = PCD_LIST[0]
for pcd in PCD_LIST:
    print("==> Phase: {}// with {} times reg in total".format(count,len(PCD_LIST)))
    count += 1
    # get base from registrated pcd list
    # preprocessing
    # pcd_base.estimate_normals()
    pcd.estimate_normals()
    print('=> Registration..')
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
    o3d.io.write_point_cloud("./tmp/simpleICP.ply", merged_pcd)
o3d.visualization.draw_geometries([merged_pcd])	



