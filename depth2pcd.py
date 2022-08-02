import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import open3d as o3d

def generate_point_cloud(pic1:str,pic2:str):
    # print("Read Redwood dataset")
    color_raw = o3d.io.read_image(pic1)
    depth_raw = o3d.io.read_image(pic2)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
#         o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault),
#         project_valid_depth_only=False
    )
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd
#     o3d.io.write_point_cloud("sample.ply", pcd)

def concate_pcds(pcd0, pcd1):

    tmp_points = np.concatenate((pcd0.points,pcd1.points), axis=0)
    tmp_colors = np.concatenate((pcd0.colors,pcd1.colors), axis=0)
    tmp = o3d.geometry.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(tmp_points)
    tmp.colors = o3d.utility.Vector3dVector(tmp_colors)
    
    return tmp

def fusion_pcds(pcd_base, pcd_add, T_base, T_add):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    color = o3d.io.read_image("./data/redwood/image/{}.jpg".format(pcd_base))
    depth = o3d.io.read_image("./data/redwood/depth/{}.png".format(pcd_base))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)

    volume.integrate(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
            np.linalg.inv(T_base)
            # T_base
        )

    color = o3d.io.read_image("./data/redwood/image/{}.jpg".format(pcd_add))
    depth = o3d.io.read_image("./data/redwood/depth/{}.png".format(pcd_add))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)

    volume.integrate(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
            np.linalg.inv(T_add)
            # T_add
        )
    
    pcd = volume.extract_point_cloud()
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd