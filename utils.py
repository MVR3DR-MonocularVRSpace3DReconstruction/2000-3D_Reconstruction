import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import open3d as o3d
import open3d.core as o3c

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

def generate_point_cloud_with_matrix(pic1:str,pic2:str,M):
    color_raw = o3d.io.read_image(pic1)
    depth_raw = o3d.io.read_image(pic2)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        M
    )
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd

def merge_pcds(pcd_list):


    tmp_points = np.concatenate(tuple([i.points for i in pcd_list]), axis=0)
    tmp_colors = np.concatenate(tuple([i.colors for i in pcd_list]), axis=0)
    tmp = o3d.geometry.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(tmp_points)
    tmp.colors = o3d.utility.Vector3dVector(tmp_colors)

    # o3d.visualization.draw_geometries([tmp])
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



def get_max_distance(min0:float,max0:float,min1:float,max1:float):
    d = [max0 - min0,max0 - min1,max1 - min0,max1 - min1]
    # print(d)
    ans = d.index(max(d))
    if ans == 0 : return min0, max0
    if ans == 1 : return min1, max0
    if ans == 2 : return min0, max1
    if ans == 3 : return min1, max1
    return False

def get_max_box_corner(box_0,box_1):
    min_x0 = np.argmin(box_0[:,0])
    min_y0 = np.argmin(box_0[:,1])
    min_z0 = np.argmin(box_0[:,2])

    max_x0 = np.argmax(box_0[:,0])
    max_y0 = np.argmax(box_0[:,1])
    max_z0 = np.argmax(box_0[:,2])


    min_x1 = np.argmin(box_1[:,0])
    min_y1 = np.argmin(box_1[:,1])
    min_z1 = np.argmin(box_1[:,2])

    max_x1 = np.argmax(box_1[:,0])
    max_y1 = np.argmax(box_1[:,1])
    max_z1 = np.argmax(box_1[:,2])

    x_min, x_max = get_max_distance(min_x0, max_x0, min_x1, max_x1)
    y_min, y_max = get_max_distance(min_y0, max_y0, min_y1, max_y1)
    z_min, z_max = get_max_distance(min_z0, max_z0, min_z1, max_z1)
    return [[x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max],
        ] 

def get_max_box_center_extent(box_0,box_1):
    min_x0 = min(box_0[:,0])
    min_y0 = min(box_0[:,1])
    min_z0 = min(box_0[:,2])

    max_x0 = max(box_0[:,0])
    max_y0 = max(box_0[:,1])
    max_z0 = max(box_0[:,2])


    min_x1 = min(box_1[:,0])
    min_y1 = min(box_1[:,1])
    min_z1 = min(box_1[:,2])

    max_x1 = max(box_1[:,0])
    max_y1 = max(box_1[:,1])
    max_z1 = max(box_1[:,2])

    # print(max_x0,max_y0,max_z0)
    x_min, x_max = get_max_distance(min_x0, max_x0, min_x1, max_x1)
    y_min, y_max = get_max_distance(min_y0, max_y0, min_y1, max_y1)
    z_min, z_max = get_max_distance(min_z0, max_z0, min_z1, max_z1)
    # print(x_min, x_max,y_min, y_max,z_min, z_max)
    return np.array([(x_max + x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2]),np.array([(x_max - x_min)/2, (y_max - y_min)/2, (z_max - z_min)/2])

def ply_double_to_float(path:str):
    try:
        pcd0 = o3d.t.io.read_point_cloud(path)
        pcd0.point["positions"]= o3c.Tensor(pcd0.point["positions"].numpy().astype('float32'))
        o3d.t.io.write_point_cloud(path, pcd0, compressed=True)
        return True
    except:
        return False


