import os
import re
import random
import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import open3d as o3d
import open3d.core as o3c

class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)

def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj

def show_rgbd(rgbd):
    plt.subplot(1, 2, 1)
    plt.title('Color image')
    plt.imshow(rgbd.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth image')
    plt.imshow(rgbd.depth)
    plt.show()

def generate_point_cloud(image_dir:str, depth_dir:str):
    color_raw = o3d.io.read_image(image_dir)
    depth_raw = o3d.io.read_image(depth_dir)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
    )
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd

def generate_point_cloud_with_camera_pose(image_dir:str, depth_dir:str, camera_pose):

    color = o3d.io.read_image(image_dir)
    depth = o3d.io.read_image(depth_dir)
    # random noise

    random.seed(time())
    camera_pose[0][3] += random.randint(-5000,5000) / 10000
    camera_pose[1][3] += random.randint(-5000,5000) / 10000
    camera_pose[2][3] += random.randint(-5000,5000) / 10000
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        # camera_pose,
    )
    pcd.transform(camera_pose)
    return pcd

def merge_pcds(pcd_list):
    tmp_points = np.concatenate(tuple([i.points for i in pcd_list]), axis=0)
    tmp_colors = np.concatenate(tuple([i.colors for i in pcd_list]), axis=0)
    tmp = o3d.geometry.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(tmp_points)
    tmp.colors = o3d.utility.Vector3dVector(tmp_colors)
    return tmp

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
    x_min, x_max = get_max_distance(min_x0, max_x0, min_x1, max_x1)
    y_min, y_max = get_max_distance(min_y0, max_y0, min_y1, max_y1)
    z_min, z_max = get_max_distance(min_z0, max_z0, min_z1, max_z1)
    return np.array([(x_max + x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2]),np.array([(x_max - x_min)/2, (y_max - y_min)/2, (z_max - z_min)/2])

def ply_double_to_float(path:str):
    try:
        pcd0 = o3d.t.io.read_point_cloud(path)
        pcd0.point["positions"]= o3c.Tensor(pcd0.point["positions"].numpy().astype('float32'))
        o3d.t.io.write_point_cloud(path, pcd0, compressed=True)
        return True
    except:
        return False

def color_icp_cpp(pcd_trans, pcd_base):
    T = np.identity(4)
    print("=> Start color icp in cpp")
    # save temp ply file
    o3d.io.write_point_cloud("./color_icp/data/pcd_trans.ply", pcd_trans)
    o3d.io.write_point_cloud("./color_icp/data/pcd_base.ply", pcd_base)
    # set ply headers' format
    ply_double_to_float("./color_icp/data/pcd_trans.ply")
    ply_double_to_float("./color_icp/data/pcd_base.ply")
    # run color icp
    os.chdir("./color_icp/build/")
    os.system("./color_icp")
    print("=> Colored ICP finished")
    # read results
    os.chdir("../../")
    T_file = open('color_icp/data/Estimated_transformation.txt','r').read()
    T_lines = T_file.split('\n')
    T = np.asarray([[float(num) for num in re.split('[ ]+',line.strip(' '))] for line in T_lines])
    return T
