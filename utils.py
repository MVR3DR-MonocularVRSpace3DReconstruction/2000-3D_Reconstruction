import os
import re
import glob
import random
import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import open3d as o3d
import open3d.core as o3c
import copy

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

def load_point_clouds(data_dir = "data/redwood-livingroom/",
                      camera_pose_file = "livingroom.log",
                      step = 10,
                      down_sample=1):
    image_dir = sorted(glob.glob(data_dir+'image/*.jpg'))
    depth_dir = sorted(glob.glob(data_dir+'depth/*.png'))
    camera_poses = read_trajectory("{}{}".format(data_dir, camera_pose_file))

    pcds = []
    # print("=> Start loading point clouds...")
    for i in tqdm(range(0, len(image_dir), step), desc="Load point clouds"):
        # print("=> Load [{}/{}] point cloud".format(i//step,len(image_dir)//step))
        pcd = generate_point_cloud(
                image_dir[i],
                depth_dir[i],
                # camera_poses[i].pose
                )
        if down_sample != 1:
            pcd = pcd.random_down_sample(voxel_size=down_sample)
            pcd.estimate_normals()
        pcds.append(pcd)
    # print("# Load {} point clouds from {}".format(len(pcds),data_dir))
    return pcds
    
def read_point_clouds(data_dir = "./data/redwood-livingroom/",down_sample=0.1):
    pcds = []
    for pcd in tqdm(sorted(glob.glob(data_dir+'fragments/*.ply'))):
        temp_pcd = o3d.io.read_point_cloud(pcd)
        temp_pcd = temp_pcd.random_down_sample(down_sample)
        temp_pcd.estimate_normals()
        temp_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcds.append(temp_pcd)
    return pcds

def read_pose_graph(data_dir = "./data/redwood-livingroom/"):
    graphs = []
    for graph in tqdm(sorted(glob.glob(data_dir+'fragments/fragment_posegraph_opti_*.json'))):
        temp = o3d.io.read_pose_graph(graph)
        graphs.append(temp)
    return graphs

def read_rgbd_image(color_file, depth_file, convert_rgb_to_intensity):
    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        convert_rgb_to_intensity=convert_rgb_to_intensity)
    return rgbd_image

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
    # random.seed(time())
    # camera_pose[0][3] += random.randint(-5000,5000) / 100000
    # camera_pose[1][3] += random.randint(-5000,5000) / 100000
    # camera_pose[2][3] += random.randint(-5000,5000) / 100000
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        # camera_pose,
    )
    pcd.transform(camera_pose)
    return pcd

def merge_pcds(pcd_list):
    tmp_points = copy.deepcopy(np.concatenate(tuple([i.points for i in pcd_list]), axis=0))
    tmp_colors = copy.deepcopy(np.concatenate(tuple([i.colors for i in pcd_list]), axis=0))
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
