
import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import open3d as o3d

import os

from utils import read_trajectory

DATA_DIR = "./data/redwood-livingroom/"
POSE_FILE = "livingroom.log"
COLOR_LIST = sorted(os.listdir(DATA_DIR+'image/'))
DEPTH_LIST = sorted(os.listdir(DATA_DIR+'depth/'))

STEP = 10

camera_poses = read_trajectory("{}{}".format(DATA_DIR, POSE_FILE))

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=4.0 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

for i in range(0,len(camera_poses),STEP):
    print("Integrate {:d}-th image into the volume.".format(i))
    # print("{}image/{}".format(DATA_DIR, COLOR_LIST[i]))
    color = o3d.io.read_image("{}image/{}".format(DATA_DIR, COLOR_LIST[i]))
    depth = o3d.io.read_image("{}depth/{}".format(DATA_DIR, DEPTH_LIST[i]))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
    volume.integrate(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        np.linalg.inv(camera_poses[i].pose))

pcd = volume.extract_point_cloud()
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("./tmp/tmp_camera_pose.ply", pcd)

# print("Extract a triangle mesh from the volume and visualize it.")
# mesh = volume.extract_triangle_mesh()
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh],
#                                   front=[0.5297, -0.1873, -0.8272],
#                                   lookat=[2.0712, 2.0312, 1.7251],
#                                   up=[-0.0558, -0.9809, 0.1864],
#                                   zoom=0.47)
