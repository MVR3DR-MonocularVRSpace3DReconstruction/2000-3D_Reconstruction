
import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import open3d as o3d

from camera_pose import read_trajectory


STEP = 10

camera_poses = read_trajectory("./data/redwood/livingroom1-traj.txt")

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=4.0 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

for i in range(0,len(camera_poses),STEP):
    print("Integrate {:d}-th image into the volume.".format(i))
    color = o3d.io.read_image("./data/redwood/image/{:05d}.jpg".format(i))
    depth = o3d.io.read_image("./data/redwood/depth/{:05d}.png".format(i))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
    volume.integrate(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        np.linalg.inv(camera_poses[i].pose))

pcd = volume.extract_point_cloud()
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("./tmp/output_camera_pose.ply", pcd)

# print("Extract a triangle mesh from the volume and visualize it.")
# mesh = volume.extract_triangle_mesh()
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh],
#                                   front=[0.5297, -0.1873, -0.8272],
#                                   lookat=[2.0712, 2.0312, 1.7251],
#                                   up=[-0.0558, -0.9809, 0.1864],
#                                   zoom=0.47)
