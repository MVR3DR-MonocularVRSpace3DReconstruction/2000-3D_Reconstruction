

import copy
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from rtvec2extrinsic import transformation2AnglePos, angelPos2Transformation

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

print("=> Start Integrating...")

visual = False
if visual:
    vis = o3d.visualization.Visualizer()
    # vis.add_geometry()
    vis.create_window()

for i in tqdm(range(0,len(camera_poses),STEP)):
    # print("Integrate {:d}-th image into the volume.".format(i))
    color = o3d.io.read_image("{}image/{}".format(DATA_DIR, COLOR_LIST[i]))
    depth = o3d.io.read_image("{}depth/{}".format(DATA_DIR, DEPTH_LIST[i]))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
    volume.integrate(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        np.linalg.inv(camera_poses[i].pose))
    
    pcd_temp = volume.extract_point_cloud()
    pcd_temp.transform(angelPos2Transformation(0, -90, 0, 0, 0, 0))
    # pcd_temp.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if visual:
        vis.add_geometry(pcd_temp)
        vis.poll_events()
        vis.update_renderer()

if visual:
    vis.destroy_window()
    
print("=> Done!")
pcd = volume.extract_point_cloud()
o3d.io.write_point_cloud("./outputs/livingroom.ply", pcd)
o3d.visualization.draw_geometries([pcd])

