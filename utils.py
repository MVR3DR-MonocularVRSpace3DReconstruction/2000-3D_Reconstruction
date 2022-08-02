
import os
import numpy as np
import open3d as o3d
from camera_pose import read_trajectory
# from core.deep_global_registration import DeepGlobalRegistration


def display_inlier_outlier(cloud, ind):

    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)


    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# pcd = o3d.io.read_point_cloud("./tmp/tmp-02.ply")
# voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.03)
# # pcd = pcd.uniform_down_sample(voxel_size=0.03)
# cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
#                                                     std_ratio=2.0)
# display_inlier_outlier(voxel_down_pcd, ind)
# # o3d.visualization.draw_geometries([pcd])

# [[ 0.99956958  0.00933272 -0.02781266  0.01262829]
#  [-0.00588444  0.99256472  0.12157586  0.24534924]
#  [ 0.02874049 -0.12135987  0.99219242  0.02284139]
#  [ 0.          0.          0.          1.        ]]

# [[ 0.89232086 -8.44748547  1.00823318  1.00233291]
#  [ 6.79393713  0.98849955  6.0128668   1.58693927]
#  [ 0.99663805  6.7384582   0.88205877  0.76794377]
#  [ 0.          0.          0.          1.        ]]



camera_poses = read_trajectory("./data/redwood/livingroom1-traj.txt")
m0 = np.linalg.inv(camera_poses[0].pose)
m1 = np.linalg.inv(camera_poses[10].pose)

print(m0)
print('-'*10)
print(m1)
print('-'*10)
ans = np.divide(m1,m0)
ans[np.isnan(ans)] = 0
print(ans)