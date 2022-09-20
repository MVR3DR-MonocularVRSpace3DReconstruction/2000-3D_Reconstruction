from math import comb
import os
import time
from unittest import result
import numpy as np
import open3d as o3d

from core.deep_global_registration import DeepGlobalRegistration
from config import get_config

from simpleicp import PointCloud, SimpleICP
from utils import generate_point_cloud_with_camera_pose, merge_pcds, color_icp_cpp, read_trajectory, generate_point_cloud
from colored_icp import colored_icp

###########################################################
# Load Data
###########################################################

def load_point_clouds(data_dir = "./data/redwood-livingroom/",
                      camera_pose_file = "livingroom.log",
                      step = 10,
                      voxel_size=0.01):
    image_dir = sorted(os.listdir(data_dir+'image/'))
    depth_dir = sorted(os.listdir(data_dir+'depth/'))
    camera_poses = read_trajectory("{}{}".format(data_dir, camera_pose_file))

    pcds = []
    print("=> Start loading point clouds...")
    for i in range(0, len(image_dir), step):
        print("=> Load [{}/{}] point cloud".format(i//step,len(image_dir)//step))
        pcd = generate_point_cloud_with_camera_pose(
                data_dir+'image/'+image_dir[i],
                data_dir+'depth/'+depth_dir[i],
                camera_poses[i].pose
                )
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd.estimate_normals()
        pcds.append(pcd)
    print("# Load {} point clouds from {}".format(len(pcds),data_dir))
    return pcds

###########################################################
# Pair Registration - point to plane 
###########################################################

# 0-len()//20 time cost: 0m20s

def icp_point2plane_registration(source, target):
    print("=> Apply point-to-plane ICP")
    voxel_radius = [15*voxel_size, 3*voxel_size, 1.5*voxel_size]
    # max_iter = [60, 35, 20]# [60, 35, 20]
    _transformation_icp = np.identity(4)
    for scale in range(3):
        # max_it = max_iter[scale]
        radius = voxel_radius[scale]
        # print("=> scale_level = {0}, voxel_size = {1}, max_iter = {2}".format(scale, radius, max_it))
        print("=> scale_level = {0}, voxel_size = {1}".format(scale, radius))
        result = o3d.pipelines.registration.registration_icp(
            source, target, radius, _transformation_icp,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            # o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
            #                                                         relative_rmse=1e-6,
            #                                                         max_iteration=max_it)
                                                                    )
        _transformation_icp = result.transformation
    _information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, correspondence_distance,
        result.transformation)
    return _transformation_icp, _information_icp

###########################################################
# Global Registration - RANSAC
###########################################################

# 0-len()//20 time cost: 3m41s

def execute_global_registration(source_down, target_down):
    distance_threshold = voxel_size * 1.5

    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    print("=> voxel size: {} // distance threshold: {}".format(voxel_size, distance_threshold))
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def global_registration(source, target):
    print("=> Apply Global RANSAC ")
    ransac_result = execute_global_registration(source, target)
    _transformation_ransac = ransac_result.transformation
    _information_ransac = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, correspondence_distance,
        ransac_result.transformation)
    return _transformation_ransac, _information_ransac

###########################################################
# Color ICP Registration
###########################################################

# 0-len()//20 time cost: 1m13s

def colored_icp_registration(source, target):
    print("=> Colored ICP registration")
    voxel_radius = [5*voxel_size, 3*voxel_size, voxel_size]
    max_iter = [60, 35, 20]# [60, 35, 20]
    _transformation_cicp = np.identity(4)
    for scale in range(3):
        max_it = max_iter[scale]
        radius = voxel_radius[scale]
        print("=> scale_level = {0}, voxel_size = {1}, max_iter = {2}".format(scale, radius, max_it))
        try:
            result = o3d.pipelines.registration.registration_colored_icp(
                source, 
                target, 
                radius, 
                _transformation_cicp,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                    relative_rmse=1e-6,
                                                                    max_iteration=max_it))
        except Exception as e:
            print("=> No correspondence found. ")
            continue
        _transformation_cicp = result.transformation
    _information_cicp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, correspondence_distance,
        result.transformation)
    return _transformation_cicp, _information_cicp

###########################################################
# Color ICP Registration
###########################################################

# 0-len()//20 time cost: 1m13s

def color_icp_registration_by_cpp(source, target):
    print("=> Color ICP registration by CPP")
    _transformation_cicp_cpp = color_icp_cpp(source, target)
    _information_cicp_cpp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, correspondence_distance,
        _transformation_cicp_cpp)
    return _transformation_cicp_cpp, _information_cicp_cpp

###########################################################
# Deep Global Registration
###########################################################

# 0-len()//20 time cost: 1m24s

def deep_global_registration(source, target):
    print("=> Apply Deep Global Reg ")
    # o3d.visualization.draw_geometries([source, target])
    if 'DGR' not in globals():
        config = get_config()
        config.weights = "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
        global DGR
        DGR = DeepGlobalRegistration(config)
    _transformation_dgr, isGoodReg = DGR.register(source, target)
    _information_dgr = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, correspondence_distance,
        _transformation_dgr)
    # o3d.visualization.draw_geometries([source, target.transform(_transformation_dgr)])
    return _transformation_dgr, _information_dgr

###########################################################
# Combination Registration
###########################################################

def combination_registration(source, target):
    print("no transform")
    # o3d.visualization.draw_geometries([source, target])
    transformation_dgr, _ = deep_global_registration(source, target)
    # tmp = source
    print("dgr tranformed")
    # o3d.visualization.draw_geometries([source.transform(transformation_dgr), target])
    transformation_cicp, _ = color_icp_registration_by_cpp(source.transform(transformation_dgr), target)
    print("cicp transformed")
    # o3d.visualization.draw_geometries([source.transform(transformation_cicp), target])
    transformation = np.dot( transformation_dgr, transformation_cicp)
    # print(transformation_dgr,transformation_cicp,transformation,np.dot(transformation_dgr, transformation_cicp))
    # o3d.visualization.draw_geometries([tmp.transform(transformation), target])
    # input()
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, correspondence_distance,
        transformation)
    return transformation, information

###########################################################
# Multiway Full Registration
###########################################################

def full_registration(pcds, correspondence_range_ratio):
    print("===> Start FULL REG.")
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1,min(n_pcds, int(source_id + n_pcds * correspondence_range_ratio))):      
            print("==> Source: [{0}//{2}] with target: [{1}//{3}]-len:{4} reg...".format(
                source_id, target_id, n_pcds-1, min(n_pcds, int(source_id + n_pcds * correspondence_range_ratio)), int(n_pcds * correspondence_range_ratio)))
            # first reg
            # o3d.visualization.draw_geometries([pcds[source_id], pcds[target_id]])
            transformation, information = combination_registration(
                pcds[source_id], pcds[target_id])
            # o3d.visualization.draw_geometries([pcds[source_id].transform(transformation), pcds[target_id]])
            print("==> Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation,
                                                             information,
                                                             uncertain=True)) # False
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation,
                                                             information,
                                                             uncertain=True))

    print("===> FULL REG FIN!!!")
    return pose_graph

###########################################################
# Main process
###########################################################

start_time = time.time()

voxel_size = 0.01
pcds_down = load_point_clouds()

pause_time = time.time()
o3d.visualization.draw_geometries(pcds_down)
pause_time = time.time() - pause_time

print("Full registration ...")
correspondence_distance = voxel_size * 1.5 # 1.5
# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
pose_graph = full_registration(pcds_down, 15/len(pcds_down))                      
########

print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=correspondence_distance,
    edge_prune_threshold=0.25,
    reference_node=0,# len(pcds_down)//2
    preference_loop_closure=2.0)


with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as m:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        # o3d.pipelines.registration.GlobalOptimizationGaussNewton(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
########

print("Pose graph length: ",pose_graph)
print("Transform points and display")
for point_id in range(len(pcds_down)):
    # print(pose_graph.nodes[point_id].pose)
    pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)

end_time = time.time()
time_cost = end_time-start_time-pause_time
print("## Total cost {}s = {}m{}s.".format(
    time_cost, int((time_cost)//60), int(time_cost - (time_cost)//60*60)))
o3d.visualization.draw_geometries(pcds_down)

pcd_combined = merge_pcds(pcds_down)
o3d.io.write_point_cloud("./outputs/multiway_registration.ply", pcd_combined)
# o3d.visualization.draw_geometries([pcd_combined])
