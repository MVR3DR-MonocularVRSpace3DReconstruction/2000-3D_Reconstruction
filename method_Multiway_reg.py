import os
from time import time
import numpy as np
import open3d as o3d

from tqdm import tqdm
from deep_global_registration.core.deep_global_registration import DeepGlobalRegistration
from deep_global_registration.config import get_config

from utils import *
from colored_icp import colored_icp
from overlap import overlap_predator

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
    # print("=> Start loading point clouds...")
    for i in tqdm(range(0, len(image_dir), step)):
        # print("=> Load [{}/{}] point cloud".format(i//step,len(image_dir)//step))
        pcd = generate_point_cloud(
                data_dir+'image/'+image_dir[i],
                data_dir+'depth/'+depth_dir[i],
                # camera_poses[i].pose
                )
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd.estimate_normals()
        pcds.append(pcd)
    # print("# Load {} point clouds from {}".format(len(pcds),data_dir))
    return pcds


###########################################################
# Pair Registration - point to plane 
###########################################################

# 0-len()//20 time cost: 0m20s

def icp_point2plane_registration(source, target):
    # print("=> Apply point-to-plane ICP")
    voxel_radius = [15*voxel_size, 3*voxel_size, 1.5*voxel_size]
    max_iter = [60, 35, 20]# [60, 35, 20]
    _transformation_icp = np.identity(4)
    for scale in range(3):
        max_it = max_iter[scale]
        radius = voxel_radius[scale]
        # print("=> scale_level = {0}, voxel_size = {1}, max_iter = {2}".format(scale, radius, max_it))
        # print("=> scale_level = {0}, voxel_size = {1}".format(scale, radius))
        result = o3d.pipelines.registration.registration_icp(
            source, target, radius, _transformation_icp,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                    relative_rmse=1e-6,
                                                                    max_iteration=max_it)
                                                                    )
        _transformation_icp = result.transformation
    return _transformation_icp
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
    # print("=> voxel size: {} // distance threshold: {}".format(voxel_size, distance_threshold))
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
    # print("=> Apply Global RANSAC ")
    ransac_result = execute_global_registration(source, target)
    _transformation_ransac = ransac_result.transformation
    _information_ransac = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, correspondence_distance,
        ransac_result.transformation)
    return _transformation_ransac, _information_ransac, True

###########################################################
# Color ICP Registration
###########################################################

# 0-len()//20 time cost: 1m13s

def colored_icp_registration(source, target):
    # print("=> Colored ICP registration")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    voxel_radius = [5*voxel_size, 3*voxel_size, voxel_size]
    max_iter = [60, 35, 20]# [60, 35, 20]
    _transformation_cicp = np.identity(4)
    for scale in range(3):
        max_it = max_iter[scale]
        radius = voxel_radius[scale]
        # print("=> scale_level = {0}, voxel_size = {1}, max_iter = {2}".format(scale, radius, max_it))
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
            _transformation_cicp = result.transformation                                                    
        except Exception as e:
            # print("=> No correspondence found. ")
            continue
    return _transformation_cicp

###########################################################
# Color ICP Registration
###########################################################

# 0-len()//20 time cost: 1m13s

def color_icp_registration_by_cpp(source, target):
    # print("=> Color ICP registration by CPP")
    _transformation_cicp_cpp = color_icp_cpp(source, target)
    _information_cicp_cpp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, correspondence_distance,
        _transformation_cicp_cpp)
    return _transformation_cicp_cpp, _information_cicp_cpp, True

###########################################################
# Deep Global Registration
###########################################################

# 0-len()//20 time cost: 1m24s

def deep_global_registration(source, target):
    # print("=> Apply Deep Global Reg ")
    # o3d.visualization.draw_geometries([source, target])
    if 'DGR' not in globals():
        config = get_config()
        config.weights = "deep_global_registration/pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
        global DGR
        DGR = DeepGlobalRegistration(config)
    source_down = copy.deepcopy(source)
    target_down = copy.deepcopy(target)
    source_down.voxel_down_sample(0.05)
    target_down.voxel_down_sample(0.05)
    source_down.estimate_normals()
    target_down.estimate_normals()
    _transformation_dgr, isGoodReg = DGR.register(source_down, target_down)
    return _transformation_dgr

###########################################################
# Combination Registration
###########################################################

def combination_registration(source, target):
    _transformation_overlap, _, certain = deep_global_registration(source, target)
    pcd = merge_pcds([source])
    pcd = pcd.transform(_transformation_overlap)
    transformation_cicp = colored_icp_registration(pcd, target)
    transformation = np.dot(transformation_cicp, _transformation_overlap)
    # print(transformation_cicp,'\n', transformation_dgr,'\n', transformation)
    # source.paint_uniform_color([1, 0, 0]) # Red
    # pcd.paint_uniform_color([0, 1, 0]) # Green
    # target.paint_uniform_color([0, 0, 1]) # Blue
    return transformation

###########################################################
# Multiway Full Registration
###########################################################

def full_registration(pcds, correspondence_range_ratio):
    print("===> Start FULL REG.")
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in tqdm(range(n_pcds)):
        for target_id in tqdm(range(source_id + 1,min(n_pcds, int(source_id + n_pcds * correspondence_range_ratio)))):      
            # print("==> Source: [{0}//{2}] with target: [{1}//{3}]-len:{4} reg...".format(
            #     source_id, target_id, n_pcds-1, min(n_pcds, int(source_id + n_pcds * correspondence_range_ratio)), int(n_pcds * correspondence_range_ratio)))
            # first reg
            # o3d.visualization.draw_geometries([pcds[source_id], pcds[target_id]])
            transformation = deep_global_registration(
                pcds[source_id], pcds[target_id])

            information= o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                pcds[source_id], pcds[target_id], correspondence_distance,
                transformation)

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

start_time = time()

voxel_size = 0.01
# pcds_down = load_point_clouds(
#     data_dir = "./data/redwood-livingroom/",
#     camera_pose_file = "livingroom.log",
#     step = 10,
#     voxel_size=voxel_size)

pcds_down = read_point_clouds(data_dir = "outputs/",down_sample=0.8)
# pcds_down = load_point_clouds()
pause_time = time()
o3d.visualization.draw_geometries(pcds_down)
pause_time = time() - pause_time

print("\n\n# Full registration ...")
correspondence_distance = voxel_size * 1.5 # 1.5
# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
pose_graph = full_registration(pcds_down, 0.3) # 15/len(pcds_down)                     
########

print("\n\n# Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=correspondence_distance,
    edge_prune_threshold=0.25,
    reference_node=0,# len(pcds_down)//2
    preference_loop_closure=2.0)


with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        # o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationGaussNewton(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
########

print("\n\n# Pose graph length: ",pose_graph)
print("\n\n# Transform points and display")
for point_id in tqdm(range(len(pcds_down))):
    # print(pose_graph.nodes[point_id].pose)
    pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)

end_time = time()
time_cost = end_time-start_time-pause_time
print("\n## Total cost {}s = {}m{}s.".format(
    time_cost, int((time_cost)//60), int(time_cost - (time_cost)//60*60)))

pcd_combined = merge_pcds(pcds_down)
o3d.io.write_point_cloud("./outputs/multiway_registration.ply", pcd_combined)
o3d.visualization.draw_geometries(pcds_down)