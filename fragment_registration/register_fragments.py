
import os
import sys
import copy
import numpy as np
import open3d as o3d

from deep_global_registration.config import get_config
from deep_global_registration.core.deep_global_registration import DeepGlobalRegistration

from fragment_registration.refine_registration import multiscale_icp
from fragment_registration.optimize_posegraph import optimize_posegraph_for_scene

from fragment_registration.open3d_utils import join, get_file_list, make_clean_folder, draw_registration_result

def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(
            point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds

def compute_overlap_ratio(pcd0, pcd1, trans, voxel_size):
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    matching01 = get_matching_indices(
        pcd0_down, pcd1_down, trans, voxel_size, 1)
    matching10 = get_matching_indices(pcd1_down, pcd0_down, np.linalg.inv(trans),
                                      voxel_size, 1)
    overlap0 = len(matching01) / len(pcd0_down.points)
    overlap1 = len(matching10) / len(pcd1_down.points)
    return max(overlap0, overlap1)

def preprocess_point_cloud(pcd, config):
    voxel_size = config["voxel_size"]
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                             max_nn=100))
    return (pcd_down, pcd_fpfh)

def register_point_cloud_fpfh(source, target, source_fpfh, target_fpfh, config):
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    distance_threshold = config["voxel_size"] * 1.4
    if config["global_registration"] == "dgr":

        transformation, _ = DGR.register(source, target)

        overlap_ratio = compute_overlap_ratio(
            source, target, transformation, config["voxel_size"])
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, distance_threshold, transformation)
        if overlap_ratio < 0.3:
            success = False
        else:
            success = True
        return (success, transformation, information)
    if config["global_registration"] == "fgr":
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
    if config["global_registration"] == "ransac":
        # Fallback to preset parameters that works better
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, False, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False), 4,
            [
                o3d.pipelines.registration.
                CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                1000000, 0.999))
    if (result.transformation.trace() == 4.0):
        return (False, np.identity(4), np.zeros((6, 6)))
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, distance_threshold, result.transformation)
    if information[5, 5] / min(len(source.points), len(target.points)) < 0.3:
        return (False, np.identity(4), np.zeros((6, 6)))
    return (True, result.transformation, information)

def compute_initial_registration(s, t, source_down, target_down, source_fpfh,
                                 target_fpfh, path_dataset, config):

    if t == s + 1:  # odometry case
        print("Using RGBD odometry")
        pose_graph_frag = o3d.io.read_pose_graph(
            join(path_dataset,
                 config["template_fragment_posegraph_optimized"] % s))
        n_nodes = len(pose_graph_frag.nodes)
        transformation_init = np.linalg.inv(pose_graph_frag.nodes[n_nodes -
                                                                  1].pose)
        (transformation, information) = \
            multiscale_icp(source_down, target_down,
                           [config["voxel_size"]], [50], config, transformation_init)
    else:  # loop closure case
        (success, transformation,
         information) = register_point_cloud_fpfh(source_down, target_down,
                                                  source_fpfh, target_fpfh,
                                                  config)
        if not success:
            print("No reasonable solution. Skip this pair")
            return (False, np.identity(4), np.zeros((6, 6)))
    print(transformation)

    if config["debug_mode"]:
        draw_registration_result(source_down, target_down, transformation)
    return (True, transformation, information)

def update_posegraph_for_scene(s, t, transformation, information, odometry, pose_graph):
    if t == s + 1:  # odometry case
        odometry = np.dot(transformation, odometry)
        odometry_inv = np.linalg.inv(odometry)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(odometry_inv))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                s, t, transformation,
                information, uncertain=False))
    else:  # loop closure case
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                s, t, transformation,
                information, uncertain=True))
    return (odometry, pose_graph)

def register_point_cloud_pair(ply_file_names, s, t, config):
    print("reading %s ..." % ply_file_names[s])
    source = o3d.io.read_point_cloud(ply_file_names[s])
    print("reading %s ..." % ply_file_names[t])
    target = o3d.io.read_point_cloud(ply_file_names[t])
    (source_down, source_fpfh) = preprocess_point_cloud(source, config)
    (target_down, target_fpfh) = preprocess_point_cloud(target, config)
    (success, transformation, information) = \
        compute_initial_registration(
        s, t, source_down, target_down,
        source_fpfh, target_fpfh, config["path_dataset"], config)
    if t != s + 1 and not success:
        return (False, np.identity(4), np.identity(6))
    if config["debug_mode"]:
        print(transformation)
        print(information)
    return (True, transformation, information)

class matching_result:
    
    def __init__(self, s, t):
        self.s = s
        self.t = t
        self.success = False
        self.transformation = np.identity(4)
        self.infomation = np.identity(6)

def make_posegraph_for_scene(ply_file_names, config):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    n_files = len(ply_file_names)
    matching_results = {}
    for s in range(n_files):
        for t in range(s + 1, n_files):
            matching_results[s * n_files + t] = matching_result(s, t)

    if config["global_registration"] == "dgr" and 'DGR' not in globals():
        dgr_config = get_config()
        dgr_config.weights = "deep_global_registration/pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
        global DGR
        DGR = DeepGlobalRegistration(dgr_config)

    if config["python_multi_threading_reg"] == True:
        from joblib import Parallel, delayed
        import multiprocessing
        import subprocess
        MAX_THREAD = min(multiprocessing.cpu_count(),
                         max(len(matching_results), 1))
        results = Parallel(n_jobs=MAX_THREAD)(delayed(
            register_point_cloud_pair)(ply_file_names, matching_results[r].s,
                                       matching_results[r].t, config)
            for r in matching_results)
        for i, r in enumerate(matching_results):
            matching_results[r].success = results[i][0]
            matching_results[r].transformation = results[i][1]
            matching_results[r].information = results[i][2]
    else:
        for r in matching_results:
            (matching_results[r].success, matching_results[r].transformation,
             matching_results[r].information) = register_point_cloud_pair(
                ply_file_names, matching_results[r].s, matching_results[r].t, config)

    for r in matching_results:
        if matching_results[r].success:
            (odometry, pose_graph) = update_posegraph_for_scene(
                matching_results[r].s, matching_results[r].t,
                matching_results[r].transformation,
                matching_results[r].information, odometry, pose_graph)
    o3d.io.write_pose_graph(config["path_dataset"]+config["template_global_posegraph"],pose_graph)

def run(config):
    print("register fragments.")
    ply_file_names = get_file_list(config["path_dataset"]+config["folder_fragment"]+".ply")
    make_clean_folder(config["path_dataset"]+config["folder_scene"])
    make_posegraph_for_scene(ply_file_names, config)
    optimize_posegraph_for_scene(config["path_dataset"], config)
