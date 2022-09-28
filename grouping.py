# Base package
import os
from tokenize import group
from turtle import shape
import numpy as np
import glob
from tqdm import tqdm

# Image package
from PIL import Image
from sewar.full_ref import uqi
import matplotlib.pyplot as plt

####################################################################
# Image Grouping
####################################################################
# data_dir = "data/redwood-livingroom/"
data_dir = "data/redwood-livingroom/"

# os.system("rm -rf outputs/fragments/")
# os.system("rm -rf outputs/posegraph/")
# os.system("mkdir outputs/fragments/")
# os.system("mkdir outputs/posegraph/")

image_names = sorted(glob.glob(data_dir+'image/*.jpg'))
depth_names = sorted(glob.glob(data_dir+'depth/*.png'))
print("=> Load Images.. ")
rgbds=[]
for idx in tqdm(range(len(image_names))):
    with Image.open(image_names[idx]) as img: 
        with Image.open(depth_names[idx]) as dep:
            img = np.array(img)
            dep = np.array(dep)
            rgbds.append([img, dep])
groups=[]
sid = 0
_skip_steps = 10
for tid in tqdm(range(0, len(rgbds), _skip_steps)):
    score = uqi(rgbds[sid][0], rgbds[tid][0])
    if score < 0.85:
        groups.append([sid, tid])
        sid = tid
groups.append([sid, len(rgbds)])
del rgbds

print("=> Align groups.. ")
temp_groups=[]
_extend_img_ratio = 5
for [sid, tid] in groups:
    length = tid - sid
    sid = max(0, sid-length//_extend_img_ratio)
    tid = min(len(image_names), tid+length//_extend_img_ratio)
    temp_groups.append([sid, tid])
# temp_groups = np.array(temp_groups)
groups=[]
for idx in range(len(temp_groups)):
    passShrink = True
    for iidx in range(idx+1, len(temp_groups)):
        if temp_groups[idx][0] > temp_groups[iidx][0] and temp_groups[idx][1] < temp_groups[iidx][1]:
            passShrink = False
    if passShrink:
        groups.append(temp_groups[idx])
groups = np.array(sorted(groups, key=lambda x:(x[0], x[1])))
print(groups)
print("Origin: {}, shrink to:{}".format(len(temp_groups), len(groups)))
####################################################################
# Point Clouds Fragments Process
####################################################################

import open3d as o3d
from utils import *
from deep_global_registration.core.deep_global_registration import DeepGlobalRegistration
from deep_global_registration.config import get_config
# check opencv python package

from fragment_registration.open3d_utils import initialize_opencv, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
with_opencv = initialize_opencv()
if with_opencv:
    from fragment_registration.opencv_pose_estimation import pose_estimation

####################################################################
# Optimize Pose Graph
####################################################################

def run_posegraph_optimization(pose_graph_name, pose_graph_optimized_name,
                               max_correspondence_distance,
                               preference_loop_closure):
    # to display messages from o3d.pipelines.registration.global_optimization
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(
    )
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance,
        edge_prune_threshold=0.25,
        preference_loop_closure=preference_loop_closure,
        reference_node=0)
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    o3d.pipelines.registration.global_optimization(pose_graph, method, criteria,
                                                   option)
    o3d.io.write_pose_graph(pose_graph_optimized_name, pose_graph)


def optimize_posegraph_for_fragment(sid, eid, max_correspondence_distence, preference_loop_closure):
    pose_graph_name = "outputs/posegraph/fragment_{:0>5}-{:0>5}.json".format(sid,eid)
    pose_graph_optimized_name = "outputs/posegraph/fragment_opti_{:0>5}-{:0>5}.json".format(sid,eid)
    run_posegraph_optimization(pose_graph_name, pose_graph_optimized_name,
            max_correspondence_distance = max_correspondence_distence,
            preference_loop_closure = preference_loop_closure)

####################################################################
# Make Fragments
####################################################################

def register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic, with_opencv):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    # source_rgbd_image = generate_point_cloud(color_files[s], depth_files[s])
    # target_rgbd_image = generate_point_cloud(color_files[t], depth_files[t])
    color = o3d.io.read_image(color_files[s])
    depth = o3d.io.read_image(depth_files[s])
    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        convert_rgb_to_intensity=False)
    color = o3d.io.read_image(color_files[t])
    depth = o3d.io.read_image(depth_files[t])
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        convert_rgb_to_intensity=False)
    option = o3d.pipelines.odometry.OdometryOption()
    # option.depth_diff_max = config["depth_diff_max"]
    if abs(s - t) != 1:
        if with_opencv:
            # print("-> Opencv Detected. ")
            success_5pt, odo_init = pose_estimation(source_rgbd_image,
                                                    target_rgbd_image,
                                                    intrinsic, False)
            if success_5pt:
                [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
                    source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    option)
                return [success, trans, info]
        return [False, np.identity(4), np.identity(6)]
    else:
        odo_init = np.identity(4)
        [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        return [success, trans, info]

def make_posegraph_for_fragment(sid, eid, 
                                color_files, depth_files,
                                intrinsic, with_opencv):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    trans_odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(trans_odometry))
    for s in range(sid, eid):
        for t in range(s + 1, eid):
            # odometry
            if t == s + 1:
                print("=> Matching Fragment [%05d-%05d] RGBD between frame [%d:%d]" % (sid, eid, s, t))
                [success, trans, info] = register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic, with_opencv)
                trans_odometry = np.dot(trans, trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        trans_odometry_inv))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(s,
                                                             t,
                                                             trans,
                                                             info,
                                                             uncertain=False))
    o3d.io.write_pose_graph("outputs/posegraph/fragment_{:0>5}-{:0>5}.json".format(sid,eid), pose_graph)

def integrate_rgb_frames_for_fragment(sid, eid,
                                    color_files, depth_files, 
                                    intrinsic, tsdf_cubic_size):
    pose_graph = o3d.io.read_pose_graph("outputs/posegraph/fragment_opti_{:0>5}-{:0>5}.json".format(sid,eid))
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=tsdf_cubic_size / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i in range(sid, eid):
        print("=> INTEGRATE Fragment <%05d-%05d> rgbd frame [%d/%d]" % (sid, eid, i-sid, eid-sid))
        rgbd = read_rgbd_image(color_files[i], depth_files[i], False)
        pose = pose_graph.nodes[i-sid].pose
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh

def make_pointcloud_for_fragment(sid, eid, color_files, depth_files, intrinsic):
    mesh = integrate_rgb_frames_for_fragment(sid, eid, color_files, depth_files, intrinsic, 3.0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd_name = "outputs/fragments/fragment_{:0>5}-{:0>5}.ply".format(sid,eid)
    o3d.io.write_point_cloud(pcd_name, pcd, False, True)

####################################################################
# Point Clouds Main Process
####################################################################

def process_single_fragment(sid, eid, image_names, depth_names, intrinsic):
    if intrinsic != "":
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsic)
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    make_posegraph_for_fragment(sid, eid, 
                                image_names, depth_names,
                                intrinsic, with_opencv)
    optimize_posegraph_for_fragment(sid, eid, 0.07, 0.1)
    make_pointcloud_for_fragment(sid, eid, 
                                image_names, depth_names,
                                intrinsic)

n_fragments = len(groups)

# if True:
#     from joblib import Parallel, delayed
#     import multiprocessing
#     import subprocess
#     MAX_THREAD = min(multiprocessing.cpu_count(), n_fragments)
#     Parallel(n_jobs=MAX_THREAD)(delayed(process_single_fragment)(sid, eid, image_names, depth_names, "") for [sid, eid] in groups)
# else:
#     for [sid, eid] in groups:
#         process_single_fragment(sid, eid, image_names, depth_names, "")

####################################################################
# Fragments Registration
####################################################################
from deep_global_registration.core.deep_global_registration import DeepGlobalRegistration
from deep_global_registration.config import get_config

def deep_global_registration(source, target):
    # print("=> Apply Deep Global Reg ")
    if 'DGR' not in globals():
        config = get_config()
        config.weights = "deep_global_registration/pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
        global DGR
        DGR = DeepGlobalRegistration(config)
    _transformation_dgr, _ = DGR.register(source, target)
    return _transformation_dgr


def colored_icp_registration(source, target, voxel_size):
    print("=> Colored ICP registration")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
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
            _transformation_cicp = result.transformation                                                    
        except Exception as e:
            print("=> No correspondence found. ")
            continue
    return _transformation_cicp

def get_transformation_from_correspondence_fragment(sf_range, tf_range):
    if tf_range[0] > sf_range[1]:
        print("=> Error: Two Fragments No Overlapped !!!")
        return None
    sf_name = "outputs/posegraph/fragment_opti_{:0>5}-{:0>5}.json".format(*sf_range)
    sf_pose = o3d.io.read_pose_graph(sf_name)
    tf_name = "outputs/posegraph/fragment_opti_{:0>5}-{:0>5}.json".format(*tf_range)
    tf_pose = o3d.io.read_pose_graph(tf_name)

    sf_trans = sf_pose.nodes[tf_range[0]-sf_range[0]].pose
    tf_trans = tf_pose.nodes[tf_range[0]-tf_range[0]].pose
    print(sf_trans,'\n',tf_trans)
    T_cal = np.dot(np.linalg.inv(tf_trans), sf_trans)
    # T = sf_trans
    sf_pcd_name = "outputs/fragments/fragment_{:0>5}-{:0>5}.ply".format(*sf_range)
    tf_pcd_name = "outputs/fragments/fragment_{:0>5}-{:0>5}.ply".format(*tf_range)
    sf_pcd = o3d.io.read_point_cloud(sf_pcd_name)
    tf_pcd = o3d.io.read_point_cloud(tf_pcd_name)
    tf_pcd.transform(T_cal)
    T_global_reg = deep_global_registration(tf_pcd, sf_pcd)
    tf_pcd.transform(T_global_reg)
    T_color_reg = colored_icp_registration(tf_pcd, sf_pcd, 0.05)
    T_trans = np.dot(T_cal, np.dot(T_global_reg, T_color_reg))
    return T_trans
print("=> Loading pose graphs...")

graphs = []
pose_graph_files = sorted(glob.glob('outputs/posegraph/fragment_opti_*.json'))
# print(pose_graph_files)
for graph in tqdm(pose_graph_files):
    temp = o3d.io.read_pose_graph(graph)
    graphs.append(temp)


# intrinsic = o3d.camera.PinholeCameraIntrinsic(
#             o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
# volume = o3d.pipelines.integration.ScalableTSDFVolume(
#     voxel_length=3.0 / 512.0,
#     sdf_trunc=0.04,
#     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
T = np.identity(4)
pcds = []
for fragment_idx in tqdm(range(0, n_fragments), desc="Load fragment"):
    
    # get transformation with next fragment
    if fragment_idx != 0:
        source_fragment_range = groups[fragment_idx-1]
        target_fragment_range = groups[fragment_idx]
        T_trans = get_transformation_from_correspondence_fragment(source_fragment_range, target_fragment_range)
    else: 
        source_fragment_range = groups[fragment_idx]
        target_fragment_range = groups[fragment_idx]
        # T_local = np.identity(4)
        T_trans = np.identity(4)
    print("T local:\n",T_trans)
    print("T global:\n",T)
    
    # integrate frames from current fragment
    sf_pcd_name = "outputs/fragments/fragment_{:0>5}-{:0>5}.ply".format(*source_fragment_range)
    tf_pcd_name = "outputs/fragments/fragment_{:0>5}-{:0>5}.ply".format(*target_fragment_range)
    sf_pcd = o3d.io.read_point_cloud(sf_pcd_name)
    tf_pcd = o3d.io.read_point_cloud(tf_pcd_name)
    T = np.dot(T, T_trans)
    # tf_pcd.transform(T_global)
    tf_pcd.transform(T)
    # T_global = T
    # for idx in range(*target_fragment_range):
    #     print("\n=> INTEGRATE {} to volume from Fragment_{}-{}".format(idx, *target_fragment_range))
    #     idx_local = idx-target_fragment_range[0]
    #     print("# graph len:{} idx:{}".format(len(graphs[fragment_idx].nodes), idx_local))
    #     print("# Fid:{} \ngraph:{} \ngraph_files:{}\nrange:{}".format(fragment_idx, graphs[fragment_idx], pose_graph_files[fragment_idx],target_fragment_range))
    #     print("# surround graph: \n{}".format(groups[max(fragment_idx-5,0):min(fragment_idx+5,len(groups))]))
    #     pose = np.dot(T_global, graphs[fragment_idx].nodes[idx_local].pose, )
    #     image = o3d.io.read_image(image_names[idx])
    #     depth = o3d.io.read_image(depth_names[idx])
    #     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #         image,
    #         depth,
    #         convert_rgb_to_intensity=False)
    #     volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

    # pcd = volume.extract_point_cloud()
    pcds.append(tf_pcd)
    o3d.visualization.draw_geometries(pcds)
# mesh = volume.extract_triangle_mesh()
# mesh.compute_vertex_normals()
# pcd = o3d.geometry.PointCloud()
# pcd.points = mesh.vertices
# pcd.colors = mesh.vertex_colors
pcd = merge_pcds(pcds)
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd_name = "outputs/fragments/MAIN.ply"
o3d.io.write_point_cloud(pcd_name, pcd, False, True)
o3d.visualization.draw_geometries([pcd])