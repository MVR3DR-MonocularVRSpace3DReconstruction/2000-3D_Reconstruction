# Base package
import os
from turtle import width
import numpy as np
import glob
from tqdm import tqdm

# Image package
from PIL import Image
from sewar.full_ref import uqi, ssim, msssim
import matplotlib.pyplot as plt

import cv2

####################################################################
# Image Grouping
####################################################################
# data_dir = "data/redwood-livingroom/"
data_dir = "data/redwood-livingroom/"
image_names = sorted(glob.glob(data_dir+'image/*.jpg'))
depth_names = sorted(glob.glob(data_dir+'depth/*.png'))
print("==> Evaluate Images.. ")

def make_fragments_groups(image_names, skip_steps = 100, extend_img_ratio = 5, theroshold = 0.85):
    # fresh documents
    os.system("rm -rf outputs/fragments/")
    os.system("rm -rf outputs/posegraph/")
    os.system("mkdir outputs/fragments/")
    os.system("mkdir outputs/posegraph/")

    # evaluate images & grouping
    print("=> Cluster images")
    groups=[]
    sid = 0
    source = np.array(Image.open(image_names[sid]).convert('L'))
    for tid in tqdm(range(0, len(image_names), skip_steps)):
        target = np.array(Image.open(image_names[tid]).convert('L'))
        score = uqi(source, target)
        # print(score, sid, tid)
        if score < theroshold:
            groups.append([sid, tid]) # need fix range in [sid, tid)
            sid = tid
            source = np.array(Image.open(image_names[sid]).convert('L'))
    if groups[-1][1] != len(image_names)-1:
        groups.append([sid, len(image_names)-1])
            
            
            
    # if groups[-1][1] != len(image_names)-1:
    #     groups.append([sid, len(image_names)-1])

    # extend groups with overlap frames
    print("=> Align groups.. ")
    temp_groups=[]
    for [sid, tid] in tqdm(groups):
        length = tid - sid + 1
        sid = max(0, sid-length//extend_img_ratio)
        tid = min(len(image_names)-1, tid+length//extend_img_ratio)
        temp_groups.append([sid, tid+1])  # fix range in [sid, tid)
        
        
    # remove useless groups which were full-covered by other groups
    print("=> Remove useless groups")
    groups=[]
    for idx in tqdm(range(len(temp_groups))):
        passShrink = True
        for iidx in range(idx+1, len(temp_groups)):
            if temp_groups[idx][0] > temp_groups[iidx][0] and temp_groups[idx][1] < temp_groups[iidx][1]:
                passShrink = False
                break
        if passShrink:
            groups.append(temp_groups[idx])
    groups = np.array(sorted(groups, key=lambda x:(x[0], x[1])))
    print(groups)
    print("Origin: {}, shrink to:{}".format(len(temp_groups), len(groups)))
    # store groups
    np.savetxt("outputs/posegraph/groups.txt",groups)
    # input()
    return groups

# groups = np.loadtxt("outputs/posegraph/groups.txt")
# groups = [[int(group[0]), int(group[1])] for group in groups] 
steps = len(image_names) // 50
groups = make_fragments_groups(image_names, skip_steps=steps, extend_img_ratio=5, theroshold=1)
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
                                intrinsic, with_opencv, voxel_size):
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

    
    print("=> Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel_size * 1.5,
        edge_prune_threshold=0.25,
        reference_node=0,# len(pcds_down)//2
        preference_loop_closure=2.0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        # o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationGaussNewton(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
    o3d.io.write_pose_graph("outputs/posegraph/fragment_{:0>5}-{:0>5}.json".format(sid,eid), pose_graph)
####################################################################
# Integrate Point Clouds
####################################################################
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

def make_pointcloud_for_fragment(sid, eid, color_files, depth_files, intrinsic, use_loss, loss_ratio):
    mesh = integrate_rgb_frames_for_fragment(sid, eid, color_files, depth_files, intrinsic, 3.0)
    # to point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors

    if use_loss:
        # delete noise according to loss ratio
        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=int(1e1), print_progress=True))
        clusters = sorted(list(set(labels)))
        clusters_idx = [[idx for idx in range(len(labels)) if labels[idx] == cluster] for cluster in clusters]
        del_clusters = []
        for idx in range(len(clusters_idx)):
            if len(clusters_idx[idx]) < len(pcd.points) * loss_ratio:
                del_clusters+=clusters_idx[idx]
        print(del_clusters)
        pcd_filter = o3d.geometry.PointCloud()
        pcd_filter.points = o3d.utility.Vector3dVector(np.delete(np.array(pcd.points), del_clusters, 0))
        pcd_filter.colors = o3d.utility.Vector3dVector(np.delete(np.array(pcd.colors), del_clusters, 0))
        pcd = pcd_filter

    # o3d.visualization.draw_geometries([pcd_filter])
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd_name = "outputs/fragments/fragment_{:0>5}-{:0>5}.ply".format(sid,eid)
    o3d.io.write_point_cloud(pcd_name, pcd, False, True)


def process_single_fragment(sid, eid, image_names, depth_names, intrinsic, voxel_size):
    if intrinsic != "":
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsic)
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    print("=> Make posegraph")
    make_posegraph_for_fragment(sid, eid, 
                                image_names, depth_names,
                                intrinsic, with_opencv, voxel_size)
    print("=> Optimize posegraph")
    optimize_posegraph_for_fragment(sid, eid, 0.07, 0.1)
    print("=> Make pointcloud")
    make_pointcloud_for_fragment(sid, eid, 
                                image_names, depth_names,
                                intrinsic, True, 0.01)

####################################################################
# Point Clouds Main Process
####################################################################
n_fragments = len(groups)

print("==> Make Fragments to point cloud")
if True:
    from joblib import Parallel, delayed
    import multiprocessing
    import subprocess
    MAX_THREAD = min(multiprocessing.cpu_count(), n_fragments)
    Parallel(n_jobs=MAX_THREAD)(delayed(process_single_fragment)(sid, eid, image_names, depth_names, "", 0.05) for [sid, eid] in groups)
else:
    for [sid, eid] in groups:
        process_single_fragment(sid, eid, image_names, depth_names, "", 0.05)

####################################################################
# Fragments Registration
####################################################################

from deep_global_registration.core.deep_global_registration import DeepGlobalRegistration
from deep_global_registration.config import get_config
from overlap import overlap_predator
from rtvec2extrinsic import *


def execute_global_registration(source_down, target_down, voxel_size):
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
    return result.transformation


def deep_global_registration(source, target, model='3dmatch'):
    print("# Apply Deep Global Reg ")
    if model=='3dmatch' and 'DGR_3dm' not in globals():
        config = get_config()
        # ResUNetBN2C-feat32-3dmatch-v0.05.pth   ResUNetBN2C-feat32-kitti-v0.3.pth   acc.875.pth
        config.weights = "deep_global_registration/pth/acc.875.pth"
        global DGR_3dm
        DGR_3dm = DeepGlobalRegistration(config)
    if model=='kitti' and 'DGR_kitti' not in globals():
        config = get_config()
        # ResUNetBN2C-feat32-3dmatch-v0.05.pth   ResUNetBN2C-feat32-kitti-v0.3.pth
        config.weights = "deep_global_registration/pth/ResUNetBN2C-feat32-kitti-v0.3.pth"
        global DGR_kitti
        DGR_kitti = DeepGlobalRegistration(config)
    
    if model == '3dmatch':
        _transformation_dgr, useSafeGuard = DGR_3dm.register(source, target)
    if model == 'kitti' :
        _transformation_dgr, useSafeGuard = DGR_kitti.register(source, target)
    return _transformation_dgr, useSafeGuard

def colored_icp_registration(source, target, voxel_size):
    print("# Colored ICP registration")
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

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    H = np.matmul(np.transpose(AA),BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = np.matmul(Vt.T,U.T)

    t = -np.matmul(R, centroid_A) + centroid_B
    # err = B - np.matmul(A,R.T) - t.reshape([1, 3])
    T = np.c_[R, t]
    T = np.r_[T, [[0, 0, 0, 1]]]
    return T

def get_transformation_from_correspondence_fragment(sf_range, tf_range):
    assert tf_range[0] <= sf_range[1]
    
    sf_name = "outputs/posegraph/fragment_opti_{:0>5}-{:0>5}.json".format(*sf_range)
    tf_name = "outputs/posegraph/fragment_opti_{:0>5}-{:0>5}.json".format(*tf_range)
    sf_pose = o3d.io.read_pose_graph(sf_name)
    tf_pose = o3d.io.read_pose_graph(tf_name)

    print("=> get transformation from correspondence fragment")
    # print(f"=> sf range {sf_range}   tf range {tf_range}")
    # print(f"=> sf: {tf_range[0] - sf_range[0]}  tf: {tf_range[0] - tf_range[0]}")

    sf_pos = []; tf_pos = []
    for idx in range(tf_range[0], sf_range[1]):
        if idx-tf_range[0] == len(tf_pose.nodes):
            break 
        sf_trans = sf_pose.nodes[idx-sf_range[0]].pose
        tf_trans = tf_pose.nodes[idx-tf_range[0]].pose
        _, _, _, rx, ry, rz = transformation2AnglePos(sf_trans)
        sf_pos.append([rx, ry, rz ])
        _, _, _, rx, ry, rz = transformation2AnglePos(tf_trans)
        tf_pos.append([rx, ry, rz ])
        # T_cal = np.dot(np.linalg.inv(tf_trans), sf_trans)
    T_rigid = rigid_transform_3D(np.array(tf_pos), np.array(sf_pos))
    sf_trans = sf_pose.nodes[tf_range[0]-sf_range[0]].pose
    tf_trans = tf_pose.nodes[tf_range[0]-tf_range[0]].pose
    T_cal = np.dot(np.linalg.inv(tf_trans), sf_trans)
    print("%.5f %.5f %.5f %.5f %.5f %.5f " % transformation2AnglePos(T_rigid))
    print("%.5f %.5f %.5f %.5f %.5f %.5f " % transformation2AnglePos(T_cal))
    
    # T = sf_trans
    sf_pcd_name = "outputs/fragments/fragment_{:0>5}-{:0>5}.ply".format(*sf_range)
    tf_pcd_name = "outputs/fragments/fragment_{:0>5}-{:0>5}.ply".format(*tf_range)

    
    sf_pcd = o3d.io.read_point_cloud(sf_pcd_name)
    tf_pcd = o3d.io.read_point_cloud(tf_pcd_name)
    tf_pcd.transform(T_rigid)
    # o3d.visualization.draw_geometries([sf_pcd, tf_pcd])
    
    print("=> global registration refine")


    sf_pcd_down = copy.deepcopy(sf_pcd)
    tf_pcd_down = copy.deepcopy(tf_pcd)
    sf_pcd_down.voxel_down_sample(0.05)
    tf_pcd_down.voxel_down_sample(0.05)
    sf_pcd_down.estimate_normals()
    tf_pcd_down.estimate_normals()

    T_global_reg, useSageGuard = deep_global_registration(tf_pcd_down, sf_pcd_down, '3dmatch')
    if useSageGuard:
        o3d.visualization.draw_geometries([sf_pcd, tf_pcd])
        sf_pcd_kitti = copy.deepcopy(sf_pcd)
        tf_pcd_kitti = copy.deepcopy(tf_pcd)
        
        # sf_pcd_kitti.voxel_down_sample(0.3)
        # tf_pcd_kitti.voxel_down_sample(0.3)
        sf_pcd_kitti.estimate_normals()
        tf_pcd_kitti.estimate_normals()

        T_global_reg = overlap_predator(tf_pcd_kitti, sf_pcd_kitti)
    
    tf_pcd.transform(T_global_reg)
    tf_pcd_down.transform(T_global_reg)
    # o3d.visualization.draw_geometries([sf_pcd, tf_pcd])
    
    print("=> color icp registration deep refine")
    sf_pcd_down = copy.deepcopy(sf_pcd)
    tf_pcd_down = copy.deepcopy(tf_pcd)
    sf_pcd_down.voxel_down_sample(0.05)
    tf_pcd_down.voxel_down_sample(0.05)
    sf_pcd_down.estimate_normals()
    tf_pcd_down.estimate_normals()
    T_color_reg = colored_icp_registration(tf_pcd_down, sf_pcd_down, 0.05)
    tf_pcd.transform(T_color_reg)
    tf_pcd_down.transform(T_color_reg)

    T_trans = np.dot(T_rigid, np.dot(T_global_reg, T_color_reg))
    # o3d.visualization.draw_geometries([sf_pcd, tf_pcd])
    
    return T_trans


####################################################################
# Merge Fragments Main Process
####################################################################

print("=> Loading pose graphs...")

graphs = []
pose_graph_files = sorted(glob.glob('outputs/posegraph/fragment_opti_*.json'))
# print(pose_graph_files)
for graph in tqdm(pose_graph_files):
    temp = o3d.io.read_pose_graph(graph)
    graphs.append(temp)

T_global = np.identity(4)

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=3.0 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    
pcds = []
pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.VisualizerWithEditing()
vis.add_geometry(pcd)
vis.create_window()


for fragment_idx in tqdm(range(0, n_fragments), desc="Load fragment"):
    
    # get transformation with next fragment
    if fragment_idx != 0:
        source_fragment_range = groups[fragment_idx-1]
        target_fragment_range = groups[fragment_idx]
        T_trans = get_transformation_from_correspondence_fragment(source_fragment_range, target_fragment_range)
    else: 
        source_fragment_range = groups[fragment_idx]
        target_fragment_range = groups[fragment_idx]
        T_trans = np.identity(4)
    print("T local:\n",T_trans)
    print("T global:\n",T_global)
    T_global = np.dot(T_global, T_trans)
    for idx in range(*groups[fragment_idx]):
        print("=> INTEGRATE Fragment [%05d-%05d] rgbd frame [%d/%d]" % (*groups[fragment_idx], idx-groups[fragment_idx][0], groups[fragment_idx][1] - groups[fragment_idx][0]))
        rgbd = read_rgbd_image(image_names[idx], depth_names[idx], False)
        pose = graphs[fragment_idx].nodes[idx-groups[fragment_idx][0]].pose
        pose = np.dot(T_global,pose)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

    pcd = volume.extract_point_cloud()
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    
vis.destroy_window()    
# merged_pcd = merge_pcds(pcds)
# o3d.io.write_point_cloud("outputs/fragments/MAIN.ply", merged_pcd, False, True)
# o3d.visualization.draw_geometries([merged_pcd])
o3d.visualization.draw_geometries([pcd], width=640, height=320)


####################################################################
# Merged Fragments Global Registration
####################################################################

merged_pcd = pcds[0]
for pcd_trans in tqdm(pcds):

    merged_pcd_down = copy.deepcopy(merged_pcd)
    pcd_trans_down = copy.deepcopy(pcd_trans)

    merged_pcd_down.voxel_down_sample(0.3)
    pcd_trans_down.voxel_down_sample(0.3)
    merged_pcd_down.estimate_normals()
    pcd_trans_down.estimate_normals()

    T = deep_global_registration(pcd_trans_down, merged_pcd_down, 'kitti')
    pcd_trans.transform(T)
    
    
    merged_pcd = merge_pcds([merged_pcd, pcd_trans])
o3d.visualization.draw_geometries([merged_pcd])
o3d.io.write_point_cloud("outputs/fragments/MAIN.ply", merged_pcd, False, True)