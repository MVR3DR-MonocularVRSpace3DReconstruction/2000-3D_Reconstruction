# Base package
import os
import numpy as np
import glob
from tqdm import tqdm

# Image package
from PIL import Image
from sewar.full_ref import uqi, ssim, msssim
import matplotlib.pyplot as plt

import cv2

####################################################################
# Config 
####################################################################

SKIP_STEP = 3
N_PER_FRAGMENT = 50
EXTEND_RATIO = 5

####################################################################
# Grouping
####################################################################
# data_dir = "data/redwood-livingroom/"
data_dir = "data/redwood-livingroom/"
image_names = sorted(glob.glob(data_dir+'image/*.jpg'))
depth_names = sorted(glob.glob(data_dir+'depth/*.png'))

image_names = [image_names[idx] for idx in range(0,len(image_names), SKIP_STEP)]
depth_names = [depth_names[idx] for idx in range(0,len(depth_names), SKIP_STEP)]
print("==> Evaluate Images.. ")

def make_fragments_groups(image_names, mode = "value", steps=50, extend_img_ratio = 5, theroshold = 0.85):
    # fresh documents
    os.system("rm -rf outputs/fragments/")
    os.system("rm -rf outputs/posegraph/")
    os.system("mkdir outputs/fragments/")
    os.system("mkdir outputs/posegraph/")
    n_images = len(image_names)
    if mode == "value":
        # evaluate images & grouping
        print("=> Cluster images")
        groups=[]
        sid = 0
        
        source = np.array(Image.open(image_names[sid]).convert('L'))
        for tid in tqdm(range(n_images)):
            target = np.array(Image.open(image_names[tid]).convert('L'))
            score = uqi(source, target)
            # print(score, sid, tid)
            if score < theroshold:
                groups.append([sid, tid]) # need fix range in [sid, tid)
                sid = tid
                source = np.array(Image.open(image_names[sid]).convert('L'))
        if groups[-1][1] != n_images-1:
            groups.append([sid, n_images-1])
    
    if mode == "static":
        # print(n_images)
        groups=[[idx*steps, (idx+1)*steps] for idx in range(0, n_images//steps)]
        
    # extend groups with overlap frames
    print("=> Align groups.. ")
    temp_groups=[]
    for [sid, tid] in tqdm(groups):
        length = tid - sid + 1
        sid = max(0, sid-length//extend_img_ratio)
        tid = min(n_images-1, tid+length//extend_img_ratio)
        temp_groups.append([sid, tid+1])  # fix range in [sid, tid)
    groups = np.array(temp_groups)
    
    if mode == "value":
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
        print("Origin: {}, shrink to:{}".format(len(temp_groups), len(groups)))
    
    print(groups)
    np.savetxt("outputs/posegraph/groups.txt",groups)
    return groups

#===================================================================
# Main proc
#===================================================================

read_file = False
if read_file:
    groups = np.loadtxt("outputs/posegraph/groups.txt")
    groups = [[int(group[0]), int(group[1])] for group in groups] 
else:
    groups = make_fragments_groups(image_names, mode="static", 
        steps=N_PER_FRAGMENT//SKIP_STEP, extend_img_ratio=EXTEND_RATIO, theroshold=0.9)

####################################################################
# Get basic data
####################################################################

import open3d as o3d
from utils import *

# check opencv python package
def generate_point_cloud(image_dir:str, depth_dir:str):
    color_raw = o3d.io.read_image(image_dir)
    depth_raw = o3d.io.read_image(depth_dir)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
    )
    # Flip it, otherwise the pointcloud will be upside down
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd

def load_point_clouds(sid, eid, 
                      image_file, depth_file,
                      step = 1,
                      data_dir = "./data/redwood-livingroom/",
                      voxel_size=0):
    # image_dir = sorted(glob.glob(data_dir+'image/*.jpg'))
    # depth_dir = sorted(glob.glob(data_dir+'depth/*.png'))
    # camera_poses = read_trajectory("data/redwood-livingroom/livingroom.log")

    pcds = []
    # print("=> Start loading point clouds...")
    for idx in range(sid, eid, step):  #, desc="load pcd"):
        # print("=> Load {}/[{}-{}] point cloud".format(idx//step, sid, eid))
        pcd = generate_point_cloud(
                image_file[idx],
                depth_file[idx],
                # camera_poses[i].pose
                )
        if voxel_size != 0:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd.estimate_normals()
        pcds.append(pcd)
    return pcds
# PCDS = load_point_clouds()

def read_point_clouds(data_dir = "outputs/",down_sample=0):
    pcds = []
    for pcd in tqdm(sorted(glob.glob(data_dir+'fragments/*.ply'))):
        temp_pcd = o3d.io.read_point_cloud(pcd)
        if down_sample != 0:
            temp_pcd = temp_pcd.voxel_down_sample(down_sample)
        temp_pcd.estimate_normals()
        # temp_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcds.append(temp_pcd)
    return pcds

####################################################################
# Evaluate registration
####################################################################

def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds

def compute_overlap_ratio(pcd0, pcd1, trans, voxel_size):
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    matching01 = get_matching_indices(pcd0_down, pcd1_down, trans, voxel_size, 1)
    matching10 = get_matching_indices(pcd1_down, pcd0_down, np.linalg.inv(trans),
                                    voxel_size, 1)
    overlap0 = len(matching01) / len(pcd0_down.points)
    overlap1 = len(matching10) / len(pcd1_down.points)
    return max(overlap0, overlap1)

####################################################################
# Registration
####################################################################

from deep_global_registration.core.deep_global_registration import DeepGlobalRegistration
from deep_global_registration.config import get_config
from overlap import overlap_predator

def colored_icp_registration(source, target, voxel_size):
    # print("# Colored ICP registration")
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

def ransac_global_registration(source_down, target_down, voxel_size):
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

def global_registration(source, target, max_correspondence_distance_fine=0.03):
    # print("# Apply Deep Global Reg ")
    if 'DGR' not in globals():
        config = get_config()
		# best_val_checkpoint.pth  ResUNetBN2C-feat32-3dmatch-v0.05.pth   ResUNetBN2C-feat32-kitti-v0.3.pth
        config.weights = "deep_global_registration/pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
        global DGR
        DGR = DeepGlobalRegistration(config)
    down_voxel_size = 0.05
    source_down = copy.deepcopy(source)
    target_down = copy.deepcopy(target)
    source_down.voxel_down_sample(down_voxel_size)
    target_down.voxel_down_sample(down_voxel_size)
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=down_voxel_size*2, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=down_voxel_size*2, max_nn=30))

    transformation_dgr, useSafeGuard = DGR.register(source_down, target_down)
    overlap_ratio = compute_overlap_ratio(source_down, target_down, transformation_dgr, down_voxel_size)
    # print(overlap_ratio)
    
    if overlap_ratio < 0.3:
        transformation_dgr = overlap_predator(source_down, target_down)
        overlap_ratio = compute_overlap_ratio(source_down, target_down, transformation_dgr, down_voxel_size)
        print("fix to:",overlap_ratio)
    if overlap_ratio < 0.3:
        transformation_dgr = ransac_global_registration(source_down, target_down, down_voxel_size)
        overlap_ratio = compute_overlap_ratio(source_down, target_down, transformation_dgr, down_voxel_size)
        print("fix to:",overlap_ratio)
        # input()

    source_down.transform(transformation_dgr)
    
    transformation_icp = colored_icp_registration(source_down, target_down, down_voxel_size)
    
    transformation = transformation_dgr @ transformation_icp
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source_down, target_down, max_correspondence_distance_fine,
        transformation)
    # source_down.transform(transformation)
    # o3d.visualization.draw_geometries([source_down, target_down])
    return transformation, information, useSafeGuard

####################################################################
# Integrate Fragments
####################################################################

def make_fragments(sid, eid, color_files, depth_files, intrinsic="", tsdf_cubic_size=3.0):
    assert eid > sid
    if intrinsic != "":
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsic)
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    
    pcds = load_point_clouds(sid, eid, color_files, depth_files)
    
    pose_graph = o3d.pipelines.registration.PoseGraph()
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length = tsdf_cubic_size / 512.0,
        sdf_trunc = 0.04,
        color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    T_global = np.identity(4)
    for idx in tqdm(range(sid, eid), desc="load frame"):
        # print("=> INTEGRATE {} [{}-{}]".format(idx-sid, sid, eid))
        if idx == sid:
            T_local = np.identity(4)
        else:
            T_local, info, _ = global_registration(pcds[idx-sid], pcds[idx-sid-1])
        T_global = T_global @ T_local
        rgbd = read_rgbd_image(color_files[idx], depth_files[idx])
        volume.integrate(rgbd, intrinsic, np.linalg.inv(T_global))
        
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(T_global)))
        if idx != sid:
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    idx-1, idx, np.linalg.inv(T_local), info, uncertain=False))

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd_name = "outputs/fragments/fragment_{:0>5}-{:0>5}.ply".format(sid,eid)
    o3d.io.write_pose_graph("outputs/posegraph/fragment_{:0>5}-{:0>5}.json".format(sid,eid), pose_graph)
    o3d.io.write_point_cloud(pcd_name, pcd, False, True)

#===================================================================
# Main proc
#===================================================================

n_fragments = len(groups)
print("==> Make Fragments to point cloud")
if False:
    from joblib import Parallel, delayed
    import multiprocessing
    import subprocess
    MAX_THREAD = min(multiprocessing.cpu_count()-1, n_fragments)
    Parallel(n_jobs=MAX_THREAD)(delayed(make_fragments)(sid, eid, image_names, depth_names) for [sid, eid] in groups)
else:
    for [sid, eid] in tqdm(groups, desc="make fragments"):
        make_fragments(sid, eid, image_names, depth_names)
        # step value according to extend ratio, e.g. length 50, extend 5 => 10 frames overlap => step < 10 is prefered

####################################################################
# Pose Alignment by overlap frames
####################################################################

from rtvec2extrinsic import *

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

# 0:7     4:10
# 7       6
#        [4]
# 0 1 2 3 4 5 6
#         4 5 6 7 8 9
#        [0]

def get_transformation_from_correspondence_fragment(
    base_graph, trans_graph, n_per_fragment, extend_ratio):
    print("=> get transformation from correspondence fragment")
    print(len(base_graph),len(trans_graph))
    base_pos = []; trans_pos = []
    loop_range = n_per_fragment//extend_ratio
    base_len = len(base_graph.nodes)
    for idx in range(loop_range):
        base_trans = base_graph.nodes[base_len-loop_range+idx].pose
        trans_trans = trans_graph.nodes[idx].pose
        _, _, _, rx, ry, rz = transformation2AnglePos(base_trans)
        base_pos.append([rx, ry, rz ])
        _, _, _, rx, ry, rz = transformation2AnglePos(trans_trans)
        trans_pos.append([rx, ry, rz ])
    print(base_pos, trans_pos)
    T_rigid = rigid_transform_3D(np.array(trans_pos), np.array(base_pos))
    print("%.5f %.5f %.5f %.5f %.5f %.5f " % transformation2AnglePos(T_rigid))
    T_ori = base_graph.nodes[base_len-n_per_fragment//extend_ratio].pose
    return T_ori

#===================================================================
# Main proc
#===================================================================
print("=> Loading pose graphs...")
pose_graph_files = sorted(glob.glob('outputs/posegraph/fragment_*.json'))
graphs = [o3d.io.read_pose_graph(graph) for graph in pose_graph_files]
print("=> Loading fragments...")
pcds = read_point_clouds("outputs/",0.01)
assert len(graphs) == len(pcds)

pcds_align = []
T_global = np.identity(4)
for fragment_idx in tqdm(range(0, n_fragments), desc="Load fragment"):
    # get transformation with next fragment
    if fragment_idx != 0:
        T_trans = get_transformation_from_correspondence_fragment(
            graphs[fragment_idx-1], graphs[fragment_idx],
            N_PER_FRAGMENT, EXTEND_RATIO, SKIP_STEP)
        pcd_base = copy.deepcopy(pcds[fragment_idx-1])
        pcd_trans = copy.deepcopy(pcds[fragment_idx])
    else:
        T_trans = np.identity(4)
        pcd_base = copy.deepcopy(pcds[fragment_idx])
        pcd_trans = copy.deepcopy(pcds[fragment_idx])
    print("T local:\n",T_trans)
    print("T global:\n",T_global)
    T_global = T_global @ np.linalg.inv(T_trans)
    
    # tf_pcd.transform(T_global)
    pcd_trans.transform(T_global)
    pcds_align.append(pcd_trans)
    o3d.visualization.draw_geometries(pcds_align)
merged_pcd = merge_pcds(pcds_align)
# o3d.io.write_point_cloud("outputs/fragments/MAIN.ply", merged_pcd, False, True)
o3d.visualization.draw_geometries([merged_pcd])


input()

####################################################################
# DFS registration 
####################################################################


def pcd_fusion_dfs(_pcd_list, depth):
	n_pcds = len(_pcd_list)
	# return single pcd
	if n_pcds < 2:
		# print("="*50)
		print("  |"*(depth-1)+"---> Single Point Cloud [Returned]")
		# print("="*50)
		return _pcd_list[0]
	if n_pcds > 4:
		# get half of merged pcds
		left_pcd = pcd_fusion_dfs(_pcd_list[:n_pcds//2+1], depth+1)
		right_pcd = pcd_fusion_dfs(_pcd_list[n_pcds//2+1:], depth+1)
	else:
		# get half of merged pcds
		left_pcd = pcd_fusion_dfs(_pcd_list[:n_pcds//2], depth+1)
		right_pcd = pcd_fusion_dfs(_pcd_list[n_pcds//2:], depth+1)
	print("  |"*(depth-1)+"---> Registration..")

	T, _, _ = global_registration(left_pcd, right_pcd)
	left_pcd.transform(T)
	print("  |"*(depth-1)+"---> Merge pcds")
	merged_pcd = merge_pcds([left_pcd,right_pcd])
	# storage temp
	timestamp = int(round(time() * 1000))
	o3d.io.write_point_cloud("./outputs/dfs/D{:0>3}_L{:0>3}_{}.ply".format(depth, n_pcds, timestamp), merged_pcd)
	# o3d.visualization.draw_geometries([merged_pcd])

	# print("="*50)
	print("  |"*(depth-1)+"---> List length: {} Stack Depth: {} [Merged Complete]".format(n_pcds, depth))
	# print("="*50)
	return merged_pcd

#===================================================================
# Main proc
#===================================================================
os.system("rm -rf outputs/dfs/")
os.system("mkdir outputs/dfs/")
start_time = time()
pcd_dfs =  pcd_fusion_dfs(pcds, 1)
end_time = time()
time_cost = end_time-start_time
print("\n## Total cost {}s = {}m{}s.".format(
    time_cost, int((time_cost)//60), int(time_cost - (time_cost)//60*60)))

o3d.io.write_point_cloud("./outputs/dfs/DFS-outputs.ply", pcd_dfs)
o3d.visualization.draw_geometries([pcd_dfs])




















####################################################################
# Full registration 
####################################################################


def full_registration(pcds,max_correspondence_distance):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)

    for source_id in tqdm(range(n_pcds), desc="FULL REG"):
        for target_id in tqdm(range(source_id + 1,min(source_id + 2, n_pcds)), desc="FRAME"):
            transformation_icp, information_icp, _ = global_registration(
                pcds[source_id], pcds[target_id],
                max_correspondence_distance)
            # print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                            target_id,
                                                            transformation_icp,
                                                            information_icp,
                                                            uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                            target_id,
                                                            transformation_icp,
                                                            information_icp,
                                                            uncertain=True))
            for target_id in tqdm(range(source_id+3,n_pcds,7), desc="FRAME"):
                transformation_icp, information_icp, _ = global_registration(
                    pcds[source_id], pcds[target_id],max_correspondence_distance)
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                            target_id,
                                                            transformation_icp,
                                                            information_icp,
                                                            uncertain=True))

    return pose_graph

#===================================================================
# Main proc
#===================================================================

voxel_size = 0.02

pcds = read_point_clouds()
o3d.visualization.draw(pcds)

print("=> Full registration ...")
max_correspondence_distance = voxel_size * 1.5
pose_graph = full_registration(pcds,max_correspondence_distance)

print("=> Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance,
    edge_prune_threshold=0.25,
    reference_node=0)
o3d.pipelines.registration.global_optimization(
    pose_graph,
    o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
    option)
print("=> Transform points and display")
for point_id in tqdm(range(len(pcds)), desc="MERGE"):
    print(pose_graph.nodes[point_id].pose)
    pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    
pcd = merge_pcds(pcds)
pcd_name = "outputs/multi_dgr.ply"
o3d.io.write_point_cloud(pcd_name, pcd, False, True)
o3d.visualization.draw(pcds)




