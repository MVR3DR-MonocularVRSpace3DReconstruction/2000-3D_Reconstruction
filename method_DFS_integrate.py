import os
import numpy as np

from time import time
import datetime

import math
import open3d as o3d

from deep_global_registration.core.deep_global_registration import DeepGlobalRegistration
from deep_global_registration.config import get_config

from utils import *
from colored_icp import *
from overlap import overlap_predator

from rtvec2extrinsic import *

SKIP_STEP = 5
N_PER_FRAGMENT = 50
EXTEND_RATIO = 5

DOWN_SAMPLE_VOXEL_SIZE = 0.05
BASIC_VOXEL_SIZE = 0.02


data_dir = "outputs/livingroom_S5_N50_E5/"
image_names = sorted(glob.glob('data/redwood-livingroom/image/*.jpg'))
depth_names = sorted(glob.glob('data/redwood-livingroom/depth/*.png'))

image_names = [image_names[idx] for idx in range(0,len(image_names), SKIP_STEP)]
depth_names = [depth_names[idx] for idx in range(0,len(depth_names), SKIP_STEP)]

groups = np.loadtxt(data_dir+"posegraph/groups.txt")
groups = [[int(group[0]), int(group[1])] for group in groups]

pose_graph_files = sorted(glob.glob(data_dir+'posegraph/fragment_*.json'))
graphs = [o3d.io.read_pose_graph(graph) for graph in pose_graph_files]

assert len(graphs) == len(groups)

def read_point_clouds(data_dir = "./data/redwood-livingroom/",down_sample=0.1):
    pcds = []
    for pcd in tqdm(sorted(glob.glob(data_dir+'fragments/*.ply'))):
        temp_pcd = o3d.io.read_point_cloud(pcd)
        temp_pcd = temp_pcd.voxel_down_sample(down_sample)
        temp_pcd.estimate_normals()
        # temp_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcds.append(temp_pcd)
    return pcds

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
  matching10 = get_matching_indices(pcd1_down, pcd0_down, np.linalg.inv(trans), voxel_size, 1)
  overlap0 = len(matching01) / len(pcd0_down.points)
  overlap1 = len(matching10) / len(pcd1_down.points)
  return max(overlap0, overlap1)

def colored_icp_registration(source, target, voxel_size):
    # print("# Colored ICP registration")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    voxel_radius = [5*voxel_size, 3*voxel_size, 1.5*voxel_size]
    max_iter = [60, 35, 20]
    _transformation_cicp = np.identity(4)
    for scale in range(3):
        max_it = max_iter[scale]
        radius = voxel_radius[scale]
        # print("=> scale_level = {0}, voxel_size = {1}, max_iter = {2}".format(scale, radius, max_it))
        try:
            result = o3d.pipelines.registration.registration_colored_icp(
                source, target, radius, _transformation_cicp,
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
    source_down = copy.deepcopy(source)
    target_down = copy.deepcopy(target)
    source_down.voxel_down_sample(DOWN_SAMPLE_VOXEL_SIZE)
    target_down.voxel_down_sample(DOWN_SAMPLE_VOXEL_SIZE)
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=DOWN_SAMPLE_VOXEL_SIZE*2, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=DOWN_SAMPLE_VOXEL_SIZE*2, max_nn=30))

    uncertain = False
    transformation_dgr, useSafeGuard = DGR.register(source_down, target_down)
    dgr_overlap_ratio = compute_overlap_ratio(source_down, target_down, transformation_dgr, BASIC_VOXEL_SIZE)
    # print(overlap_ratio)
    # if False:
    if dgr_overlap_ratio < 0.3:
        transformation_overlap = overlap_predator(source_down, target_down)
        op_overlap_ratio = compute_overlap_ratio(source_down, target_down, transformation_overlap, BASIC_VOXEL_SIZE)
        # print("fix to:",overlap_ratio)

        transformation_ransac = ransac_global_registration(source_down, target_down, DOWN_SAMPLE_VOXEL_SIZE)
        ransac_overlap_ratio = compute_overlap_ratio(source_down, target_down, transformation_ransac, BASIC_VOXEL_SIZE)
        # print("fix to:",overlap_ratio)
        # input()
        chooseT = 0
        if dgr_overlap_ratio >= op_overlap_ratio and dgr_overlap_ratio >= ransac_overlap_ratio:
            transformation = transformation_dgr
        elif op_overlap_ratio >= dgr_overlap_ratio and op_overlap_ratio >= ransac_overlap_ratio:
            transformation = transformation_overlap
            chooseT = 1
        else:
            transformation = transformation_ransac
            chooseT = 2
        print("=> DGR Overlap ratio:{} too small, fix to {}\n//overlap predator:{}\n//ransac:{}".format(
            dgr_overlap_ratio, ["DGR","PREDATOR","RANSAC"][chooseT], op_overlap_ratio, ransac_overlap_ratio))
    else:
        transformation = transformation_dgr
        
    if dgr_overlap_ratio < 0.3:
        uncertain = True
    source_down.transform(transformation)
    
    transformation_icp = colored_icp_registration(source_down, target_down, DOWN_SAMPLE_VOXEL_SIZE)
    
    transformation = transformation @ transformation_icp
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source_down, target_down, max_correspondence_distance_fine,
        transformation)

    return transformation, information, uncertain
    
def integrate_pcd(left_range, right_range, left_trans, right_trans):
    # print(left_range, right_range, left_trans, right_trans)
    assert left_range[1]-left_range[0] == len(left_trans)
    assert right_range[1]-right_range[0] == len(right_trans)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=3.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # enumerate left fragments
    for idx in range(*left_range):
        # enumerate frames in fragments
        idx_abs = idx-left_range[0]
        poses = graphs[idx]
        for iidx in range(*groups[idx]):
            # print("poses",len(poses.nodes),"idx",idx,"left range",left_range,"groups",groups[idx],"iidx", iidx)
            pose = left_trans[idx_abs] @ np.linalg.inv(poses.nodes[iidx-groups[idx][0]].pose)
            color = o3d.io.read_image(image_names[iidx])
            depth = o3d.io.read_image(depth_names[iidx])
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_trunc=1000, convert_rgb_to_intensity=False)
            volume.integrate(rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
                np.linalg.inv(pose))
    
    # enumerate right fragments
    for idx in range(*right_range):
        # enumerate frames in fragments
        idx_abs = idx-right_range[0]
        poses = graphs[idx]
        for iidx in range(*groups[idx]):
            pose = right_trans[idx_abs] @ np.linalg.inv(poses.nodes[iidx-groups[idx][0]].pose)
            color = o3d.io.read_image(image_names[iidx])
            depth = o3d.io.read_image(depth_names[iidx])
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_trunc=1000, convert_rgb_to_intensity=False)
            volume.integrate(rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
                np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    # o3d.visualization.draw_geometries([pcd])
    return pcd
        
def pcd_integrate_dfs(pcd_range, pcd_list, depth):
    n_pcds = pcd_range[1] - pcd_range[0]
    # return single pcd
    if n_pcds < 2:
        print("  |"*(depth-1)+"---> Single Point Cloud [Returned]")
        return pcd_list[0], [np.identity(4)]
    
    left_range = [pcd_range[0], pcd_range[0]+n_pcds//2]
    right_range = [pcd_range[0]+n_pcds//2, pcd_range[1]]
    print("  |"*(depth-1)+"---> left range:{}  right range:{}".format(left_range, right_range))

    # get half of merged pcds
    left_pcd, left_trans = pcd_integrate_dfs(left_range, pcd_list[:n_pcds//2], depth+1)
    right_pcd, right_trans = pcd_integrate_dfs(right_range, pcd_list[n_pcds//2:], depth+1)
    print("  |"*(depth-1)+"---> Registration..")
    # o3d.visualization.draw_geometries([left_pcd, right_pcd])
    T, _, _ = global_registration(left_pcd, right_pcd)
    left_trans = [trans @ T for trans in left_trans]
    print("  |"*(depth-1)+"---> Merge pcds")
    merged_pcd = integrate_pcd(left_range, right_range, left_trans, right_trans)
    merged_pcd.voxel_down_sample(BASIC_VOXEL_SIZE)
    merged_pcd.estimate_normals()
    
    # storage temp
    time_cost = time()-start_time
    timestamp = "{}m{}s".format(int((time_cost)//60), int(time_cost - (time_cost)//60*60))
    o3d.io.write_point_cloud("./outputs/dfs_integrate/D{:0>3}_L{:0>3}_{}.ply".format(depth, n_pcds, timestamp), merged_pcd)
    # o3d.visualization.draw_geometries([merged_pcd])

    print("  |"*(depth-1)+"---> List length: {} Stack Depth: {} [Merged Complete]".format(n_pcds, depth))
    return merged_pcd, left_trans+right_trans

if __name__ == '__main__':

	start_time = time()
	config = get_config()
	config.weights = "deep_global_registration/pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
	DGR = DeepGlobalRegistration(config)

	pcds = read_point_clouds(data_dir = data_dir,down_sample=BASIC_VOXEL_SIZE)
	
	pause_time = time()
	# o3d.visualization.draw_geometries(pcds)
	pause_time = time() - pause_time
	merged_pcd = o3d.geometry.PointCloud()
	main_pcd =  pcd_integrate_dfs([0, len(pcds)], pcds, 0)


	end_time = time()
	time_cost = end_time-start_time-pause_time
	print("\n## Total cost {}s = {}m{}s.".format(
		time_cost, int((time_cost)//60), int(time_cost - (time_cost)//60*60)))
	o3d.io.write_point_cloud("./outputs/dfs_integrate/DFS-outputs.ply", main_pcd)
	o3d.visualization.draw_geometries([main_pcd])