
from utils import *
from colored_icp import *
from tqdm import tqdm
import open3d as o3d
from overlap import overlap_predator

from deep_global_registration.core.deep_global_registration import DeepGlobalRegistration
from deep_global_registration.config import get_config


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

def colored_icp_registration(source, target, voxel_size):
    # print("# Colored ICP registration")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    voxel_radius = [15*voxel_size, 5*voxel_size, 1.5*voxel_size]
    max_iter = [100, 60, 35]# [60, 35, 20]
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
    source_down.estimate_normals()
    target_down.estimate_normals()

    transformation_dgr, useSafeGuard = DGR.register(source_down, target_down)
    overlap_ratio = compute_overlap_ratio(source_down, target_down, transformation_dgr, down_voxel_size)
    print(overlap_ratio)
    
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

def deep_global_registration(source, target):
    # print("=> Apply Deep Global Reg ")
    if 'DGR' not in globals():
        config = get_config()
		# best_val_checkpoint.pth  ResUNetBN2C-feat32-3dmatch-v0.05.pth   ResUNetBN2C-feat32-kitti-v0.3.pth
        config.weights = "deep_global_registration/pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
        global DGR
        DGR = DeepGlobalRegistration(config)
    _transformation_dgr, _ = DGR.register(source, target)
    return _transformation_dgr

def run(config):
	pcds = load_point_clouds()
	print("* Total "+str(len(pcds))+" point clouds loaded. ")
	# o3d.visualization.draw_geometries(pcds)
	reged_pcds = [pcds[0]]
	vis = o3d.visualization.VisualizerWithEditing()
	vis.add_geometry(reged_pcds[0])
	vis.create_window()
	for pcd in tqdm(pcds[1:]):
		T, _, _ = global_registration(pcd, reged_pcds[-1])
  		
		pcd.transform(T)
		
		# stored pcd
		reged_pcds.append(pcd)
		merged_pcd = merge_pcds(reged_pcds)
		# o3d.visualization.draw_geometries([merged_pcd])
		o3d.io.write_point_cloud(config['outputs_dir']+"Appended_1by1_pair2pair.ply", merged_pcd)
		vis.clear_geometries()
		vis.add_geometry(merged_pcd)
		vis.poll_events()
		vis.update_renderer()
    
	vis.destroy_window()    
	# merged_pcd = merge_pcds(reged_pcds)
	# o3d.io.write_point_cloud(config['outputs_dir']+"Appended_pair2pair.ply", merged_pcd)
	o3d.visualization.draw_geometries(reged_pcds)

if __name__ == '__main__':
	config = {'path_dataset':"outputs/",
				'outputs_dir': "outputs/",
				"down_sample": 0.01}
	run(config)
