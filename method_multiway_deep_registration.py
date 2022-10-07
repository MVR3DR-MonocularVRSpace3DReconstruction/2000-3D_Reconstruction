import os
from time import time
import numpy as np
import open3d as o3d
import numpy as np

from tqdm import tqdm
from deep_global_registration.core.deep_global_registration import DeepGlobalRegistration
from deep_global_registration.config import get_config

from utils import *
from colored_icp import colored_icp
from overlap import overlap_predator

config = get_config()
config.weights = "deep_global_registration/pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
DGR = DeepGlobalRegistration(config)
        
def load_point_clouds(data_dir = "./data/redwood-livingroom/",
                      camera_pose_file = "livingroom.log",
                      step = 1,
                      voxel_size=0.02):
    image_dir = sorted(glob.glob(data_dir+'image/*.jpg'))
    depth_dir = sorted(glob.glob(data_dir+'depth/*.png'))
    camera_poses = read_trajectory("{}{}".format(data_dir, camera_pose_file))

    pcds = []
    # print("=> Start loading point clouds...")
    for idx in tqdm(range(0, len(image_dir), step)):
        # print("=> Load [{}/{}] point cloud".format(i//step,len(image_dir)//step))
        pcd = generate_point_cloud(
                image_dir[idx],
                depth_dir[idx],
                # camera_poses[i].pose
                )
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd.estimate_normals()
        pcds.append(pcd)
    # print("# Load {} point clouds from {}".format(len(pcds),data_dir))
    return pcds


def pairwise_registration(source, target, max_correspondence_distance_coarse,
                          max_correspondence_distance_fine):
    source_down = copy.deepcopy(source)
    target_down = copy.deepcopy(target)
    source_down.voxel_down_sample(0.05)
    target_down.voxel_down_sample(0.05)
    source_down.estimate_normals()
    target_down.estimate_normals()
    transformation, _ = DGR.register(source_down, target_down)
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source_down, target_down, max_correspondence_distance_fine,
        transformation)
    # print("Apply point-to-plane ICP")
    # icp_coarse = o3d.pipelines.registration.registration_icp(
    #     source, target, max_correspondence_distance_coarse, np.identity(4),
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # icp_fine = o3d.pipelines.registration.registration_icp(
    #     source, target, max_correspondence_distance_fine,
    #     icp_coarse.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # transformation_icp = icp_fine.transformation
    # information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
    #     source, target, max_correspondence_distance_fine,
    #     icp_fine.transformation)
    return transformation, information


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)

    for source_id in tqdm(range(n_pcds), desc="FULL REG"):
        for target_id in tqdm(range(source_id + 1, min(source_id+15, n_pcds)), desc="FRAME"):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id],
                max_correspondence_distance_coarse,
                max_correspondence_distance_fine)
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
    return pose_graph


if __name__ == "__main__":
    voxel_size = 0.02
    pcds_down = load_point_clouds(step=3)
    o3d.visualization.draw(pcds_down)

    print("=> Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pcds_down,
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)

    print("=> Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    print("=> Transform points and display")
    for point_id in tqdm(range(len(pcds_down)), desc="MERGE"):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
        
    pcd = merge_pcds(pcds_down)
    pcd_name = "outputs/multi_dgr.ply"
    o3d.io.write_point_cloud(pcd_name, pcd, False, True)
    o3d.visualization.draw(pcds_down)
