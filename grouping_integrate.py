# Base package
import os
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
data_dir = "data/shrink-test/"

image_names = sorted(glob.glob(data_dir+'image/*.jpg'))
print("=> Load Images.. ")
images=[]
for filepath in tqdm(image_names):
    with Image.open(filepath) as img: 
        img = np.array(img)
        images.append(img)

groups=[]
sid = 0
_skip_steps = 10
for tid in tqdm(range(0, len(images), _skip_steps)):
    score = uqi(images[sid], images[tid])
    if score < 0.75:
        groups.append([sid, tid])
        sid = tid
    # input()
print(groups)

print("=> Align groups.. ")
groups_align=[]
_extend_img_ratio = 5
for [sid, tid] in groups:
    length = tid - sid
    sid = max(0, sid-length//_extend_img_ratio)
    tid = min(len(images), tid+length//_extend_img_ratio)
    groups_align.append([sid, tid])
print(groups_align)

####################################################################
# Point Clouds Fragments Process
####################################################################

import open3d as o3d
from utils import *
from deep_global_registration.core.deep_global_registration import DeepGlobalRegistration
from deep_global_registration.config import get_config
from overlap import overlap_predator

def colored_icp_registration(source, target, voxel_size):
    # print("=> Colored ICP registration")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    voxel_radius = [5*voxel_size, 3*voxel_size, voxel_size]
    max_iter = [60, 35, 20]# [60, 35, 20]
    T = np.identity(4)
    for scale in range(3):
        max_it = max_iter[scale]
        radius = voxel_radius[scale]
        # print("=> scale_level = {0}, voxel_size = {1}, max_iter = {2}".format(scale, radius, max_it))
        try:
            result = o3d.pipelines.registration.registration_colored_icp(
                source, 
                target, 
                radius, 
                T,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                    relative_rmse=1e-6,
                                                                    max_iteration=max_it))
            T = result.transformation                                                    
        except Exception as e:
            # print("=> No correspondence found. ")
            continue
    return T

def deep_global_registration(source, target):
    # print("=> Apply Deep Global Reg ")
    if 'DGR' not in globals():
        config = get_config()
        config.weights = "deep_global_registration/pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
        global DGR
        DGR = DeepGlobalRegistration(config)
    T, _ = DGR.register(source, target)
    return T

def run_once_registration(pcds,sid,eid, model="dgr", shrink_step=1):
    pcds = pcds[sid:eid]
    transed_pcds = [pcds[0]]
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    for idx in tqdm(range(0,len(pcds)//shrink_step,shrink_step)):
        pcd_trans = pcds[idx]
        pcd_base = transed_pcds[-1]
        if model=="dgr":
            T1 = deep_global_registration(pcd_trans, pcd_base)
        elif model=="overlap":
            T1 = overlap_predator(pcd_trans, pcd_base)
        pcd_trans.transform(T1)
        T2 = color_icp_cpp(pcd_trans, pcd_base)
        pcd_trans.transform(T2)
        T = np.dot(T1, T2)
        odometry = np.dot(T, odometry)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(odometry)
            )
        )
        transed_pcds.append(pcd_trans)
    merged_pcd = merge_pcds(transed_pcds)
    return merged_pcd, pose_graph



pcds = load_point_clouds(data_dir=data_dir, step=1)
# o3d.visualization.draw_geometries(pcds)
fragments=[]
for [fragments_sid, fragments_tid] in groups_align:
    pcd_fragment = run_once_registration(pcd_fragment,fragments_sid,fragments_tid, "dgr", 5)
    o3d.io.write_point_cloud("outputs/Appended_simple_seperate_{}-{}.ply".format( fragments_sid, fragments_tid))
    # o3d.visualization.draw_geometries([pcd_fragment])
    fragments.append(pcd_fragment)


main_pcd = run_once_registration(fragments, "overlap", 1)
o3d.visualization.draw_geometries([main_pcd])