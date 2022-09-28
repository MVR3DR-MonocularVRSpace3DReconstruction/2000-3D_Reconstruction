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
data_dir = "data/redwood-livingroom/"
_steps = 10

image_names = sorted(glob.glob(data_dir+'image/*.jpg'))
print("=> Load Images.. ")
images=[]
for idx in tqdm(range(0, len(image_names), _steps)):
    with Image.open(image_names[idx]) as img: 
        img = np.array(img)
        images.append(img)
# true idx = idx * _steps

groups=[]
sid = 0
_skip_steps = 10
for tid in tqdm(range(0, len(images), _skip_steps)):
    score = uqi(images[sid], images[tid])
    if score < 0.85:
        groups.append([sid, tid])
        sid = tid
groups.append([tid,len(images)])
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
groups_align = sorted(groups_align,key= lambda x:(x[0], x[1]))    
print(groups_align)

####################################################################
# Point Clouds Fragments Process
####################################################################

import open3d as o3d
from utils import *
from deep_global_registration.core.deep_global_registration import DeepGlobalRegistration
from deep_global_registration.config import get_config
from overlap import overlap_predator
from colored_icp import colored_icp_registration


def deep_global_registration(source, target):
    # print("=> Apply Deep Global Reg ")
    if 'DGR' not in globals():
        config = get_config()
        config.weights = "deep_global_registration/pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
        global DGR
        DGR = DeepGlobalRegistration(config)
    T, _ = DGR.register(source, target)
    return T

def run_once_registration(pcds, model="dgr", shrink_step=1):
    transed_pcds = [pcds[0]]
    for idx in tqdm(range(0,len(pcds)//shrink_step,shrink_step), desc="Append Fragment"):
        pcd_trans = pcds[idx]
        pcd_base = transed_pcds[-1]
        if model=="dgr":
            T = deep_global_registration(pcd_trans, pcd_base)
        elif model=="overlap":
            T = overlap_predator(pcd_trans, pcd_base)
        pcd_trans.transform(T)
        T = colored_icp_registration(pcd_trans, pcd_base, 0.05)
        pcd_trans.transform(T)
        transed_pcds.append(pcd_trans)
    merged_pcd = merge_pcds(transed_pcds)
    o3d.visualization.draw_geometries([merged_pcd])
    return merged_pcd

pcds = load_point_clouds(data_dir=data_dir, step=_steps, down_sample=1)

fragments=[]

for [fragments_sid, fragments_tid] in tqdm(groups_align, desc="Make Fragments"):
    pcd_fragment = pcds[fragments_sid:fragments_tid]
    pcd_fragment = run_once_registration(pcd_fragment, "dgr", 1)
    o3d.io.write_point_cloud("outputs/simple/Appended_simple_seperate_{}-{}.ply".format(fragments_sid, fragments_tid), pcd_fragment)
    # o3d.visualization.draw_geometries([pcd_fragment])
    fragments.append(pcd_fragment)

main_pcd = run_once_registration(fragments, "overlap", 1)
o3d.io.write_point_cloud("outputs/simple/Appended_simple_seperate_Main.ply", main_pcd)
o3d.visualization.draw_geometries(fragments)