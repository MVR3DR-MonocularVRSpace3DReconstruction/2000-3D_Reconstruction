import open3d as o3d
from utils import *
from rtvec2extrinsic import *

from deep_global_registration.core.deep_global_registration import DeepGlobalRegistration
from deep_global_registration.config import get_config

import copy

def deep_global_registration(source, target):
    # print("=> Apply Deep Global Reg ")
    if 'DGR' not in globals():
        config = get_config()
        config.weights = "deep_global_registration/pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
        global DGR
        DGR = DeepGlobalRegistration(config)
    T, _ = DGR.register(source, target)
    return T
# pcds = read_point_clouds(data_dir="outputs/", down_sample=1)
files = sorted(glob.glob('outputs/fragments/*.ply'))
graphs = []
for graph in tqdm(sorted(glob.glob('outputs/posegraph/fragment_opti_*.json'))):
    temp = o3d.io.read_pose_graph(graph)
    graphs.append(temp)

# pcd = pcds[0]
pcd0 = o3d.io.read_point_cloud(files[0])
pcd1 = o3d.io.read_point_cloud(files[1])

pcd = copy.deepcopy(merge_pcds([pcd0, pcd1]))

pcd0.random_down_sample(0.03)
pcd1.random_down_sample(0.08)
pcd.random_down_sample(0.05)

pcd0.paint_uniform_color([1,0,0]) # red
pcd1.paint_uniform_color([0,1,0]) # green
pcd.paint_uniform_color([0,0,1]) # blue

o3d.visualization.draw_geometries([pcd1, pcd0, pcd])
pcd0.transform(angelPos2Transformation(10,0,90, 0,0,0))
pcd1.transform(angelPos2Transformation(20,0,90, 0,0,0))
pcd.transform(angelPos2Transformation(30,0,90, 0,0,0))

o3d.visualization.draw_geometries([pcd1, pcd0, pcd])

