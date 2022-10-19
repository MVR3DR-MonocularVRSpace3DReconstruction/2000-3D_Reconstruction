import os
import open3d as o3d
import glob
from tqdm import tqdm
import argparse

def read_point_clouds(data_dir = "./data/redwood-livingroom/",down_sample=0.1):
    pcds = []
    for pcd in tqdm(sorted(glob.glob(data_dir+'/*.ply'))):
        temp_pcd = o3d.io.read_point_cloud(pcd)
        temp_pcd = temp_pcd.voxel_down_sample(down_sample)
        temp_pcd.estimate_normals()
        # temp_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcds.append(temp_pcd)
    return pcds



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", default="data/redwood-boardroom/scene/")
    parser.add_argument("--down_sample", default=0.01)
    args = parser.parse_args()
    
    print("=> Loading...")
    pcds = read_point_clouds(args.dir, args.down_sample)

    for pcd in pcds:
        o3d.visualization.draw_geometries([pcd])
        
    
