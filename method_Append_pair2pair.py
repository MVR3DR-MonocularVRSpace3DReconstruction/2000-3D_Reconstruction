
from utils import *
from colored_icp import *
from tqdm import tqdm
import open3d as o3d
from overlap import overlap_predator

from deep_global_registration.core.deep_global_registration import DeepGlobalRegistration
from deep_global_registration.config import get_config

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
	pcds = read_point_clouds(data_dir=config['path_dataset'], down_sample=config['down_sample'])
	print("* Total "+str(len(pcds))+" point clouds loaded. ")
	o3d.visualization.draw_geometries(pcds)	
	reged_pcds = [pcds[0]]
	for pcd in tqdm(pcds[1:]):
		# get base from registrated pcd list
		pcd0 = copy.deepcopy(pcd)
		pcd1 = copy.deepcopy(reged_pcds[-1])
		
		pcd0.voxel_down_sample(0.05)
		pcd0.estimate_normals()

		pcd1.voxel_down_sample(0.05)
		pcd1.estimate_normals()
		o3d.visualization.draw_geometries([pcd0, pcd1])
		# registration
		T = deep_global_registration(pcd0, pcd1) # pcd_base
		pcd0.transform(T)
		pcd.transform(T)
		o3d.visualization.draw_geometries([pcd0, pcd1])
		# T = overlap_predator(pcd_trans, pcd_base) # pcd_base
		# pcd_trans.transform(T)

		# color registration
		T = colored_icp_registration(pcd0, pcd1, 0.05) # pcd_base
		pcd.transform(T)
		
		# stored pcd
		reged_pcds.append(pcd)
		merged_pcd = merge_pcds(reged_pcds)
		o3d.visualization.draw_geometries([pcd0, pcd1])
		o3d.io.write_point_cloud(config['outputs_dir']+"Appended_pair2pair.ply", merged_pcd)
	# merged_pcd = merge_pcds(reged_pcds)
	# o3d.io.write_point_cloud(config['outputs_dir']+"Appended_pair2pair.ply", merged_pcd)
	o3d.visualization.draw_geometries(reged_pcds)

if __name__ == '__main__':
	config = {'path_dataset':"outputs/",
				'outputs_dir': "outputs/",
				"down_sample": 0.1}
	run(config)
