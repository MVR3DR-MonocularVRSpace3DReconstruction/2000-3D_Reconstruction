
from utils import *
from colored_icp import *
from tqdm import tqdm
import open3d as o3d
from overlap import overlap_predator

from deep_global_registration.core.deep_global_registration import DeepGlobalRegistration
from deep_global_registration.config import get_config

def deep_global_registration(source, target):
    print("=> Apply Deep Global Reg ")
    if 'DGR' not in globals():
        config = get_config()
        config.weights = "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
        global DGR
        DGR = DeepGlobalRegistration(config)
    _transformation_dgr, _ = DGR.register(source, target)
    return _transformation_dgr

def run(config):
	pcds = read_point_clouds(data_dir=config['path_dataset'], down_sample=config['down_sample'])
	print("* Total "+str(len(pcds))+" point clouds loaded. ")
	# o3d.visualization.draw_geometries(pcds)	
	reged_pcds = [pcds[0]]
	count = 0
	for pcd in tqdm(pcds[1:]):
		count += 1
		# get base from registrated pcd list
		pcd_base = reged_pcds[-1]
		pcd_trans = pcd
		# registration
		T = overlap_predator(pcd_trans, pcd_base) # pcd_base
		pcd_trans.transform(T)
		# color registration
		# print('=> Color ICP Registration..')
		T = colored_icp(pcd_trans, pcd_base) # pcd_base
		pcd_trans.transform(T)
		# stored pcd
		reged_pcds.append(pcd_trans)
		# merged_pcd = merge_pcds(reged_pcds)
		# o3d.visualization.draw_geometries([merged_pcd])
		# o3d.io.write_point_cloud("./outputs/Appended.ply", merged_pcd)
	merged_pcd = merge_pcds(reged_pcds)
	o3d.io.write_point_cloud(config['outputs_dir']+"Appended.ply", merged_pcd)
	# o3d.visualization.draw_geometries(reged_pcds)

if __name__ == '__main__':
	config = {'path_dataset':"data/redwood-boardroom/",
				'outputs_dir': "outputs/",
				"down_sample": 1}
	run(config)
