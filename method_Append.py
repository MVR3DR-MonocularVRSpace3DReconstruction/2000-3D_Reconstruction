
import open3d as o3d
from utils import *
from colored_icp import *
from tqdm import tqdm
from overlap import overlap_predator

from core.deep_global_registration import DeepGlobalRegistration
from config import get_config

data_dir = "./data/redwood-bedroom/"
def deep_global_registration(source, target):
    print("=> Apply Deep Global Reg ")
    # o3d.visualization.draw_geometries([source, target])
    if 'DGR' not in globals():
        config = get_config()
        config.weights = "./pth/ResUNetBN2C-feat32-3dmatch-v0.05.pth"
        global DGR
        DGR = DeepGlobalRegistration(config)
    _transformation_dgr, _ = DGR.register(source, target)

    return _transformation_dgr

if __name__ == '__main__':
	pcds = read_point_clouds(data_dir=data_dir, down_sample=1)
	print("* Total "+str(len(pcds))+" point clouds loaded. ")

	# pose_graphs = read_pose_graph()
	# for idx in range(len(pcds)):
	# 	invT = pose_graphs[idx].nodes[len(pose_graphs[idx].nodes)//2].pose
	# 	pcds[idx].transform(np.linalg.inv(invT))


	o3d.visualization.draw_geometries(pcds)	
	reged_pcds = [pcds[0]]
	# merged_pcd = merge_pcds([pcds[0]])
	count = 0
	for pcd in tqdm(pcds[1:]):
		# print("==> Phase: {}// with {} times reg in total".format(count,len(pcds)))
		count += 1
		# get base from registrated pcd list
		pcd_base = reged_pcds[-1]
		pcd_trans = pcd
		# preprocessing
		# pcd_base.estimate_normals()
		# pcd_trans.estimate_normals()
		# print('=> Overlap Registration..')
		# registration
		T = overlap_predator(pcd_trans, pcd_base) # pcd_base
		pcd_trans.transform(T)

		# T = deep_global_registration(pcd_trans, pcd_base)
		# pcd_trans.transform(T)
		# color registration
		# print('=> Color ICP Registration..')
		T = colored_icp(pcd_trans, pcd_base) # pcd_base
		pcd_trans.transform(T)
		# stored pcd
		reged_pcds.append(pcd_trans)
		# merged_pcd = merge_pcds([merged_pcd, pcd_trans])
		# o3d.visualization.draw_geometries([merged_pcd])
		# o3d.io.write_point_cloud("./outputs/Appended.ply", merged_pcd)

	merged_pcd = merge_pcds(reged_pcds)
	o3d.visualization.draw_geometries(reged_pcds)

