
import numpy as np
from tqdm import tqdm
import open3d as o3d
from torch import optim, nn
from torch.utils.data import Dataset
from easydict import EasyDict as edict
import os, torch, time, json, glob, sys, copy, argparse
cwd = os.getcwd()
sys.path.append(cwd)
# from overlap_predator.datasets.indoor import IndoorDataset
from overlap_predator.datasets.dataloader import get_dataloader
from overlap_predator.models.architectures import KPFCNN
from overlap_predator.lib.utils import setup_seed, load_config
from overlap_predator.lib.benchmark_utils import ransac_pose_estimation

setup_seed(0)

class ThreeDMatch(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """
    def __init__(self, config, src_pcd, tgt_pcd):
        super(ThreeDMatch,self).__init__()
        self.config = config
        self.src_pcd = src_pcd
        self.tgt_pcd = tgt_pcd

    def __len__(self):
        return 1

    def __getitem__(self, item): 
        # src_pcd = o3d.io.read_point_cloud(self.src_pcd)
        # tgt_pcd = o3d.io.read_point_cloud(self.tgt_pcd)
        src_pcd = self.src_pcd.voxel_down_sample(0.025)
        tgt_pcd = self.tgt_pcd.voxel_down_sample(0.025)
        src_pcd = np.array(src_pcd.points).astype(np.float32)
        tgt_pcd = np.array(tgt_pcd.points).astype(np.float32)


        src_feats=np.ones_like(src_pcd[:,:1]).astype(np.float32)
        tgt_feats=np.ones_like(tgt_pcd[:,:1]).astype(np.float32)

        # fake the ground truth information
        rot = np.eye(3).astype(np.float32)
        trans = np.ones((3,1)).astype(np.float32)
        correspondences = torch.ones(1,2).long()

        return src_pcd, tgt_pcd, src_feats, tgt_feats, rot,trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)

def overlap_predator(source_pcd, target_pcd):
    if 'config' not in locals().keys():
        ###############################################
        # load configs
        ###############################################
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default="overlap_predator/config.yaml")
        args = parser.parse_args()
        config = load_config(args.config)
        config = edict(config)
        if config.gpu_mode:
            config.device = torch.device('cuda')
        else:
            config.device = torch.device('cpu')
        ###############################################
        # model initialization
        ###############################################
        config.architecture = ['simple', 'resnetb']

        for i in range(config.num_layers-1):
            config.architecture.append('resnetb_strided')
            config.architecture.append('resnetb')
            config.architecture.append('resnetb')
        for i in range(config.num_layers-2):
            config.architecture.append('nearest_upsample')
            config.architecture.append('unary')
        config.architecture.append('nearest_upsample')
        config.architecture.append('last_unary')
        config.model = KPFCNN(config).to(config.device)
    # print("=> Overlap predict. ")
    config.model.eval()
    ###############################################
    # load pretrained weights
    ###############################################
    assert config.pretrain != None
    state = torch.load(config.pretrain)
    config.model.load_state_dict(state['state_dict'])
    ###############################################
    # create dataset and dataloader
    ###############################################
    data_set = ThreeDMatch(config, source_pcd, target_pcd)
    neighborhood_limits = [0, 0, 0, 0]
    data_loader, _ = get_dataloader(dataset=data_set,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=1,
                                    neighborhood_limits=neighborhood_limits)
    
    c_loader_iter = data_loader.__iter__()
    with torch.no_grad():
        inputs = c_loader_iter.next()
        ###############################################
        # load inputs to device.
        ###############################################
        for k, v in inputs.items():  
            if type(v) == list:
                inputs[k] = [item.to(config.device) for item in v]
            else:
                inputs[k] = v.to(config.device)
        ###############################################
        # forward pass
         ###############################################
        feats, scores_overlap, scores_saliency = config.model(inputs)  #[N1, C1], [N2, C2]
        pcd = inputs['points'][0]
        len_src = inputs['stack_lengths'][0][0]
        
        src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
        src_feats, tgt_feats = feats[:len_src].detach().cpu(), feats[len_src:].detach().cpu()
        src_overlap, src_saliency = scores_overlap[:len_src].detach().cpu(), scores_saliency[:len_src].detach().cpu()
        tgt_overlap, tgt_saliency = scores_overlap[len_src:].detach().cpu(), scores_saliency[len_src:].detach().cpu()

        ###############################################
        # do probabilistic sampling guided by the score
        ###############################################
        src_scores = src_overlap * src_saliency
        tgt_scores = tgt_overlap * tgt_saliency

        if(src_pcd.size(0) > config.n_points):
            idx = np.arange(src_pcd.size(0))
            probs = (src_scores / src_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size= config.n_points, replace=False, p=probs)
            src_pcd, src_feats = src_pcd[idx], src_feats[idx]
        if(tgt_pcd.size(0) > config.n_points):
            idx = np.arange(tgt_pcd.size(0))
            probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size= config.n_points, replace=False, p=probs)
            tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

        ###############################################
        # run ransac and draw registration
        ###############################################
        tsfm = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False)

    return tsfm
