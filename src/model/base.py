import numpy as np
import logging
from pytorch_lightning import LightningModule
import torch
from pathlib import Path
from einops import rearrange
from torch import Tensor
from typing import List
from os.path import exists
from src.utils.genutils import cast_dict_to_tensors
from src.model.utils.tools import pack_to_render

# A logger for this file
log = logging.getLogger(__name__)

# Monkey patch SMPLH faster
from src.utils.file_io import hack_path

class BaseModel(LightningModule):
    def __init__(self, statistics_path: str, nfeats: int, 
                 norm_type: str,
                 input_feats: List[str], 
                 dim_per_feat: List[int],
                 smpl_path: str, num_vids_to_render: str,
                 renderer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False, 
                                  ignore=['eval_model','renderer']) # ignore TEMOS score

        # Save visuals, one validation step per validation epoch
        self.render_data_buffer = {"train": [], "val":[]}
        self.set_buf = {
                        'text_cond': [],
                        'mot_cond': [],
                        'full_cond': []
                       }

        self.loss_dict = {'train': None,
                          'val': None,}
        self.stats = self.load_norm_statistics(statistics_path, self.device)

        self.nfeats = nfeats
        self.dim_per_feat = dim_per_feat
        self.norm_type = norm_type
        self.first_pose_feats_dims = [3, 6, 21*6]
        self.first_pose_feats = ['body_transl', 'body_orient', 'body_pose']
        self.input_feats_dims = list(dim_per_feat)
        self.input_feats = list(input_feats)
        self.num_vids_to_render = num_vids_to_render
        smpl_path = hack_path(smpl_path, keyword='data')

        from aitviewer.models.smpl import SMPLLayer
        if renderer is not None:
            self.smpl_ait = SMPLLayer(model_type='smplh',
                                    ext='npz',
                                    gender='neutral')
        log.info(f'Using these features: {self.input_feats}')
        data_path = Path(hack_path(smpl_path, keyword='data')).parent
        # self.test_subset = joblib.load(data_path / 'test_kinedit.pth.tar')
        self.paths_of_rendered_subset = []
        self.paths_of_rendered_subset_tgt = []
        self.paths_of_rendered_subset_src = []

    def __post_init__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())
        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable


    def load_norm_statistics(self, path, device):
        # workaround for cluster local/sync
        path = hack_path(path)
        assert exists(path)
        stats = np.load(path, allow_pickle=True)[()]
        return cast_dict_to_tensors(stats, device=device)

    def run_smpl_fwd(self, body_transl, body_orient, body_pose, fast=True):
        if len(body_transl.shape) > 2:
            body_transl = body_transl.flatten(0, 1)
            body_orient = body_orient.flatten(0, 1)
            body_pose = body_pose.flatten(0, 1)
  
        batch_size = body_transl.shape[0]
        from src.tools.transforms3d import transform_body_pose
        self.body_model.batch_size = batch_size
        if fast:
            return self.body_model.smpl_forward_fast(transl=body_transl,
                                body_pose=transform_body_pose(body_pose,
                                                                'aa->rot'),
                                global_orient=transform_body_pose(body_orient,
                                                                    'aa->rot'))
        else:
            return self.body_model(transl=body_transl,
                                body_pose=transform_body_pose(body_pose,
                                                                'aa->rot'),
                                global_orient=transform_body_pose(body_orient,
                                                                    'aa->rot'))


    



    def norm_and_cat(self, batch, features_types):
        """
        turn batch data into the format the forward() function expects
        """
        seq_first = lambda t: rearrange(t, 'b s ... -> s b ...') 
        input_batch = {}
        ## PREPARE INPUT ##
        motion_condition = any('source' in value for value in batch.keys())
        if motion_condition:
            mo_types = ['source', 'target']
        else:
            mo_types = ['target']
            self.motion_condition = None
        for mot in mo_types:
            list_of_feat_tensors = [seq_first(batch[f'{feat_type}_{mot}']) 
                                    for feat_type in features_types if f'{feat_type}_{mot}' in batch.keys()]
            # normalise and cat to a unified feature vector
            list_of_feat_tensors_normed = self.norm_inputs(list_of_feat_tensors,
                                                           features_types)
            # list_of_feat_tensors_normed = [x[1:] if 'delta' in nx else x for nx,
                                                # x in zip(features_types, 
                                                # list_of_feat_tensors_normed)]
            x_norm, _ = self.cat_inputs(list_of_feat_tensors_normed)
            input_batch[mot] = x_norm
        return input_batch
    
    def norm_and_cat_single_motion(self, batch, features_types):
        """
        turn batch data into the format the forward() function expects
        """
        seq_first = lambda t: rearrange(t, 'b s ... -> s b ...') 
        input_batch = {}
        ## PREPARE INPUT ##
            
        list_of_feat_tensors = [seq_first(batch[feat_type]) 
                                for feat_type in features_types]
        # normalise and cat to a unified feature vector
        list_of_feat_tensors_normed = self.norm_inputs(list_of_feat_tensors,
                                                        features_types)
        # list_of_feat_tensors_normed = [x[1:] if 'delta' in nx else x for nx,
                                            # x in zip(features_types, 
                                            # list_of_feat_tensors_normed)]
        
        x_norm, _ = self.cat_inputs(list_of_feat_tensors_normed)
        input_batch['motion'] = x_norm
        return input_batch
    
    def append_first_frame(self, batch, which_motion):

        seq_first = lambda t: rearrange(t, 'b s ... -> s b ...') 
        list_of_feat_tensors = [seq_first(batch[f'{feat_type}_{which_motion}']) 
                                for feat_type in self.first_pose_feats]
        seqlen, bsz = list_of_feat_tensors[0].shape[:2]
        norm_pose_smpl = self.norm_inputs(list_of_feat_tensors,
                                          self.first_pose_feats)
        norm_pose_smpl = torch.cat(norm_pose_smpl, dim=-1)
        ## PAD THE INITIAL POSE ##
        padding_sz = np.sum(self.input_feats_dims) - np.sum(self.first_pose_feats_dims)
        norm_pose_smpl_pad = torch.zeros(1, bsz,
                                    batch[f'{which_motion}_motion'].shape[-1],
                                    device=self.device)
        norm_pose_smpl_pad[:, :, :norm_pose_smpl.shape[-1]] = norm_pose_smpl[:1]
        batch[f'{which_motion}_motion'] = torch.cat([norm_pose_smpl_pad,
                                                    batch[f'{which_motion}_motion']
                                                    ],
                                                    dim=0)

        return batch

    def norm(self, x, stats):
        if self.norm_type == "standardize":
            mean = stats['mean'].to(self.device)
            std = stats['std'].to(self.device)
            return (x - mean) / (2*(std + 1e-5))
        elif self.norm_type == "min_max":
            max = stats['max'].to(self.device)
            min = stats['min'].to(self.device)
            assert ((x - min) / (max - min + 1e-5)).min() >= 0
            assert ((x - min) / (max - min + 1e-5)).max() <= 1
            return (x - min) / (max - min + 1e-5)

    def unnorm(self, x, stats):
        if self.norm_type == "standardize":
            mean = stats['mean'].to(self.device)
            std = stats['std'].to(self.device)
            return x * 2 * (std + 1e-5) + mean
        elif self.norm_type == "min_max":
            max = stats['max'].to(self.device)
            min = stats['min'].to(self.device)
            return x * (max - min + 1e-5) + min

    def unnorm_state(self, state_norm: Tensor) -> Tensor:
        # unnorm state
        return self.cat_inputs(
            self.unnorm_inputs(self.uncat_inputs(state_norm,
                                                 self.first_pose_feats_dims),
                               self.first_pose_feats))[0]
        
    def unnorm_delta(self, delta_norm: Tensor) -> Tensor:
        # unnorm delta
        return self.cat_inputs(
            self.unnorm_inputs(self.uncat_inputs(delta_norm,
                                                 self.input_feats_dims),
                               self.input_feats))[0]

    def norm_state(self, state:Tensor) -> Tensor:
        # normalise state
        return self.cat_inputs(
            self.norm_inputs(self.uncat_inputs(state, 
                                               self.first_pose_feats_dims),
                             self.first_pose_feats))[0]

    def norm_delta(self, delta:Tensor) -> Tensor:
        # normalise delta
        return self.cat_inputs(
            self.norm_inputs(self.uncat_inputs(delta, self.input_feats_dims),
                             self.input_feats))[0]

    def cat_inputs(self, x_list: List[Tensor]):
        """
        cat the inputs to a unified vector and return their lengths in order
        to un-cat them later
        """
        return torch.cat(x_list, dim=-1), [x.shape[-1] for x in x_list]
    
    def uncat_inputs(self, x: Tensor, lengths: List[int]):
        """
        split the unified feature vector back to its original parts
        """
        return torch.split(x, lengths, dim=-1)
    
    def norm_inputs(self, x_list: List[Tensor], names: List[str]):
        """
        Normalise inputs using the self.stats metrics
        """
        x_norm = []
        for x, name in zip(x_list, names):
            
            x_norm.append(self.norm(x, self.stats[name]))
        return x_norm

    def unnorm_inputs(self, x_list: List[Tensor], names: List[str]):
        """
        Un-normalise inputs using the self.stats metrics
        """
        x_unnorm = []
        for x, name in zip(x_list, names):
            x_unnorm.append(self.unnorm(x, self.stats[name]))
        return x_unnorm



    


    


    def prepare_mot_masks(self, source_lens, target_lens, max_len=300):
        from src.data.tools.tensors import lengths_to_mask
        import torch.nn.functional as F
        mask_target = lengths_to_mask(target_lens,
                                              self.device)
        mask_source = lengths_to_mask(source_lens, self.device)

        if not source_lens[0] == 1 and max_len is not None:
            padded_mask_target = F.pad(mask_target,
                                    (0, max_len - mask_target.size(1)),
                                    value=0)
            padded_mask_source = F.pad(mask_source,
                                    (0, max_len - mask_source.size(1)),
                                    value=0)
            
            return padded_mask_source, padded_mask_target
        else:
            return mask_source, mask_target

    @torch.no_grad()
    def process_batch(self, batch):
        batch_to_gpu = { k: v.to(self.device) for k,
                            v in self.test_subset.items() 
                         if torch.is_tensor(v) }

        input_batch = self.norm_and_cat(batch_to_gpu, self.input_feats)
        for k, v in input_batch.items():
            if self.input_deltas:
                batch[f'{k}_motion'] = v[1:]
            else:
                batch[f'{k}_motion'] = v
                batch[f'length_{k}'] = [v.shape[0]] * v.shape[1]
        return batch

        

    def batch2motion(self, batch, pack_to_dict=True,
                     slice_til=None, single_motion=False):
        # batch_to_cpu = { k: v.detach().cpu() for k, v in batch.items() 
        #                 if torch.is_tensor(v) }
        tot_dim_deltas = 0
        if self.using_deltas:
            for idx_feat, in_feat in enumerate(self.input_feats):
                if 'delta' in in_feat:
                    tot_dim_deltas += self.input_feats_dims[idx_feat]
        source_motion_gt = None
        if batch['source_motion'] is not None:
            # source motion
            source_motion = batch['source_motion']
            # source_motion = self.unnorm_delta(source_motion)[..., tot_dim_deltas:]
            source_motion = self.diffout2motion(source_motion.detach().permute(1,
                                                                            0,
                                                                            2))
            source_motion = source_motion.detach().cpu()
            if pack_to_dict:
                source_motion_gt = pack_to_render(rots=source_motion[..., 3:],
                                                  trans=source_motion[...,:3])
            else:
                source_motion_gt = source_motion
            if slice_til is not None:
                source_motion_gt = {k: v[slice_til] 
                                    for k, v in source_motion_gt.items()}

        # target motion
        target_motion = batch['target_motion']
        target_motion = self.diffout2motion(target_motion.detach().permute(1,
                                                                        0,
                                                                        2))
        target_motion = target_motion.detach().cpu()
        if pack_to_dict:
            target_motion_gt = pack_to_render(rots=target_motion[..., 3:],
                                            trans=target_motion[...,:3])
        else:
            target_motion_gt = target_motion
        if slice_til is not None:
            target_motion_gt = {k: v[slice_til] 
                                for k, v in target_motion_gt.items()}

        return source_motion_gt, target_motion_gt

    # might be needed not working in multi GPU --> all_split_end
    # Logging per joint things 
    
