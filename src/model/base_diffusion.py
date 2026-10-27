from typing import List, Optional
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from src.data.tools.tensors import lengths_to_mask
from src.model.base import BaseModel
from src.model.utils.tools import remove_padding
from src.tools.transforms3d import transform_body_pose
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss
import torch
import torch.distributions as dist
import logging
from src.repr_utils.tmr_utils import load_tmr_model

from src.tmr.data.motionfix_loader import Normalizer


log = logging.getLogger(__name__)


#     # Initialize the output tensor with the same size as input
#     categorized_tensor = torch.zeros_like(input_tensor, dtype=torch.long)

#     # Apply the categorization rules
#     categorized_tensor[(input_tensor > threshold_1) & (input_tensor <= threshold_2)] = 1
#     categorized_tensor[input_tensor > threshold_2] = 2

#     return categorized_tensor

def assign_class(values, n):
    # Create class boundaries using torch
    boundaries = torch.linspace(0, 1, n + 1)
    
    # Initialize a tensor to hold the class labels
    class_labels = torch.zeros_like(values, dtype=torch.long)
    
    # Assign classes based on the boundaries
    for i in range(n):
        mask = (values >= boundaries[i]) & (values < boundaries[i + 1])
        class_labels[mask] = i
    
    # Handle the case where values are exactly 1
    class_labels[values == 1] = n - 1
    # class_labels = torch.ones_like(class_labels) # debug
    # Create V-shaped weights for the classes
    center = (n - 1) / 2
    weights = torch.abs(torch.arange(n) - center) / center
    weights = weights.to(values.device)
    return class_labels, weights

def calc_class(values, score_mask):
    # Ensure the mask and values are float tensors for computation
    values = values.float()
    score_mask = score_mask.float()

    # Multiply values by score_mask to zero out invalid entries
    masked_values = values * score_mask

    # Calculate the sum of valid values for each sample (B dimension)
    valid_sums = masked_values.sum(dim=1)

    # Count the number of valid entries for each sample
    valid_counts = score_mask.sum(dim=1)

    # Avoid division by zero by setting counts of 0 to 1 (no valid entries scenario)
    valid_counts = torch.where(valid_counts == 0, torch.ones_like(valid_counts), valid_counts)

    # Calculate the average by dividing the sum of valid values by the count of valid entries
    averages = valid_sums / valid_counts
    # B
    return (averages > 0.5).long()

class MD(BaseModel):
    def __init__(self, 
                 text_encoder: DictConfig,
                 motion_condition_encoder: DictConfig,
                 denoiser: DictConfig,
                 losses: DictConfig,
                 diff_params: DictConfig,
                 latent_dim: int,
                 nfeats: int,
                 input_feats: List[str],
                 statistics_path: str,
                 dim_per_feat: List[int],
                 norm_type: str,
                 smpl_path: str,
                 render_vids_every_n_epochs: Optional[int] = None,
                 num_vids_to_render: Optional[int] = None,
                 reduce_latents: Optional[str] = None,
                 condition: Optional[str] = "text",
                 motion_condition: Optional[str] = "source",
                 loss_func_pos: str = 'mse', # l1 mse
                 loss_func_feats: str = 'mse', # l1 mse
                 renderer = None,
                 pad_inputs = False,
                 source_encoder: str = 'trans_enc',
                 zero_len_source: bool = True,
                 copy_target: bool = False,
                 old_way: bool = False,
                 **kwargs):

        super().__init__(statistics_path, nfeats, norm_type, input_feats,
                         dim_per_feat, smpl_path, num_vids_to_render,
                         renderer=renderer)

        if set(["body_transl_delta_pelv_xy", "body_orient_delta",
                "body_pose_delta"]).issubset(self.input_feats):
            self.using_deltas = True
        else:
            self.using_deltas = False

        transl_feats = [x for x in self.input_feats if 'transl' in x]
        if set(transl_feats).issubset(["body_transl_delta", "body_transl_delta_pelv",
                                  "body_transl_delta_pelv_xy"]):
            self.using_deltas_transl = True
        else:
            self.using_deltas_transl = False
        self.zero_len_source = zero_len_source
        self.copy_target = copy_target
        self.old_way = old_way
        self.smpl_path = smpl_path
        self.condition = condition
        self.motion_condition = motion_condition
        if self.motion_condition == 'source':
            if source_encoder == 'trans_enc':
                self.motion_cond_encoder = instantiate(motion_condition_encoder)
            else:
                self.motion_cond_encoder = None
        self.pad_inputs = pad_inputs 
        self.text_encoder = instantiate(text_encoder)

        # for k, v in self.render_data_buffer.items():
        #     self.store_examples[k] = {'ref': [], 'ref_features': [], 'keyids': []}
        # self.metrics = ComputeMetrics(smpl_path)
        self.input_feats = input_feats
        self.render_vids_every_n_epochs = render_vids_every_n_epochs
        self.num_vids_to_render = num_vids_to_render
        self.renderer = renderer

        # If we want to overide it at testing time
        self.reduce_latents = reduce_latents
        self.latent_dim = latent_dim
        self.diff_params = diff_params
        denoiser.motion_condition = self.motion_condition
        self.denoiser = instantiate(denoiser)
        from src.diffusion import create_diffusion


        self.diffusion_process = create_diffusion(timestep_respacing=None,
                                     learn_sigma=False,
                                     sigma_small=True,
                                     diffusion_steps=self.diff_params.num_train_timesteps,
                                     noise_schedule=self.diff_params.noise_schedule,
                                     predict_type=self.diff_params.predict_type)  # 'sample' | 'noise' | 'v'
        if getattr(self.denoiser, 'use_identity_gate', False) or \
           getattr(self.denoiser, 'source_anchored', False):
            # give the denoiser the schedule so v-pred outputs can be blended/anchored
            # in x̂0-space (SG-DSD gate / Source-Anchored Head)
            self.denoiser.set_noise_schedule(self.diffusion_process.alphas_cumprod)

        shape = 2.0
        scale = 1.0
        self.tsteps_distr = dist.Gamma(torch.tensor(shape),
                                       torch.tensor(scale))
        self.loss_params = losses
        # loss params terrible
        if loss_func_pos == 'l1':
            self.loss_func_pos = l1_loss
        elif loss_func_pos in ['mse', 'l2']:
            self.loss_func_pos = mse_loss
        elif loss_func_pos in ['sl1']:
            self.loss_func_pos = smooth_l1_loss

        if loss_func_feats == 'l1':
            self.loss_func_feats = l1_loss
        elif loss_func_feats in ['mse', 'l2']:
            self.loss_func_feats = mse_loss
        elif loss_func_feats in ['sl1']:
            self.loss_func_feats = smooth_l1_loss
        self.validation_step_outputs = []
        self.use_regression = kwargs.get('use_regression', False)
        if self.use_regression:
            print('using regression')
        self.use_repr = kwargs.get('use_repr', False)
        self.use_cls = kwargs.get('use_cls', False)
        self.use_binary = kwargs.get('use_binary', False)
        if self.use_repr:
            # load repr matching models
            self.tmr_model = load_tmr_model()
            self.tmr_model.eval()
            self.tmr_model.to(self.device)
            self.tmr_model.requires_grad_(False)
        self.source_align_coef = kwargs.get('source_align_coef', 0)
        self.target_align_coef = kwargs.get('target_align_coef', 0)
        self.target_align_depth = kwargs.get('target_align_depth', 6)
        self.cls_coef = kwargs.get('cls_coef', 0)
        self.hf_loss_coef = kwargs.get('hf_loss_coef', 0)
        # x̂0-space source-anchor loss coef (stable anchor; trains the zero-init gate)
        self.anchor_x0_coef = kwargs.get('anchor_x0_coef', 0.0)
        self.anchor_space = kwargs.get('anchor_space', 'x0')  # 'x0' (paper) | 'v' (SPD-v control)
        self.diffusion_process.anchor_space = self.anchor_space  # diffusion_process is created above
        self.uniform_tsteps = kwargs.get('uniform_tsteps', False)
        self.residual_edit = kwargs.get('residual_edit', False)  # diffuse target−source
        # SG-DSD gate supervision: BCE(gate, 1 - similarity) resampled to the
        # target timeline; keeps the gate anchored to the edit region
        self.gate_coef = kwargs.get('gate_coef', 0.05)
        self.gate_hard_target = kwargs.get('gate_hard_target', False)
        self.gate_commit_coef = kwargs.get('gate_commit_coef', 0.0)
        self.spatial_gate = kwargs.get('spatial_gate', False)
        if self.spatial_gate:
            import json as _json
            with open(kwargs.get('text2parts_file', 'data/text2parts.json')) as f:
                self.text2parts = _json.load(f)
        # direct edit-stream supervision (fixes under-constraint of the gated loss)
        self.edit_sup_coef = kwargs.get('edit_sup_coef', 0.0)
        if self.edit_sup_coef > 0:
            self.denoiser.expose_edit_ungated = True
        if kwargs.get('gate_detach_src', False):
            self.denoiser.gate_detach_src = True
        # freeze everything except the gate head (isolate gate learnability)
        if kwargs.get('freeze_except_gate', False):
            for n, p in self.named_parameters():
                p.requires_grad = 'gate_head' in n
            n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'[freeze_except_gate] trainable params: {n_tr}', flush=True)
        # keyword-similarity alignment: (old A, new B) phrases per sample id
        self.kw_align_coef = kwargs.get('kw_align_coef', 0)
        if self.kw_align_coef > 0:
            import json as _json
            with open(kwargs.get('kw_file')) as f:
                self.kw_dict = _json.load(f)
        self.n_cls = kwargs.get('n_cls', 3)
        from pathlib import Path as _Path
        _repo_root = _Path(__file__).resolve().parents[2]
        self.normalizer = Normalizer(str(_repo_root / 'eval-deps/stats/humanml3d/amass_feats'))
        self.use_v_weight = kwargs.get('use_v_weight', True)

        self.__post_init__()


        # 
        
        
        
    def _diffusion_reverse(self,
                           text_embeds, text_masks_from_enc, 
                           motion_embeds, cond_motion_masks,
                           inp_motion_mask, diff_process,
                           src_len, tgt_len,
                           init_vec=None,
                           init_from='noise',
                           gd_text=None, gd_motion=None, 
                           mode='full_cond',
                           return_init_noise=False,
                           steps_num=None,
                           inpaint_dict=None,
                           use_linear=False,
                           prob_way='3way',
                           show_progress=True):
        # guidance_scale_text: 7.5 #
        #  guidance_scale_motion: 1.5
        # init latents

        bsz = inp_motion_mask.shape[0]
        assert mode in ['full_cond', 'text_cond', 'mot_cond']
        assert inp_motion_mask is not None
        # len_to_gen = max(lengths) if not self.input_deltas else max(lengths) + 1
        if init_vec is None:
            initial_latents = torch.randn(
                (bsz, inp_motion_mask.shape[1], self.nfeats),
                device=inp_motion_mask.device,
                dtype=torch.float,
            )
        else:
            initial_latents = init_vec

        if gd_text is None:
            gd_scale_text = self.diff_params.guidance_scale_text
        else:
            gd_scale_text = gd_text

        if gd_motion is None:
            gd_scale_motion = self.diff_params.guidance_scale_motion
        else:
            gd_scale_motion = gd_motion

        if text_embeds is not None:
            max_text_len = text_embeds.shape[1]
        else:
            max_text_len = 0
        if self.motion_condition == 'source' and motion_embeds is not None:
            max_motion_len = cond_motion_masks.shape[1]
            text_masks = text_masks_from_enc.clone()
            if max_text_len==1:
                text_masks = torch.ones_like(text_masks[:,0:1])
            if self.zero_len_source or self.old_way:
                nomotion_mask = torch.zeros(bsz, max_motion_len, 
                            dtype=torch.bool).to(self.device)
            else:
                nomotion_mask = torch.ones(bsz, max_motion_len,
                            dtype=torch.bool).to(self.device)
            motion_masks = torch.cat([nomotion_mask, 
                                      cond_motion_masks, 
                                      cond_motion_masks],
                                    dim=0)
            aug_mask = torch.cat([text_masks,
                                  motion_masks],
                                 dim=1)

        else:
            if max_text_len > 1:
                # aug_mask = text_mask
                # text_mask_aux = torch.ones(2*bsz, max_text_len, 
                #             dtype=torch.bool).to(self.device)
                aug_mask = text_masks_from_enc
            else:
                aug_mask = torch.ones(2*bsz, max_text_len, 
                            dtype=torch.bool).to(self.device)

        # Setup classifier-free guidance:
        if motion_embeds is not None:
            z = torch.cat([initial_latents, initial_latents, initial_latents], 0)
        else:
            z = torch.cat([initial_latents, initial_latents], 0)

        # y_null = torch.tensor([1000] * n, device=device)
        # y = torch.cat([y, y_null], 0)
        if use_linear:
            max_steps_diff = diff_process.num_timesteps
        else:
            max_steps_diff = None
        
        if motion_embeds is not None:
            model_kwargs = dict(# noised_motion=latent_model_input,
                                # timestep=t,
                                in_motion_mask=torch.cat([inp_motion_mask,
                                                        inp_motion_mask,
                                                        inp_motion_mask], 0),
                                text_embeds=text_embeds,
                                condition_mask=aug_mask,
                                motion_embeds=torch.cat([torch.zeros_like(motion_embeds),
                                                        motion_embeds,
                                                        motion_embeds], 1),
                                guidance_motion=gd_motion,
                                guidance_text_n_motion=gd_text,
                                inpaint_dict=inpaint_dict,
                                max_steps=max_steps_diff,
                                prob_way=prob_way,
                                src_len=src_len*3,
                                tgt_len=tgt_len*3)
        else:
            model_kwargs = dict(# noised_motion=latent_model_input,
                    # timestep=t,
                    in_motion_mask=torch.cat([inp_motion_mask,
                                            inp_motion_mask], 0),
                    text_embeds=text_embeds,
                    condition_mask=aug_mask,
                    motion_embeds=None,
                    guidance_motion=gd_motion,
                    guidance_text_n_motion=gd_text,
                    inpaint_dict=inpaint_dict,
                    max_steps=max_steps_diff,
                    src_len=src_len*2,
                    tgt_len=tgt_len*2)

        # model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        # Sample images:
        samples = diff_process.p_sample_loop(self.denoiser.forward_with_guidance,
                                             z.shape, z, 
                                             clip_denoised=False, 
                                             model_kwargs=model_kwargs,
                                             progress=show_progress,
                                             device=initial_latents.device,)
        if motion_embeds is not None:
            samples = samples.chunk(3, dim=0)[0]
        else:
            samples = samples.chunk(2, dim=0)[0]
        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]

        final_diffout = samples.permute(1, 0, 2)
        if return_init_noise:
            return initial_latents, final_diffout
        else:
            return final_diffout
    


    def _align_src_to_tgt(self, src, src_lens, tgt_lens, T_out):
        """src: (T_src, B, D) -> (T_out, B, D), per-sample linear time-resample.
        Used for residual-space editing: diffuse target - aligned_source."""
        import torch.nn.functional as F
        src = src.permute(1, 0, 2)  # B, T_src, D
        B, D = src.shape[0], src.shape[-1]
        out = torch.zeros(B, T_out, D, device=src.device, dtype=src.dtype)
        for i in range(B):
            ls = int(src_lens[i]); lt = min(int(tgt_lens[i]), T_out)
            if ls < 1 or lt < 1:
                continue
            if ls == 1:
                out[i, :lt] = src[i, 0]
            else:
                seg = src[i, :ls].T[None]  # 1, D, ls
                out[i, :lt] = F.interpolate(seg.float(), size=lt, mode='linear',
                                            align_corners=True)[0].T.to(src.dtype)
        return out.permute(1, 0, 2)  # T_out, B, D















    


    def generate_motion(self, texts_cond, motions_cond,
                        mask_source, mask_target,
                        diffusion_process, 
                        gt_lens_src, gt_lens_tgt,
                        init_vec_method='noise', init_vec=None,
                        gd_text=None, gd_motion=None, 
                        return_init_noise=False, 
                        condition_mode='full_cond',
                        num_diff_steps=None, 
                        inpaint_dict=None,
                        use_linear=False,
                        prob_way='3way',
                        show_progress=True
                        ):
        bsz, seqlen_tgt = mask_target.shape
        feat_sz = sum(self.input_feats_dims)
        if texts_cond is not None:
            no_of_texts = len(texts_cond)
            texts_cond = ['']*no_of_texts + texts_cond
            if self.motion_condition == 'source':
                texts_cond = ['']*no_of_texts + texts_cond
            text_emb, text_mask = self.text_encoder(texts_cond)

        cond_emb_motion = None
        cond_motion_mask = None
        if self.motion_condition == 'source':
            bsz, seqlen_src = mask_source.shape
            if condition_mode == 'full_cond' or condition_mode == 'mot_cond' :
                if self.motion_cond_encoder is not None:
                    source_motion_condition = motions_cond

                    cond_emb_motion = self.motion_cond_encoder(source_motion_condition, 
                                                               mask_source)
                    # assuming encoding of a single token!
                    cond_emb_motion = cond_emb_motion.unsqueeze(0)
                    cond_motion_mask = torch.ones((mask_source.shape[0], 1),
                                                dtype=bool, device=self.device)
                else:
                    source_motion_condition = motions_cond
                    cond_emb_motion = source_motion_condition
                    cond_motion_mask = mask_source
            else:
                if self.motion_cond_encoder is not None:
                    cond_emb_motion = torch.zeros(1, bsz,
                                                  self.denoiser.latent_dim,
                                                   device=self.device)
                    cond_motion_mask = torch.ones((bsz, 1),
                                                dtype=bool, device=self.device)
                else:
                    cond_emb_motion = torch.zeros(seqlen_src, bsz, feat_sz,
                                                   device=self.device)
                    cond_motion_mask = torch.ones((bsz, 1),
                                                dtype=bool, device=self.device)

        if init_vec_method == 'noise_prev':
            init_diff_rev = init_vec
        elif init_vec_method == 'source':
            init_diff_rev = motions_cond
            tgt_len = mask_target.shape[-1]
            src_len = mask_source.shape[-1]
            if tgt_len > src_len:
                init_diff_rev = torch.cat([init_diff_rev,
                                            torch.zeros((tgt_len-src_len,
                                                          *init_diff_rev.shape[1:]),
                                                        device=self.device)],
                                            dim=0)
                init_diff_rev = init_diff_rev.permute(1, 0, 2)
            else:
                init_diff_rev = init_diff_rev[:tgt_len]
                init_diff_rev = init_diff_rev.permute(1, 0, 2)
        else:
            init_diff_rev = None
            # complete noise

        # 
        with torch.no_grad():
            if return_init_noise:
                init_noise, diff_out = self._diffusion_reverse(text_emb, 
                                                text_mask,
                                                cond_emb_motion,
                                                cond_motion_mask,
                                                mask_target, 
                                                diffusion_process,
                                                gt_lens_src, gt_lens_tgt,
                                                init_vec=init_diff_rev,
                                                init_from=init_vec_method,
                                                gd_text=gd_text, 
                                                gd_motion=gd_motion,
                                                return_init_noise=return_init_noise,
                                                mode=condition_mode,
                                                steps_num=num_diff_steps,
                                                inpaint_dict=inpaint_dict,
                                                use_linear=use_linear,
                                                prob_way=prob_way,
                                                show_progress=show_progress)
                if getattr(self, 'residual_edit', False) and motions_cond is not None:
                    aligned = self._align_src_to_tgt(motions_cond, gt_lens_src,
                                                     gt_lens_tgt, diff_out.shape[0])
                    diff_out = diff_out + aligned
                return init_noise, diff_out.permute(1, 0, 2)

            else:
                diff_out = self._diffusion_reverse(text_emb,
                                                text_mask,
                                                cond_emb_motion,
                                                cond_motion_mask,
                                                mask_target, 
                                                diffusion_process,
                                                gt_lens_src, gt_lens_tgt,
                                                init_vec=init_diff_rev,
                                                init_from=init_vec_method,
                                                gd_text=gd_text, 
                                                gd_motion=gd_motion,
                                                return_init_noise=return_init_noise,
                                                mode=condition_mode,
                                                steps_num=num_diff_steps,
                                                inpaint_dict=inpaint_dict,
                                                use_linear=use_linear,
                                                show_progress=show_progress)

            if getattr(self, 'residual_edit', False) and motions_cond is not None:
                # sampled δ + time-aligned source = target (add back in normalized space)
                aligned = self._align_src_to_tgt(motions_cond, gt_lens_src,
                                                 gt_lens_tgt, diff_out.shape[0])
                diff_out = diff_out + aligned
            return diff_out.permute(1, 0, 2)

                    

    #     first_pose = self.unnorm_state(first_pose_norm)
    #     delta_motion = self.unnorm_delta(delta_motion_norm)

 
    #     # rotate R_z
    #     root_vel = change_for(delta_motion[..., :3],
    #                           R_z.squeeze(), forward=False)

    #     new_state_pos = first_pose[..., :3].squeeze() + root_vel







    def data2motion(self, single_batch, feature_types, mot = 'source'):
        # 1, T, D
        B, T = single_batch[f'{feature_types[0]}_{mot}'].shape[:2]
        first_trans = torch.zeros(B, T, 3,
                                    device=self.device)[:, [0]]
        if 'z_orient_delta' in self.input_feats:
            first_orient_z = torch.eye(3, device=self.device).unsqueeze(0)  # Now the shape is (1, 1, 3, 3)
            first_orient_z = first_orient_z.repeat(B, 1, 1)  # Now the shape is (B, 1, 3, 3)
            first_orient_z = transform_body_pose(first_orient_z, 'rot->6d') 

            # --> first_orient_z convert to 6d
            # integrate z orient delta --> z component tof orientation
            z_orient_delta = single_batch[f'z_orient_delta_{mot}']

            from src.tools.transforms3d import apply_rot_delta, get_z_rot, change_for
            prev_z = first_orient_z 
            full_z_angle = [first_orient_z[:, None]]
            for i in range(1, z_orient_delta.shape[1]):
                curr_z = apply_rot_delta(prev_z, z_orient_delta[:, i])
                prev_z = curr_z.clone()
                full_z_angle.append(curr_z[:,None])
            full_z_angle = torch.cat(full_z_angle, dim=1)
            full_z_angle_rotmat = get_z_rot(full_z_angle)
            # full_orient = torch.cat([full_z_angle, xy_orient], dim=-1)
            xy_orient = single_batch[f'body_orient_xy_{mot}']
            xy_orient_rotmat = transform_body_pose(xy_orient, '6d->rot')
            # xy_orient = remove_z_rot(xy_orient, in_format="6d")

            # GLOBAL ORIENTATION
            # full_z_angle = transform_body_pose(full_z_angle_rotmat,
            #                                    'rot->6d')

            # full_global_orient = apply_rot_delta(full_z_angle,
            #                                      xy_orient)
            full_global_orient_rotmat = full_z_angle_rotmat @ xy_orient_rotmat
            full_global_orient = transform_body_pose(full_global_orient_rotmat,
                                                        'rot->6d')

            first_trans = first_trans # 1, 1, 3

            # apply deltas
            # get velocity in global c.f. and add it to the state position
            assert 'body_transl_delta_pelv' in self.input_feats
            pelvis_delta = single_batch[f'body_transl_delta_pelv_{mot}']
            trans_vel_pelv = change_for(pelvis_delta[:, 1:],
                                        full_global_orient_rotmat[:, :-1],
                                        forward=False)

            # new_state_pos = prev_trans_norm.squeeze() + trans_vel_pelv
            full_trans = torch.cumsum(trans_vel_pelv, dim=1) + first_trans
            full_trans = torch.cat([first_trans, full_trans], dim=1)


            full_rots = torch.cat([full_global_orient, 
                                    single_batch[f'body_pose_{mot}']],
                                    dim=-1)
            full_motion_unnorm = torch.cat([full_trans,
                                            full_rots], dim=-1)
            # translation +  rotation
            return full_motion_unnorm
            

    def diffout2motion(self, diffout):
        if diffout.shape[1] == 1:
            rots_unnorm = self.cat_inputs(self.unnorm_inputs(self.uncat_inputs(
                                                            diffout,
                                                            self.input_feats_dims
                                                            ),
                                          self.input_feats))[0]
            full_motion_unnorm = rots_unnorm
        else:
            feats_unnorm = self.cat_inputs(self.unnorm_inputs(
                                            self.uncat_inputs(diffout,
                                                self.input_feats_dims),
                                            self.input_feats))[0]
            # FIRST POSE FOR GENERATION & DELTAS FOR INTEGRATION
            if "body_joints_local_wo_z_rot" in self.input_feats:
                idx = self.input_feats.index("body_joints_local_wo_z_rot")
                feats_unnorm = feats_unnorm[..., :-self.input_feats_dims[idx]]

            first_trans = torch.zeros(*diffout.shape[:-1], 3,
                                      device=self.device)[:, [0]]
            if 'z_orient_delta' in self.input_feats:
                first_orient_z = torch.eye(3, device=self.device).unsqueeze(0)  # Now the shape is (1, 1, 3, 3)
                first_orient_z = first_orient_z.repeat(feats_unnorm.shape[0], 1, 1)  # Now the shape is (B, 1, 3, 3)
                first_orient_z = transform_body_pose(first_orient_z, 'rot->6d') 

                # --> first_orient_z convert to 6d
                # integrate z orient delta --> z component tof orientation
                z_orient_delta = feats_unnorm[..., 9:15]

                from src.tools.transforms3d import apply_rot_delta, get_z_rot, change_for
                prev_z = first_orient_z 
                full_z_angle = [first_orient_z[:, None]]
                for i in range(1, z_orient_delta.shape[1]):
                    curr_z = apply_rot_delta(prev_z, z_orient_delta[:, i])
                    prev_z = curr_z.clone()
                    full_z_angle.append(curr_z[:,None])
                full_z_angle = torch.cat(full_z_angle, dim=1)
                full_z_angle_rotmat = get_z_rot(full_z_angle)
                # full_orient = torch.cat([full_z_angle, xy_orient], dim=-1)
                xy_orient = feats_unnorm[..., 3:9]
                xy_orient_rotmat = transform_body_pose(xy_orient, '6d->rot')
                # xy_orient = remove_z_rot(xy_orient, in_format="6d")

                # GLOBAL ORIENTATION
                # full_z_angle = transform_body_pose(full_z_angle_rotmat,
                #                                    'rot->6d')

                # full_global_orient = apply_rot_delta(full_z_angle,
                #                                      xy_orient)
                full_global_orient_rotmat = full_z_angle_rotmat @ xy_orient_rotmat
                full_global_orient = transform_body_pose(full_global_orient_rotmat,
                                                         'rot->6d')

                first_trans = self.cat_inputs(self.unnorm_inputs(
                                                        [first_trans],
                                                        ['body_transl'])
                                                )[0]
                # TODO：isn't it always zero?

                # apply deltas
                # get velocity in global c.f. and add it to the state position
                assert 'body_transl_delta_pelv' in self.input_feats
                pelvis_delta = feats_unnorm[..., :3]
                trans_vel_pelv = change_for(pelvis_delta[:, 1:],
                                            full_global_orient_rotmat[:, :-1],
                                            forward=False)

                # new_state_pos = prev_trans_norm.squeeze() + trans_vel_pelv
                full_trans = torch.cumsum(trans_vel_pelv, dim=1) + first_trans
                full_trans = torch.cat([first_trans, full_trans], dim=1)


                full_rots = torch.cat([full_global_orient, 
                                       feats_unnorm[...,-21*6:]],
                                      dim=-1)
                full_motion_unnorm = torch.cat([full_trans,
                                                full_rots], dim=-1)

            elif "body_orient_delta" in self.input_feats:
                delta_trans = diffout[..., 6:9]
                pelv_orient = diffout[..., 9:15]

                # for i in range(1, delta_trans.shape[1]):
                full_trans_unnorm = self.integrate_translation(pelv_orient[:, :-1],
                                                            first_trans,
                                                            delta_trans[:, 1:])
                rots_unnorm = self.cat_inputs(self.unnorm_inputs(self.uncat_inputs(
                                                                diffout[..., 9:],
                                                        self.input_feats_dims[2:]),
                                                self.input_feats[2:])
                                                )[0]
                full_motion_unnorm = torch.cat([full_trans_unnorm,
                                                rots_unnorm], dim=-1)

            else:
                delta_trans = diffout[..., :3]
                pelv_orient = diffout[..., 3:9]
                # for i in range(1, delta_trans.shape[1]):
                full_trans_unnorm = self.integrate_translation(pelv_orient[:, :-1],
                                                            first_trans,
                                                            delta_trans[:, 1:])
                rots_unnorm = self.cat_inputs(self.unnorm_inputs(self.uncat_inputs(
                                                                diffout[..., 3:],
                                                        self.input_feats_dims[1:]),
                                                self.input_feats[1:])
                                                )[0]
                full_motion_unnorm = torch.cat([full_trans_unnorm,
                                                rots_unnorm], dim=-1)
        return full_motion_unnorm

