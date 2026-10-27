import torch
import torch.nn as nn
from torch import  nn
from src.model.utils.timestep_embed import TimestepEmbedding, Timesteps, TimestepEmbedderMDM
from src.model.utils.positional_encoding import PositionalEncoding
from src.model.utils.transf_utils import SkipTransformerEncoder, TransformerEncoderLayer
from src.model.utils.all_positional_encodings import build_position_encoding
from src.data.tools.tensors import lengths_to_mask
from src.model.utils.timestep_embed import TimestepEmbedderMDM
from src.model.DiT_models import DiTMotion, DiTMotionSkip


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

# architecure ablation
class DiT_Denoiser_CLS_Arch(nn.Module):

    def __init__(self,
                 nfeats: int = 263,
                 condition: str = "text",
                 motion_condition: str = None,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 text_encoded_dim: int = 768,
                 pred_delta_motion: bool = False,
                 use_sep: bool = True,
                 **kwargs) -> None:

        super().__init__()
        self.latent_dim = latent_dim
        self.text_encoded_dim = text_encoded_dim
        self.condition = condition
        self.feat_comb_coeff = nn.Parameter(torch.tensor([1.0]))
        self.pose_proj_in_source = nn.Linear(nfeats, self.latent_dim)
        self.pose_proj_in_target = nn.Linear(nfeats, self.latent_dim)
        self.pose_proj_out = nn.Linear(self.latent_dim, nfeats)
        self.motion_condition = motion_condition

        # emb proj
        if self.condition in ["text", "text_uncond"]:
            # text condition
            # project time+text to latent_dim
            if text_encoded_dim != self.latent_dim:
                # todo 10.24 debug why relu
                self.emb_proj = nn.Linear(text_encoded_dim, self.latent_dim)
        else:
            raise TypeError(f"condition type {self.condition} not supported")
        self.use_sep = True
        self.query_pos = PositionalEncoding(self.latent_dim, dropout = 0)
        self.cond_pos = PositionalEncoding(self.latent_dim, dropout = 0)# don't want to introduce noise here
        if self.motion_condition == "source":
            self.sep_token = nn.Parameter(torch.randn(1, self.latent_dim))

        # dit encoder; long skips route temporal detail past the block stack
        dit_cls = DiTMotionSkip if kwargs.get('use_long_skips', False) else DiTMotion
        self.dit_encoder = dit_cls(
            in_channels=self.latent_dim,
            hidden_size=self.latent_dim,
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=ff_size / self.latent_dim,
        )
        # use torch transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation)
        self.cond_encoder = nn.TransformerEncoder(encoder_layer,
                                                num_layers=kwargs.get('encoder_layers', 4))
        self.has_repr =  True
        self.source_head = build_mlp(self.latent_dim, 1024, kwargs['n_cls'])# classification
        self.target_head = build_mlp(self.latent_dim, 1024, kwargs['repr_dim'])
        self.use_identity_gate = kwargs.get('use_identity_gate', False)
        self.edit_is_v = (kwargs.get('predict_type', 'sample') == 'v')
        # Source-Anchored Head (SAD): output = align(source) + zero-init proj(Δ),
        # an architectural identity skip from the source. Δ predicted in x̂0-space;
        # zero-init keeps the start at identity (output≈source) then learns edits.
        self.source_anchored = kwargs.get('source_anchored', False)
        if self.source_anchored:
            self.delta_proj = nn.Linear(nfeats, nfeats)
            nn.init.zeros_(self.delta_proj.weight); nn.init.zeros_(self.delta_proj.bias)
        # FD-SAD: a temporal high-frequency branch that補s back the detail the
        # x0-mean crushes at low t. Depthwise temporal conv over target tokens,
        # zero-init output, gated by g(t)=sqrt(alpha_bar_t) so it activates at
        # LOW noise (where identity-failure destroys high frequencies).
        self.freq_decomp = kwargs.get('freq_decomp', False)
        if self.freq_decomp:
            hf_k = 5
            self.hf_in = nn.Linear(nfeats, self.latent_dim)
            self.hf_dw = nn.Conv1d(self.latent_dim, self.latent_dim, hf_k,
                                   padding=hf_k // 2, groups=self.latent_dim)
            self.hf_pw = nn.Conv1d(self.latent_dim, self.latent_dim, 1)
            self.hf_act = nn.GELU()
            self.hf_out = nn.Linear(self.latent_dim, nfeats)
            nn.init.zeros_(self.hf_out.weight); nn.init.zeros_(self.hf_out.bias)
            # fixed low-pass (normalized Gaussian) along time; high-pass = I - LP.
            # Analytically constrains the branch output to the HIGH band, so it
            # can only restore lost high-frequency detail, never touch low freq.
            import numpy as _np
            ksz, sig = 9, 2.0
            g = _np.exp(-(_np.arange(ksz) - ksz // 2) ** 2 / (2 * sig ** 2))
            g = (g / g.sum()).astype('float32')
            self.register_buffer('_lp_kernel', torch.tensor(g).view(1, 1, ksz),
                                 persistent=False)
        self.spatial_gate = kwargs.get('spatial_gate', False)  # per-part α(τ,p)
        self.n_parts = 6
        if self.use_identity_gate:
            n_out = self.n_parts if self.spatial_gate else 1
            self.gate_head = build_mlp(self.latent_dim, 1024, n_out)
            if self.spatial_gate:
                import numpy as _np
                from pathlib import Path as _PP
                p2f = _np.load(_PP(__file__).resolve().parents[2] / 'data/part2feat.npy')
                self.register_buffer('_part2feat', torch.from_numpy(p2f).float(),
                                     persistent=False)  # [6, 207]
        self.anchor_x0_gate = kwargs.get('anchor_x0_gate', False)
        # gate activation: 'tanh' (in (-1,1); can pull toward OR push away from src)
        # or 'sigmoid' (in (0,1); RePaint-style keep-only mask toward source)
        self.anchor_gate_mode = kwargs.get('anchor_gate_mode', 'tanh')
        # ablation knobs: field granularity ('full'=per-frame-per-dim | 'frame'=per-frame
        # scalar | 'global'=one scalar per sequence), zero-init on/off, alignment mode
        self.anchor_field_dim = kwargs.get('anchor_field_dim', 'full')
        self.anchor_zero_init = kwargs.get('anchor_zero_init', True)
        self.anchor_align = kwargs.get('anchor_align', 'linear')
        if self.anchor_x0_gate:
            out_d = nfeats if self.anchor_field_dim == 'full' else 1
            self.x0_gate_head = build_mlp(self.latent_dim, 1024, out_d)
            if self.anchor_zero_init:
                last = [m for m in self.x0_gate_head.modules() if isinstance(m, nn.Linear)][-1]
                nn.init.zeros_(last.weight)
                # start at identity-to-base: tanh(0)=0; sigmoid(-6)≈0.0025≈0
                nn.init.constant_(last.bias, -6.0 if self.anchor_gate_mode == 'sigmoid' else 0.0)
        self.use_kw_align = kwargs.get('use_kw_align', False)
        if self.use_kw_align:
            self.kw_text_proj = nn.Linear(text_encoded_dim, kwargs['repr_dim'])
            self.kw_src_proj = build_mlp(self.latent_dim, 1024, kwargs['repr_dim'])
            self.kw_tgt_proj = build_mlp(self.latent_dim, 1024, kwargs['repr_dim'])
        self.use_target_mask = kwargs.get('use_target_mask', False) 
        self.ablation = kwargs.get('ablation', None)
        # assert self.ablation in ['raw_text', 'no_text', 'raw_source', 'raw_text_and_source'], 'ablation type not supported'
        

    def _identity_stream(self, motion_embeds, src_len, tgt_len, T_out):
        """Time-align raw source features (T_src, B, D) to the target timeline.
        Returns (B, T_out, D) and a per-row has-source mask (B,)."""
        import torch.nn.functional as F
        src = motion_embeds.permute(1, 0, 2)  # B, T_src, D
        B = src.shape[0]
        out = torch.zeros(B, T_out, src.shape[-1], device=src.device, dtype=src.dtype)
        has_src = (src.abs().sum(dim=(1, 2)) > 1e-6).float()
        for i in range(B):
            ls = int(src_len[i]) if src_len is not None else src.shape[1]
            lt = min(int(tgt_len[i]) if tgt_len is not None else T_out, T_out)
            if ls < 2 or lt < 1:
                continue
            if getattr(self, 'anchor_align', 'linear') == 'pad':
                # ablation: no temporal alignment — crop, then repeat the last frame
                L = min(ls, lt)
                out[i, :L] = src[i, :L]
                if lt > L:
                    out[i, L:lt] = src[i, ls - 1]
            else:
                seg = src[i, :ls].T[None]  # 1, D, ls
                out[i, :lt] = F.interpolate(seg.float(), size=lt, mode='linear',
                                            align_corners=True)[0].T.to(src.dtype)
        return out, has_src

    def set_noise_schedule(self, alphas_cumprod):
        """Register the diffusion schedule so v-pred edit streams can be blended
        in x̂0-space (SG-DSD gate is defined on clean motion, not on v)."""
        import numpy as _np
        ac = torch.as_tensor(_np.asarray(alphas_cumprod), dtype=torch.float32)
        self.register_buffer('_sqrt_ac', torch.sqrt(ac), persistent=False)
        self.register_buffer('_sqrt_1m_ac', torch.sqrt(1.0 - ac), persistent=False)

    def _coef(self, t, x):
        sa = self._sqrt_ac.to(x.device)[t].view(-1, *([1] * (x.dim() - 1)))
        sm = self._sqrt_1m_ac.to(x.device)[t].view(-1, *([1] * (x.dim() - 1)))
        return sa, sm

    def _anchor_gate_act(self, logits):
        """Activate the x̂0-anchor gate. 'tanh'∈(-1,1) can pull toward or push
        away from source; 'sigmoid'∈(0,1) is a keep-only mask toward source."""
        if self.anchor_gate_mode == 'sigmoid':
            return torch.sigmoid(logits)
        return torch.tanh(logits)

    def _highpass(self, x):
        """x: (B,T,D) -> high-pass along time = x - lowpass(x), fixed Gaussian LP.
        Guarantees the returned signal contains only high temporal frequencies."""
        B, T, D = x.shape
        k = self._lp_kernel.to(x.device)              # 1,1,ksz
        pad = k.shape[-1] // 2
        xt = x.permute(0, 2, 1).reshape(B * D, 1, T)
        lp = torch.nn.functional.conv1d(xt, k, padding=pad).reshape(B, D, T).permute(0, 2, 1)
        return x - lp

    def _hf_branch(self, delta_x0, t):
        """High-frequency correction over the x̂0-space delta. Depthwise temporal
        conv → analytic high-pass projection (structurally high-band only) →
        zero-init out → gated by g(t)=sqrt(alpha_bar_t) (active at LOW noise,
        exactly where x0-mean destroys high frequencies)."""
        h = self.hf_in(delta_x0).transpose(1, 2)      # B, C, T
        h = self.hf_pw(self.hf_act(self.hf_dw(h))).transpose(1, 2)  # B, T, C
        hf = self._highpass(self.hf_out(h))           # zero-init, high-band only
        sa = self._sqrt_ac.to(delta_x0.device)[t].view(-1, *([1] * (delta_x0.dim() - 1)))
        return hf * sa                                 # activate at low t

    def _apply_source_anchor(self, edit_out, motion_embeds, src_len, tgt_len,
                             x_t=None, t=None, edit_is_v=False):
        """x̂0 = align(source) + zero-init-proj(Δ_low) [+ HF(Δ) at low t].
        edit_out is Δ in x̂0-space (x0-pred) or v (v-pred, converted)."""
        ident, has_src = self._identity_stream(motion_embeds, src_len, tgt_len,
                                               edit_out.shape[1])
        if edit_is_v:
            sa, sm = self._coef(t, edit_out)
            delta = sa * x_t - sm * edit_out                 # v -> x̂0 (the Δ)
            dl = self.delta_proj(delta)
            if self.freq_decomp:
                dl = dl + self._hf_branch(delta, t)
            x0 = ident + dl * has_src[:, None, None] \
                 + delta * (1.0 - has_src[:, None, None])    # no-source rows: pass Δ
            sm_safe = sm.clamp_min(0.1)
            return (sa * x_t - x0) / sm_safe
        dl = self.delta_proj(edit_out)
        if self.freq_decomp:
            dl = dl + self._hf_branch(edit_out, t)
        return ident + dl

    def _apply_gate(self, edit_out, gate_logits, motion_embeds, src_len, tgt_len,
                    x_t=None, t=None, edit_is_v=False):
        """Blend in x̂0-space. edit_out is x̂0 (x0-pred) or v (v-pred; converted
        via x_t,t). Returns the blended output in the SAME space as edit_out."""
        ident, has_src = self._identity_stream(motion_embeds, src_len, tgt_len,
                                               edit_out.shape[1])
        a = torch.sigmoid(gate_logits)
        if getattr(self, 'gate_hard_infer', False):
            a = (a > 0.5).float()   # inference-time hard commit (gate as localizer)
        elif getattr(self, 'gate_adapt_infer', False):
            # gate lives in a narrow high band; use its RANKING via a per-sample
            # adaptive threshold (mean over frames) -> edit the above-mean frames
            thr = a.mean(dim=1, keepdim=True)
            a = (a > thr).float()
        if self.spatial_gate and gate_logits.shape[-1] == self.n_parts:
            # (B,T,6) part logits -> (B,T,207) per-dim alpha via membership matrix
            alpha = a @ self._part2feat.to(gate_logits.device)
        else:
            alpha = a
        alpha = alpha * has_src[:, None, None] + (1.0 - has_src[:, None, None])
        if getattr(self, 'gate_detach_src', False) and self.training:
            ident = ident.detach()  # gradient only through edit stream + gate
        if edit_is_v:
            sa, sm = self._coef(t, edit_out)
            x0_edit = sa * x_t - sm * edit_out            # v -> x̂0
            x0_blend = alpha * x0_edit + (1.0 - alpha) * ident
            return (sa * x_t - x0_blend) / sm, gate_logits  # x̂0 -> v
        return alpha * edit_out + (1.0 - alpha) * ident, gate_logits

    def forward(self,
                noised_motion,
                timestep,
                in_motion_mask,
                text_embeds,
                condition_mask,
                motion_embeds=None,
                lengths=None,
                src_len = None,
                tgt_len = None,
                **kwargs):
        if isinstance(src_len, list):
            src_len = torch.tensor(src_len)
            tgt_len = torch.tensor(tgt_len)
        _x_t_gate = noised_motion  # (B, T, D) for x̂0-space v-pred gate blend

        # 0.  dimension matching
        # noised_motion [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        bs = noised_motion.shape[0]
        noised_motion = noised_motion.permute(1, 0, 2)
        # 0. check lengths for no vae (diffusion only)
        # if lengths not in [None, []]:
        motion_in_mask = in_motion_mask

        # time_embedding | text_embedding | frames_source | frames_target
        # 1 * lat_d | max_text * lat_d | max_frames * lat_d | max_frames * lat_d
        
        
        if self.condition in ["text", "text_uncond"]:
            # make it seq first
            text_embeds = text_embeds.permute(1, 0, 2)
            if self.text_encoded_dim != self.latent_dim:
                # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                text_emb_latent = self.emb_proj(text_embeds)
            else:
                text_emb_latent = text_embeds
            emb_latent_ori = text_emb_latent
            emb_latent = text_emb_latent
            # 1, B, D 

            if motion_embeds is not None:
                zeroes_mask = (motion_embeds == 0).all(dim=-1)
                if motion_embeds.shape[-1] != self.latent_dim:
                    motion_embeds_proj = self.pose_proj_in_source(motion_embeds)
                    motion_embeds_proj[zeroes_mask] = 0
                else:
                    motion_embeds_proj = motion_embeds
 
        else:
            raise TypeError(f"condition type {self.condition} not supported")
        # 4. transformer
        # if self.diffusion_only:
        proj_noised_motion = self.pose_proj_in_target(noised_motion)
        
        # BUILD the mask now
        if motion_embeds is None:
            aug_mask = torch.cat((condition_mask[:, :text_emb_latent.shape[0]],
                                  motion_in_mask), 1)
        else:
            sep_token_mask = torch.ones((bs, self.sep_token.shape[0]),
                                        dtype=bool,
                                        device=noised_motion.device)
            aug_mask = torch.cat((
                            condition_mask[:, text_emb_latent.shape[0]:],
                            sep_token_mask,
                            motion_in_mask,
                            ), 1)

        # NOTE: condition encoding
        # emb_latent: T_Text, B, D
        # motion_embeds_proj: T_max, B, D
        motion_embeds_proj_ori = motion_embeds_proj
        cond_seq = torch.cat((emb_latent, motion_embeds_proj), dim=0)
        cond_seq = self.cond_pos(cond_seq)
        cond_seq_processed = self.cond_encoder(cond_seq, src_key_padding_mask=~condition_mask)
        T_Text = emb_latent.size(0)
        T_max = motion_embeds_proj.size(0)
        emb_latent_processed, motion_embeds_proj_processed = torch.split(cond_seq_processed, [T_Text, T_max], dim=0)
        emb_latent, motion_embeds_proj = emb_latent_processed, motion_embeds_proj_processed

        if motion_embeds is None:
            xseq = proj_noised_motion
        else:      
            sep_token_batch = torch.tile(self.sep_token, (bs,)).reshape(bs,
                                                                        -1)
            if self.ablation in ['raw_source', 'raw_text_and_source']:
                xseq = torch.cat((motion_embeds_proj_ori,
                                sep_token_batch[None],
                                proj_noised_motion), axis=0)
            else:
                xseq = torch.cat((motion_embeds_proj,
                                sep_token_batch[None],
                                proj_noised_motion), axis=0)

        xseq = self.query_pos(xseq)
        if self.use_target_mask:
            mask = None
        else:
            mask = aug_mask
            
        if self.ablation == 'no_text':
            condition = torch.zeros_like(emb_latent[0])
        elif self.ablation == 'raw_text' or self.ablation == 'raw_text_and_source':
            condition = emb_latent_ori[0]
        elif self.ablation == 'raw_source':
            condition = emb_latent[0]

        tokens = self.dit_encoder(xseq.permute(1, 0, 2), timestep, condition, mask=mask)
        # B, T, D
        tokens = tokens.permute(1, 0, 2)
        # if self.diffusion_only:
        if motion_embeds is not None:
            denoised_motion_proj = tokens
            if self.use_sep:
                useful_tokens = motion_embeds_proj.shape[0]+1
            else:
                useful_tokens = motion_embeds_proj.shape[0]
            denoised_motion_proj = denoised_motion_proj[useful_tokens:]
        else:
            denoised_motion_proj = tokens

        denoised_motion = self.pose_proj_out(denoised_motion_proj)
        denoised_motion[~motion_in_mask.T] = 0
        # zero for padded area
        # else:
        #     sample = tokens[:sample.shape[0]]
        # 5. [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
        denoised_motion = denoised_motion.permute(1, 0, 2)
        if self.use_identity_gate and motion_embeds is not None:
            gate_logits = self.gate_head(denoised_motion_proj).permute(1, 0, 2)
            if kwargs.get('defer_gate', False):
                # guidance path: return ungated edit stream + gate so CFG can be
                # applied to the edit stream, then the gate blend done ONCE after
                return denoised_motion, gate_logits
            denoised_motion, _ = self._apply_gate(denoised_motion, gate_logits,
                                                  motion_embeds, src_len, tgt_len,
                                                  x_t=_x_t_gate, t=timestep,
                                                  edit_is_v=self.edit_is_v)
            denoised_motion = denoised_motion * in_motion_mask[..., None]
        elif self.source_anchored and motion_embeds is not None:
            if kwargs.get('defer_gate', False):
                return denoised_motion, None   # anchor once after CFG combine
            denoised_motion = self._apply_source_anchor(
                denoised_motion, motion_embeds, src_len, tgt_len,
                x_t=_x_t_gate, t=timestep, edit_is_v=self.edit_is_v)
            denoised_motion = denoised_motion * in_motion_mask[..., None]
        elif self.anchor_x0_gate and motion_embeds is not None:
            # inference always defers: return raw v + per-dim gate; the x̂0-space
            # anchor is applied in gaussian_diffusion AFTER v->x̂0, post-CFG.
            gate_logits = self.x0_gate_head(denoised_motion_proj).permute(1, 0, 2)
            if self.anchor_field_dim == 'global':
                gate_logits = gate_logits.mean(dim=1, keepdim=True)
            return denoised_motion, gate_logits
        return denoised_motion

    def forward_with_repr(self,
                noised_motion,
                timestep,
                in_motion_mask,
                text_embeds,
                condition_mask, 
                motion_embeds=None,
                lengths=None,
                src_len = None,
                tgt_len = None,
                target_align_depth = 6, 
                return_attention = False, 
                **kwargs):
        if isinstance(src_len, list):
            src_len = torch.tensor(src_len)
            tgt_len = torch.tensor(tgt_len)
        _x_t_gate = noised_motion  # (B, T, D) for x̂0-space v-pred gate blend

        # 0.  dimension matching
        # noised_motion [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        bs = noised_motion.shape[0]
        noised_motion = noised_motion.permute(1, 0, 2)
        # 0. check lengths for no vae (diffusion only)
        # if lengths not in [None, []]:
        motion_in_mask = in_motion_mask

        # time_embedding | text_embedding | frames_source | frames_target
        # 1 * lat_d | max_text * lat_d | max_frames * lat_d | max_frames * lat_d
        
        
        if self.condition in ["text", "text_uncond"]:
            # make it seq first
            text_embeds = text_embeds.permute(1, 0, 2)
            if self.text_encoded_dim != self.latent_dim:
                # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                text_emb_latent = self.emb_proj(text_embeds)
            else:
                text_emb_latent = text_embeds
            emb_latent_ori = text_emb_latent
            emb_latent = text_emb_latent
            # 1, B, D 

            if motion_embeds is not None:
                zeroes_mask = (motion_embeds == 0).all(dim=-1)
                if motion_embeds.shape[-1] != self.latent_dim:
                    motion_embeds_proj = self.pose_proj_in_source(motion_embeds)
                    motion_embeds_proj[zeroes_mask] = 0
                else:
                    motion_embeds_proj = motion_embeds
 
        else:
            raise TypeError(f"condition type {self.condition} not supported")
        # 4. transformer
        # if self.diffusion_only:
        proj_noised_motion = self.pose_proj_in_target(noised_motion)
        
        # BUILD the mask now
        if motion_embeds is None:
            aug_mask = torch.cat((condition_mask[:, :text_emb_latent.shape[0]],
                                  motion_in_mask), 1)
        else:
            sep_token_mask = torch.ones((bs, self.sep_token.shape[0]),
                                        dtype=bool,
                                        device=noised_motion.device)
            aug_mask = torch.cat((
                            condition_mask[:, text_emb_latent.shape[0]:],
                            sep_token_mask,
                            motion_in_mask,
                            ), 1)
            # B, T_output

        # NOTE: condition encoding
        # emb_latent: T_Text, B, D
        # motion_embeds_proj: T_max, B, D
        motion_embeds_proj_ori = motion_embeds_proj
        cond_seq = torch.cat((emb_latent, motion_embeds_proj), dim=0)
        cond_seq = self.cond_pos(cond_seq)
        cond_seq_processed = self.cond_encoder(cond_seq, src_key_padding_mask=~condition_mask)
        T_Text = emb_latent.size(0)
        T_max = motion_embeds_proj.size(0)
        emb_latent_processed, motion_embeds_proj_processed = torch.split(cond_seq_processed, [T_Text, T_max], dim=0)
        emb_latent, motion_embeds_proj = emb_latent_processed, motion_embeds_proj_processed

        if motion_embeds is None:
            xseq = proj_noised_motion
        else:      
            sep_token_batch = torch.tile(self.sep_token, (bs,)).reshape(bs,
                                                                        -1)
            if self.ablation in ['raw_source', 'raw_text_and_source']:
                xseq = torch.cat((motion_embeds_proj_ori,
                                sep_token_batch[None],
                                proj_noised_motion), axis=0)
            else:
                xseq = torch.cat((motion_embeds_proj,
                                sep_token_batch[None],
                                proj_noised_motion), axis=0)

        xseq = self.query_pos(xseq)
        if self.use_target_mask:
            mask = None
        else:
            mask = aug_mask
        if self.ablation == 'no_text':
            condition = torch.zeros_like(emb_latent[0])
        elif self.ablation == 'raw_text' or self.ablation == 'raw_text_and_source':
            condition = emb_latent_ori[0]
        elif self.ablation == 'raw_source':
            condition = emb_latent[0]

        if return_attention:
            tokens, target_repr, attention_mask = self.dit_encoder.forward_with_repr_att(xseq.permute(1, 0, 2), \
            timestep, condition, depth=target_align_depth, mask=mask)
        else:
            tokens, target_repr = self.dit_encoder.forward_with_repr(xseq.permute(1, 0, 2), \
            timestep, condition, depth=target_align_depth, mask=mask)
        # B, T, D
        tokens = tokens.permute(1, 0, 2)
        # if self.diffusion_only:
        if motion_embeds is not None:
            denoised_motion_proj = tokens
            if self.use_sep:
                useful_tokens = motion_embeds_proj.shape[0]+1
            else:
                useful_tokens = motion_embeds_proj.shape[0]
            denoised_motion_proj = denoised_motion_proj[useful_tokens:]
        else:
            denoised_motion_proj = tokens

        denoised_motion = self.pose_proj_out(denoised_motion_proj)
        denoised_motion[~motion_in_mask.T] = 0
        # zero for padded area
        # else:
        #     sample = tokens[:sample.shape[0]]
        # 5. [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
        denoised_motion = denoised_motion.permute(1, 0, 2)

        if self.use_kw_align:
            kw_src_feat = self.kw_src_proj(motion_embeds_proj).permute(1, 0, 2)  # B, T_src, R
            kw_tgt_feat = self.kw_tgt_proj(target_repr[:, T_max+1:])             # B, T_tgt, R
        gate_logits = None
        edit_ungated = None
        if self.use_identity_gate and motion_embeds is not None:
            gate_logits = self.gate_head(denoised_motion_proj).permute(1, 0, 2)  # B, T_tgt, K
            edit_ungated = denoised_motion  # v/x̂0 of the edit stream, pre-gate
            denoised_motion, _ = self._apply_gate(denoised_motion, gate_logits,
                                                  motion_embeds, src_len, tgt_len,
                                                  x_t=_x_t_gate, t=timestep,
                                                  edit_is_v=self.edit_is_v)
            denoised_motion = denoised_motion * motion_in_mask[..., None]
        elif self.source_anchored and motion_embeds is not None:
            denoised_motion = self._apply_source_anchor(
                denoised_motion, motion_embeds, src_len, tgt_len,
                x_t=_x_t_gate, t=timestep, edit_is_v=self.edit_is_v)
            denoised_motion = denoised_motion * motion_in_mask[..., None]
        anc_gate = anc_ident = anc_has_src = None
        if self.anchor_x0_gate and motion_embeds is not None:
            # keep denoised_motion as RAW v; expose gate + aligned source so the
            # x̂0-space anchor loss is computed in base_diffusion (stable, no 1/sm)
            _gl = self.x0_gate_head(denoised_motion_proj).permute(1, 0, 2)
            if self.anchor_field_dim == 'global':
                _gl = _gl.mean(dim=1, keepdim=True)
            anc_gate = self._anchor_gate_act(_gl)  # activated; broadcasts over (T,)D
            anc_ident, anc_has_src = self._identity_stream(
                motion_embeds, src_len, tgt_len, denoised_motion.shape[1])
        target_repr = self.target_head(target_repr[:, T_max+1:]) # B, T_target, D
        source_repr = self.source_head(motion_embeds_proj) # B, T_source, D
        if return_attention:
            return denoised_motion, source_repr.permute(1,0,2), target_repr, attention_mask
        # return ordering: (denoised, src_repr, tgt_repr [, gate_logits] [, kw_src, kw_tgt])
        extras = []
        if self.use_identity_gate:
            extras.append(gate_logits)
            if getattr(self, 'expose_edit_ungated', False):
                extras.append(edit_ungated)
        if self.use_kw_align:
            extras.extend([kw_src_feat, kw_tgt_feat])
        if self.anchor_x0_gate:
            extras.extend([anc_gate, anc_ident, anc_has_src])
        if extras:
            return (denoised_motion, source_repr.permute(1,0,2), target_repr, *extras)
        return denoised_motion, source_repr.permute(1,0,2), target_repr

    def forward_with_guidance(self,
                              noised_motion,
                              timestep,
                              in_motion_mask,
                              text_embeds,
                              condition_mask,
                              guidance_motion,
                              guidance_text_n_motion, 
                              motion_embeds=None,
                              lengths=None,
                              inpaint_dict=None,
                              max_steps=None,
                              prob_way='3way',
                              **kwargs):
        if max_steps is not None:
            curr_ts = timestep[0].item()
            g_m = max(1, guidance_motion*2*curr_ts/max_steps)
            guidance_motion = g_m
            g_t_tm = max(1, guidance_text_n_motion*2*curr_ts/max_steps)
            guidance_text_n_motion = g_t_tm

        if motion_embeds is None:
            half = noised_motion[: len(noised_motion) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward(combined, timestep,
                                    in_motion_mask=in_motion_mask,
                                    text_embeds=text_embeds,
                                    condition_mask=condition_mask, 
                                    motion_embeds=motion_embeds,
                                    lengths=lengths)
            uncond_eps, cond_eps_text = torch.split(model_out, len(model_out) // 2,
                                                     dim=0)
            # make it BxSxfeatures
            if inpaint_dict is not None:
                import torch.nn.functional as F
                source_mot = inpaint_dict['start_motion'].permute(1, 0, 2)
                if source_mot.shape[1] >= uncond_eps.shape[1]:
                    source_mot = source_mot[:, :uncond_eps.shape[1]]
                else:
                    pad = uncond_eps.shape[1] - source_mot.shape[1]
                    # Pad the tensor on the second dimension (time)
                    source_mot = F.pad(source_mot, (0, 0, 0, pad), 'constant', 0)

                mot_len = source_mot.shape[1]
                # concat mask for all the frames
                mask_src_parts = inpaint_dict['mask'].unsqueeze(1).repeat(1,
                                                                      mot_len,
                                                                      1)
                uncond_eps = uncond_eps*(mask_src_parts) + source_mot*(~mask_src_parts)
                cond_eps_text = cond_eps_text*(mask_src_parts) + source_mot*(~mask_src_parts)
            half_eps = uncond_eps + guidance_text_n_motion * (cond_eps_text - uncond_eps) 
            eps = torch.cat([half_eps, half_eps], dim=0)
        else:
            third = noised_motion[: len(noised_motion) // 3]
            combined = torch.cat([third, third, third], dim=0)
            model_out = self.forward(combined, timestep,
                                     in_motion_mask=in_motion_mask,
                                     text_embeds=text_embeds,
                                     condition_mask=condition_mask,
                                     motion_embeds=motion_embeds,
                                     lengths=lengths,
                                     src_len=kwargs.get('src_len'),
                                     tgt_len=kwargs.get('tgt_len'),
                                     defer_gate=True)
            gate_logits_g = None
            if isinstance(model_out, tuple):  # SG-DSD deferred gate
                model_out, gate_logits_g = model_out
            uncond_eps, cond_eps_motion, cond_eps_text_n_motion = torch.split(model_out,
                                                                            len(model_out) // 3,
                                                                            dim=0)
            if inpaint_dict is not None:
                import torch.nn.functional as F
                source_mot = inpaint_dict['start_motion'].permute(1, 0, 2)
                if source_mot.shape[1] >= uncond_eps.shape[1]:
                    source_mot = source_mot[:, :uncond_eps.shape[1]]
                else:
                    pad = uncond_eps.shape[1] - source_mot.shape[1]
                    # Pad the tensor on the second dimension (time)
                    source_mot = F.pad(source_mot, (0, 0, 0, pad), 'constant', 0)

                mot_len = source_mot.shape[1]
                # concat mask for all the frames
                mask_src_parts = inpaint_dict['mask'].unsqueeze(1).repeat(1,
                                                                      mot_len,
                                                                      1)
                uncond_eps = uncond_eps*(~mask_src_parts) + source_mot*mask_src_parts
                cond_eps_text = cond_eps_text*(~mask_src_parts) + source_mot*mask_src_parts
                cond_eps_text_n_motion = cond_eps_text_n_motion*(~mask_src_parts) + source_mot*mask_src_parts
            if prob_way=='3way':
                third_eps = uncond_eps + guidance_motion * (cond_eps_motion - uncond_eps) + \
                            guidance_text_n_motion * (cond_eps_text_n_motion - cond_eps_motion)
            if prob_way=='2way':
                third_eps = uncond_eps + guidance_text_n_motion * (cond_eps_text_n_motion - uncond_eps)

            if self.anchor_x0_gate and motion_embeds is not None:
                # STABLE source anchor: return raw CFG'd v + the extras the sampler
                # needs to nudge x̂0 toward source AFTER its v->x̂0 conversion.
                # No v-space blend here -> no 1/sm, no feedback.
                B0 = third_eps.shape[0]
                gate_c = gate_logits_g[-B0:]
                me_c = motion_embeds[:, -B0:]
                sl = kwargs.get('src_len'); tl = kwargs.get('tgt_len')
                if sl is not None and len(sl) == 3 * B0:
                    sl = sl[-B0:]; tl = tl[-B0:]
                ident, has_src = self._identity_stream(me_c, sl, tl, third_eps.shape[1])
                gate_c = self._anchor_gate_act(gate_c)   # activate (tanh/sigmoid)
                eps = torch.cat([third_eps, third_eps, third_eps], dim=0)
                extra = {
                    'anc_gate': torch.cat([gate_c, gate_c, gate_c], dim=0),
                    'anc_ident': torch.cat([ident, ident, ident], dim=0),
                    'anc_has_src': torch.cat([has_src, has_src, has_src], dim=0),
                }
                return eps, extra
            if self.use_identity_gate and gate_logits_g is not None:
                # blend ONCE, after CFG, using the conditional replica's gate and
                # the (shared) source identity stream — avoids extrapolating the
                # structurally-different uncond replica (alpha=1, no source)
                B0 = third_eps.shape[0]
                gate_c = gate_logits_g[-B0:]
                me_c = motion_embeds[:, -B0:] if motion_embeds is not None else None
                sl = kwargs.get('src_len'); tl = kwargs.get('tgt_len')
                if sl is not None and len(sl) == 3 * B0:
                    sl = sl[-B0:]; tl = tl[-B0:]
                t_c = timestep[:B0] if torch.is_tensor(timestep) else timestep
                third_eps, _ = self._apply_gate(third_eps, gate_c, me_c, sl, tl,
                                                x_t=third, t=t_c,
                                                edit_is_v=self.edit_is_v)
            elif self.source_anchored and motion_embeds is not None:
                # source anchor applied ONCE after CFG on the combined Δ
                B0 = third_eps.shape[0]
                me_c = motion_embeds[:, -B0:]
                sl = kwargs.get('src_len'); tl = kwargs.get('tgt_len')
                if sl is not None and len(sl) == 3 * B0:
                    sl = sl[-B0:]; tl = tl[-B0:]
                t_c = timestep[:B0] if torch.is_tensor(timestep) else timestep
                third_eps = self._apply_source_anchor(third_eps, me_c, sl, tl,
                                                      x_t=third, t=t_c,
                                                      edit_is_v=self.edit_is_v)

            eps = torch.cat([third_eps, third_eps, third_eps], dim=0)
        return eps

import torch

def test_TMED_DiTMotionDenoiser():
    # 设置随机种子以保证测试的可复现性
    torch.manual_seed(42)

    # 测试参数
    batch_size = 8
    seq_length = 100  # 序列长度
    nfeats = 263
    latent_dim = 256
    text_encoded_dim = 768

    # 创建测试输入数据
    noised_motion = torch.randn(batch_size, seq_length, nfeats)  # 输入的噪声运动数据 (N, T, nfeats)
    timestep = torch.randint(0, 1000, (batch_size,))  # 随机生成时间步 (N,)
    text_embeds = torch.randn(batch_size, 1, text_encoded_dim)  # 随机生成文本嵌入 (N, D)
    in_motion_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)  # 假设无填充
    condition_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)  # 条件掩码
    motion_embeds = torch.randn(seq_length, batch_size, nfeats)  # 随机生成运动嵌入 (N, T, latent_dim)

    # 创建模型实例
    model = DiT_Denoiser(
        nfeats=nfeats,
        latent_dim=latent_dim,
        text_encoded_dim=text_encoded_dim,
        num_layers=6,
        num_heads=4,
        dropout=0.1,
        activation='gelu',
        motion_condition = 'source'
    )

    # 将模型设置为评估模式
    model.eval()

    # 前向传播
    with torch.no_grad():
        output = model(noised_motion, timestep, in_motion_mask, text_embeds, condition_mask, motion_embeds=motion_embeds)

    # 检查输出的形状
    expected_shape = (batch_size, seq_length, nfeats)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"

    # 检查输出是否包含合理值 (例如无NaN)
    assert not torch.isnan(output).any(), "Output contains NaN values, which is unexpected."

    print("Test passed: TMED_DiTMotionDenoiser handles 1D inputs correctly!")

# 运行测试
if __name__ == "__main__":
    test_TMED_DiTMotionDenoiser()
