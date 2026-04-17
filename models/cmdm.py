import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.base import Model
from models.modules import PositionalEncoding, TimestepEmbedder
from models.modules import SceneMapEncoderDecoder, SceneMapEncoder
from models.functions import load_and_freeze_clip_model, encode_text_clip, \
    load_and_freeze_bert_model, encode_text_bert, get_lang_feat_dim_type
from utils.misc import compute_repr_dimesion, smplx_neutral_model, get_meshes_from_smplx
from utils.joints_to_smplx import JointsToSMPLX
from utils.geometry_loss import motion_repr_to_joints, sample_valid_frame_indices, \
    batched_index_select, sample_scene_points, nearest_point_distances, smplx_signed_distance

@Model.register()
class CMDM(nn.Module):

    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        
        self.motion_type = cfg.data_repr
        self.motion_dim = cfg.input_feats
        self.latent_dim = cfg.latent_dim
        self.mask_motion = cfg.mask_motion
        
        self.arch = cfg.arch
        self.num_body_joints = 22

        ## time embedding
        self.time_emb_dim = cfg.time_emb_dim
        self.timestep_embedder = TimestepEmbedder(self.latent_dim, self.time_emb_dim, max_len=1000)

        ## contact
        self.contact_type = cfg.contact_model.contact_type
        self.contact_dim = compute_repr_dimesion(self.contact_type)
        self.temporal_affordance = bool(cfg.contact_model.get('temporal_affordance', False))
        self.num_contact_phases = int(cfg.contact_model.get('num_phases', 1))
        self.planes = cfg.contact_model.planes
        if self.arch == 'trans_enc':
            SceneMapModule = SceneMapEncoder
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
            if self.temporal_affordance:
                self.contact_phase_embedding = nn.Embedding(self.num_contact_phases, self.planes[-1])
        elif self.arch == 'trans_dec':
            SceneMapModule = SceneMapEncoderDecoder
        else:
            raise NotImplementedError
        self.contact_encoder = SceneMapModule(
            point_feat_dim=self.contact_dim,
            planes=self.planes,
            blocks=cfg.contact_model.blocks,
            num_points=cfg.contact_model.num_points,
        )
        
        ## text
        self.text_model_name = cfg.text_model.version
        self.text_max_length = cfg.text_model.max_length
        self.text_feat_dim, self.text_feat_type = get_lang_feat_dim_type(self.text_model_name)
        if self.text_feat_type == 'clip':
            self.text_model = load_and_freeze_clip_model(self.text_model_name)
        elif self.text_feat_type == 'bert':
            self.tokenizer, self.text_model = load_and_freeze_bert_model(self.text_model_name)
        else:
            raise NotImplementedError
        self.language_adapter = nn.Linear(self.text_feat_dim, self.latent_dim, bias=True)

        ## model architecture
        self.motion_adapter = nn.Linear(self.motion_dim, self.latent_dim, bias=True)
        self.positional_encoder = PositionalEncoding(self.latent_dim, dropout=0.1, max_len=5000)

        self.num_layers = cfg.num_layers
        if self.arch == 'trans_enc':
            self.self_attn_layer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=cfg.num_heads,
                    dim_feedforward=cfg.dim_feedforward,
                    dropout=cfg.dropout,
                    activation='gelu',
                    batch_first=True,
                ),
                enable_nested_tensor=False,
                num_layers=sum(cfg.num_layers),
            )
        elif self.arch == 'trans_dec':
            self.self_attn_layers = nn.ModuleList()
            self.kv_mappling_layers = nn.ModuleList()
            self.cross_attn_layers = nn.ModuleList()
            for i, n in enumerate(self.num_layers):
                self.self_attn_layers.append(
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model=self.latent_dim,
                            nhead=cfg.num_heads,
                            dim_feedforward=cfg.dim_feedforward,
                            dropout=cfg.dropout,
                            activation='gelu',
                            batch_first=True,
                        ),
                        num_layers=n,
                    )
                )

                if i != len(self.num_layers) - 1:
                    self.kv_mappling_layers.append(
                        nn.Sequential(
                            nn.Linear(self.planes[-1-i], self.latent_dim, bias=True),
                            nn.LayerNorm(self.latent_dim),
                        )
                    )
                    self.cross_attn_layers.append(
                        nn.TransformerDecoderLayer(
                            d_model=self.latent_dim,
                            nhead=cfg.num_heads,
                            dim_feedforward=cfg.dim_feedforward,
                            dropout=cfg.dropout,
                            activation='gelu',
                            batch_first=True,
                        )
                    )
        else:
            raise NotImplementedError
        self.motion_layer = nn.Linear(self.latent_dim, self.motion_dim, bias=True)

        ## motion normalization for geometry-aware losses
        self.register_buffer(
            'motion_mean',
            torch.zeros(1, 1, self.motion_dim),
            persistent=False,
        )
        self.register_buffer(
            'motion_std',
            torch.ones(1, 1, self.motion_dim),
            persistent=False,
        )
        self.has_motion_stats = False

        ## geometry-aware training losses
        geometry_loss_cfg = cfg.get('geometry_loss', {})
        self.use_geometry_loss = bool(geometry_loss_cfg.get('enable', False))
        if self.use_geometry_loss:
            self.lambda_pen = float(geometry_loss_cfg.lambda_pen)
            self.lambda_contact = float(geometry_loss_cfg.lambda_contact)
            self.pen_num_frames = int(geometry_loss_cfg.pen_num_frames)
            self.pen_num_scene_points = int(geometry_loss_cfg.pen_num_scene_points)
            self.contact_num_frames = int(geometry_loss_cfg.contact_num_frames)
            self.contact_threshold = float(geometry_loss_cfg.contact_threshold)
            self.contact_sigma = max(float(geometry_loss_cfg.contact_sigma), 1e-6)
            self.contact_joint_indices = list(geometry_loss_cfg.contact_joints)

            self.joints_to_smplx_model = JointsToSMPLX()
            self.joints_to_smplx_model.load_and_freeze(geometry_loss_cfg.joints_to_smplx_model_weights)
            self.joints_to_smplx_model.eval()

            self.smplx_body_model = smplx_neutral_model
            self.smplx_body_model.eval()
            for param in self.smplx_body_model.parameters():
                param.requires_grad = False

    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Register dataset normalization statistics for geometry-aware losses."""
        mean = mean.float().reshape(1, 1, -1)
        std = std.float().reshape(1, 1, -1)

        if mean.shape[-1] != self.motion_dim or std.shape[-1] != self.motion_dim:
            raise ValueError(
                f'Expected motion stats dim {self.motion_dim}, but got {mean.shape[-1]} and {std.shape[-1]}.'
            )

        self.motion_mean.copy_(mean)
        self.motion_std.copy_(std)
        self.has_motion_stats = True

    def denormalize_motion(self, motion: torch.Tensor) -> torch.Tensor:
        if not self.has_motion_stats:
            raise RuntimeError(
                'Motion normalization stats have not been set. '
                'Please pass dataset mean/std to CMDM before training.'
            )
        return motion * self.motion_std.to(motion.device) + self.motion_mean.to(motion.device)

    def _sample_motion_frames(self, joints: torch.Tensor, x_mask: torch.Tensor, num_frames: int):
        frame_indices, frame_valid = sample_valid_frame_indices(x_mask, num_frames)
        sampled_joints = batched_index_select(joints, frame_indices)
        return sampled_joints, frame_indices, frame_valid

    def _compute_contact_consistency_loss(
        self,
        pred_joints: torch.Tensor,
        gt_joints: torch.Tensor,
        scene_xyz: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_joints, frame_indices, frame_valid = self._sample_motion_frames(
            pred_joints, x_mask, self.contact_num_frames
        )
        gt_joints = batched_index_select(gt_joints, frame_indices)

        pred_contact_joints = pred_joints[:, :, self.contact_joint_indices, :]
        gt_contact_joints = gt_joints[:, :, self.contact_joint_indices, :]

        batch_size, num_frames, num_joints = pred_contact_joints.shape[:3]
        scene_xyz = scene_xyz.unsqueeze(1).expand(-1, num_frames, -1, -1)
        scene_xyz = scene_xyz.reshape(batch_size * num_frames, scene_xyz.shape[-2], 3)

        pred_contact_joints = pred_contact_joints.reshape(batch_size * num_frames, num_joints, 3)
        gt_contact_joints = gt_contact_joints.reshape(batch_size * num_frames, num_joints, 3)

        pred_dist = nearest_point_distances(pred_contact_joints, scene_xyz).reshape(batch_size, num_frames, num_joints)
        gt_dist = nearest_point_distances(gt_contact_joints, scene_xyz).reshape(batch_size, num_frames, num_joints)

        contact_weight = torch.exp(-0.5 * (gt_dist / self.contact_sigma) ** 2)
        if self.contact_threshold > 0:
            contact_weight = contact_weight * (gt_dist < self.contact_threshold).float()
        contact_weight = contact_weight * frame_valid.unsqueeze(-1).float()

        weighted_dist = (contact_weight * pred_dist).sum(dim=(1, 2))
        norm = contact_weight.sum(dim=(1, 2))
        loss = weighted_dist / norm.clamp_min(1e-6)
        return torch.where(norm > 0, loss, torch.zeros_like(loss))

    def _compute_penetration_loss(
        self,
        pred_joints: torch.Tensor,
        scene_xyz: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        sampled_joints, _, frame_valid = self._sample_motion_frames(pred_joints, x_mask, self.pen_num_frames)
        batch_size, num_frames = sampled_joints.shape[:2]

        sampled_scene_xyz = sample_scene_points(scene_xyz, self.pen_num_scene_points)
        sampled_joint_seq = sampled_joints.reshape(batch_size, num_frames, -1)
        sampled_joint_mask = ~frame_valid

        pred_params = self.joints_to_smplx_model(sampled_joint_seq, sampled_joint_mask)
        pred_verts, body_faces = get_meshes_from_smplx(self.smplx_body_model, pred_params)

        sampled_scene_xyz = sampled_scene_xyz.unsqueeze(1).expand(-1, num_frames, -1, -1)
        sampled_scene_xyz = sampled_scene_xyz.reshape(batch_size * num_frames, sampled_scene_xyz.shape[-2], 3)
        pred_verts = pred_verts.reshape(batch_size * num_frames, pred_verts.shape[-2], 3)

        scene_to_human_sdf, _ = smplx_signed_distance(sampled_scene_xyz, pred_verts, body_faces)
        penetration = torch.relu(scene_to_human_sdf).mean(dim=-1).reshape(batch_size, num_frames)

        frame_weight = frame_valid.float()
        return (penetration * frame_weight).sum(dim=1) / frame_weight.sum(dim=1).clamp_min(1.0)

    def compute_aux_losses(self, pred_xstart, x_start, model_kwargs):
        """Compute geometry-aware auxiliary losses for Idea 5.

        Args:
            pred_xstart: predicted denoised motion, [B, L, D]
            x_start: GT motion, [B, L, D]
            model_kwargs: diffusion model kwargs containing x_mask and c_pc_xyz
        """
        if not self.use_geometry_loss:
            return {}

        if 'x_mask' not in model_kwargs or 'c_pc_xyz' not in model_kwargs:
            raise KeyError('Geometry-aware loss requires `x_mask` and `c_pc_xyz` in model_kwargs.')

        pred_motion = self.denormalize_motion(pred_xstart)
        gt_motion = self.denormalize_motion(x_start)
        pred_joints = motion_repr_to_joints(pred_motion, self.motion_type, self.num_body_joints)
        gt_joints = motion_repr_to_joints(gt_motion, self.motion_type, self.num_body_joints)

        x_mask = model_kwargs['x_mask']
        scene_xyz = model_kwargs['c_pc_xyz']
        if x_mask.ndim != 2:
            raise ValueError(f'Expected x_mask with shape [B, L], but got {x_mask.shape}.')
        if scene_xyz.ndim != 3 or scene_xyz.shape[-1] != 3:
            raise ValueError(
                f'Expected c_pc_xyz with shape [B, N, 3], but got {scene_xyz.shape}.'
            )
        if pred_joints.shape[:2] != x_mask.shape:
            raise ValueError(
                f'Predicted joints shape {pred_joints.shape[:2]} does not match x_mask shape {x_mask.shape}.'
            )

        contact_loss = self._compute_contact_consistency_loss(pred_joints, gt_joints, scene_xyz, x_mask)
        penetration_loss = self._compute_penetration_loss(pred_joints, scene_xyz, x_mask)
        aux_loss = self.lambda_contact * contact_loss + self.lambda_pen * penetration_loss

        return {
            'contact_loss': contact_loss,
            'penetration_loss': penetration_loss,
            'aux_loss': aux_loss,
        }

    def _encode_contact_condition(self, contact_xyz: torch.Tensor, contact_map: torch.Tensor) -> torch.Tensor:
        """Encode static or temporal affordance conditions.

        Static contact keeps the original shape ``[B, N, J]``.
        Temporal contact uses ``[B, K, N, J]`` and is fused by:
        1. shared per-phase contact encoding
        2. adding a learned phase embedding
        3. flattening all phase tokens into one conditioning sequence
        """
        if contact_map.ndim == 3:
            if contact_map.shape[-1] != self.contact_dim:
                raise ValueError(
                    f'Expected contact dim {self.contact_dim}, but got {contact_map.shape[-1]}.'
                )
            return self.contact_encoder(contact_xyz, contact_map)

        if contact_map.ndim != 4:
            raise ValueError(
                'CMDM expects contact condition with shape [B, N, J] or [B, K, N, J], '
                f'but got {contact_map.shape}.'
            )

        if not self.temporal_affordance:
            raise ValueError(
                'Received temporal affordance input, but temporal conditioning is disabled in CMDM config.'
            )
        if self.arch != 'trans_enc':
            raise NotImplementedError('Temporal affordance MVP currently supports CMDM trans_enc only.')

        batch_size, num_phases, num_points, contact_dim = contact_map.shape
        if num_phases != self.num_contact_phases:
            raise ValueError(
                f'Expected {self.num_contact_phases} affordance phases, but got {num_phases}.'
            )
        if contact_dim != self.contact_dim:
            raise ValueError(
                f'Expected per-phase contact dim {self.contact_dim}, but got {contact_dim}.'
            )

        repeated_xyz = contact_xyz.unsqueeze(1).expand(-1, num_phases, -1, -1)
        repeated_xyz = repeated_xyz.reshape(batch_size * num_phases, num_points, contact_xyz.shape[-1])
        contact_map = contact_map.reshape(batch_size * num_phases, num_points, contact_dim)

        cont_emb = self.contact_encoder(repeated_xyz, contact_map) # [B * K, num_groups, planes[-1]]
        num_groups, feat_dim = cont_emb.shape[1], cont_emb.shape[2]
        cont_emb = cont_emb.reshape(batch_size, num_phases, num_groups, feat_dim)

        phase_ids = torch.arange(num_phases, device=contact_map.device)
        phase_emb = self.contact_phase_embedding(phase_ids).view(1, num_phases, 1, feat_dim)
        cont_emb = cont_emb + phase_emb

        return cont_emb.reshape(batch_size, num_phases * num_groups, feat_dim)

    def forward(self, x, timesteps, **kwargs):
        """ Forward pass of the model.

        Args:
            x: input motion, [bs, seq_len, motion_dim]
            kwargs: other inputs, e.g., contact, text
        
        Return:
            Output motion, [bs, seq_len, motion_dim]
        """
        ## time embedding
        time_emb = self.timestep_embedder(timesteps) # [bs, 1, latent_dim]
        time_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=self.device)

        ## text embedding
        if self.text_feat_type == 'clip':
            text_emb = encode_text_clip(self.text_model, kwargs['c_text'], max_length=self.text_max_length, device=self.device)
            text_emb = text_emb.unsqueeze(1).detach().float()
            text_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=self.device)
        elif self.text_feat_type == 'bert':
            text_emb, text_mask = encode_text_bert(self.tokenizer, self.text_model, kwargs['c_text'], max_length=self.text_max_length, device=self.device)
            text_mask = ~(text_mask.to(torch.bool)) # 0 for valid, 1 for invalid
        else:
            raise NotImplementedError
        if 'c_text_mask' in kwargs:
            text_mask = torch.logical_or(text_mask, kwargs['c_text_mask'].repeat(1, text_mask.shape[1]))
        if 'c_text_erase' in kwargs:
            text_emb = text_emb * (1. - kwargs['c_text_erase'].unsqueeze(-1).float())
        text_emb = self.language_adapter(text_emb) # [bs, 1, latent_dim]

        ## encode contact
        cont_emb = self._encode_contact_condition(kwargs['c_pc_xyz'], kwargs['c_pc_contact'])
        if hasattr(self, 'contact_adapter'): # trans_enc
            cont_mask = torch.zeros((x.shape[0], cont_emb.shape[1]), dtype=torch.bool, device=self.device)
            if 'c_pc_mask' in kwargs:
                cont_mask = torch.logical_or(cont_mask, kwargs['c_pc_mask'].repeat(1, cont_mask.shape[1]))
            if 'c_pc_erase' in kwargs:
                cont_emb = cont_emb * (1. - kwargs['c_pc_erase'].unsqueeze(-1).float())
            cont_emb = self.contact_adapter(cont_emb) # [bs, num_groups, latent_dim], for trans_enc

        ## motion embedding
        x = self.motion_adapter(x) # [bs, seq_len, latent_dim]
        if self.arch == 'trans_enc':
            x = torch.cat([time_emb, text_emb, cont_emb, x], dim=1) # [bs, 2 + num_groups + seq_len, latent_dim]
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            x_mask = None
            if self.mask_motion:
                x_mask = torch.cat([time_mask, text_mask, cont_mask, kwargs['x_mask']], dim=1) # [bs, 2 + num_groups + seq_len]
            x = self.self_attn_layer(x, src_key_padding_mask=x_mask)

            non_motion_token = time_mask.shape[1] + text_mask.shape[1] + cont_mask.shape[1]
            x = x[:, non_motion_token:, :]
        elif self.arch == 'trans_dec':
            x = torch.cat([time_emb, text_emb, x], dim=1) # [bs, 2 + seq_len, latent_dim]
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            x_mask = None
            if self.mask_motion:
                x_mask = torch.cat([time_mask, text_mask, kwargs['x_mask']], dim=1) # [bs, 2 + seq_len]
            for i in range(len(self.num_layers)):
                x = self.self_attn_layers[i](x, src_key_padding_mask=x_mask) # self attention
                if i != len(self.num_layers) - 1: # cross attention
                    mem = cont_emb[i]
                    mem_mask = torch.zeros((x.shape[0], mem.shape[1]), dtype=torch.bool, device=self.device)
                    if 'c_pc_mask' in kwargs:
                        mem_mask = torch.logical_or(mem_mask, kwargs['c_pc_mask'].repeat(1, mem_mask.shape[1]))
                    if 'c_pc_erase' in kwargs:
                        mem = mem * (1. - kwargs['c_pc_erase'].unsqueeze(-1).float())
                    mem = self.kv_mappling_layers[i](mem)
                    x = self.cross_attn_layers[i](x, mem, tgt_key_padding_mask=x_mask, memory_key_padding_mask=mem_mask)

            non_motion_token = time_mask.shape[1] + text_mask.shape[1]
            x = x[:, non_motion_token:, :]
        else:
            raise NotImplementedError

        x = self.motion_layer(x)
        return x
