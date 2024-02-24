from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import math

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from data.utils.types import FeatureMap, BackboneFeatures, LstmState, LstmStates
from models.layers.rnn import DWSConvLSTM2d
from models.layers.sast.maxvit import (
    PartitionAttentionCl,
    nhwC_2_nChw,
    get_downsample_layer_Cf2Cl,
    PartitionType)

from .base import BaseDetector
from torch.autograd.function import Function
# @torch.jit.script
# def non_zero_ratio(x: torch.Tensor) -> torch.Tensor:
#     if x.shape[0] > 1:
#         return torch.sum(x != 0, dim=(2, 3)) / torch.prod(torch.tensor(x.shape[1:]))
#     else:
#         return torch.count_nonzero(x) / x.numel()

# class get_non_zero_mask(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         x = torch.any(x.unsqueeze(-1) != 0, dim=-1)
#         return x.float()

@torch.jit.script
def get_non_zero_mask(x: torch.Tensor) -> torch.Tensor:
    x = torch.any(x.unsqueeze(-1) != 0, dim=-1)
    return x.float()

# class get_non_zero_mask(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.squeeze = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         x = self.squeeze(x)
#         x = self.sigmoid(x).squeeze(1)
#         return x

class RNNDetector(BaseDetector):
    def __init__(self, mdl_config: DictConfig):
        super().__init__()

        ###### Config ######
        in_channels = mdl_config.input_channels
        embed_dim = mdl_config.embed_dim
        dim_multiplier_per_stage = tuple(mdl_config.dim_multiplier)
        num_blocks_per_stage = tuple(mdl_config.num_blocks)
        T_max_chrono_init_per_stage = tuple(mdl_config.T_max_chrono_init)
        enable_masking = mdl_config.enable_masking

        num_stages = len(num_blocks_per_stage)
        assert num_stages == 4

        assert isinstance(embed_dim, int)
        assert num_stages == len(dim_multiplier_per_stage)
        assert num_stages == len(num_blocks_per_stage)
        assert num_stages == len(T_max_chrono_init_per_stage)

        ###### Compile if requested ######
        compile_cfg = mdl_config.get('compile', None)
        if compile_cfg is not None:
            compile_mdl = compile_cfg.enable
            if compile_mdl and th_compile is not None:
                compile_args = OmegaConf.to_container(compile_cfg.args, resolve=True, throw_on_missing=True)
                self.forward = th_compile(self.forward, **compile_args)
            elif compile_mdl:
                print('Could not compile backbone because torch.compile is not available')
        ##################################

        input_dim = in_channels
        patch_size = mdl_config.stem.patch_size
        stride = 1
        self.stage_dims = [embed_dim * x for x in dim_multiplier_per_stage]

        self.stages = nn.ModuleList()
        self.strides = []
        self.align_convs = nn.ModuleList()
        for stage_idx, (num_blocks, T_max_chrono_init_stage) in \
                enumerate(zip(num_blocks_per_stage, T_max_chrono_init_per_stage)):
            spatial_downsample_factor = patch_size if stage_idx == 0 else 2
            stage_dim = self.stage_dims[stage_idx]
            enable_masking_in_stage = enable_masking and stage_idx == 0
            stage = RNNDetectorStage(dim_in=input_dim,
                                     stage_dim=stage_dim,
                                     spatial_downsample_factor=spatial_downsample_factor,
                                     num_blocks=num_blocks,
                                     enable_token_masking=enable_masking_in_stage,
                                     T_max_chrono_init=T_max_chrono_init_stage,
                                     stage_cfg=mdl_config.stage)
            stride = stride * spatial_downsample_factor
            self.strides.append(stride)

            input_dim = stage_dim
            self.stages.append(stage)
            if stage_idx != num_stages - 1:
                self.align_convs.append(nn.Conv2d(stage_dim, 128, kernel_size=1, stride=1, padding=0))
            else:
                self.align_convs.append(None)

        self.num_stages = num_stages
        # self.get_non_zero_mask = get_non_zero_mask()

    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.stage_dims[stage_idx] for stage_idx in stage_indices)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        stage_indices = [x - 1 for x in stages]
        assert min(stage_indices) >= 0, stage_indices
        assert max(stage_indices) < len(self.stages), stage_indices
        return tuple(self.strides[stage_idx] for stage_idx in stage_indices)

    def forward(self, x: torch.Tensor, prev_states: Optional[LstmStates] = None, token_mask: Optional[torch.Tensor] = None) \
            -> Tuple[BackboneFeatures, LstmStates, torch.Tensor]:
        if prev_states is None:
            prev_states = [None] * self.num_stages
        assert len(prev_states) == self.num_stages
        states: LstmStates = list()
        output: Dict[int, FeatureMap] = {}

        # mask = get_non_zero_mask(x)  # Non-zero ratio p of event representation.
        mask = x
        # mask = self.upconv(non_zero_mask)
        P = 0
        for stage_idx, stage in enumerate(self.stages):
            x, state, mask, p_loss = stage(x, prev_states[stage_idx], token_mask if stage_idx == 0 else None, mask)
            states.append(state)
            stage_number = stage_idx + 1
            output[stage_number] = x
            P += ((stage_idx + 1)**2 * p_loss) / 16
            if self.align_convs[stage_idx] is not None:
                mask = mask + self.align_convs[stage_idx](x)
        return output, states, P

class MaxVitAttentionPairCl(nn.Module):
    def __init__(self,
                 dim: int,
                 skip_first_norm: bool,
                 attention_cfg: DictConfig):
        super().__init__()
        # self.att = PartitionAttentionCl(dim=dim,
        #                                 partition_type=PartitionType.GLOBAL,
        #                                 attention_cfg=attention_cfg,
        #                                 skip_first_norm=skip_first_norm)
        self.att_window = PartitionAttentionCl(dim=dim,
                                               partition_type=PartitionType.WINDOW,
                                               attention_cfg=attention_cfg,
                                               skip_first_norm=skip_first_norm)
        self.att_grid = PartitionAttentionCl(dim=dim,
                                             partition_type=PartitionType.GRID,
                                             attention_cfg=attention_cfg,
                                             skip_first_norm=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb1: torch.Tensor, pos_emb2: torch.Tensor) -> torch.Tensor:
        x, mask, p_loss1 = self.att_window(x, mask, pos_emb1, pos_emb2)
        x, mask, p_loss2 = self.att_grid(x, mask, pos_emb1, pos_emb2)
        # x, mask = self.att(x, mask, pos_emb)
        return x, mask, p_loss1 + p_loss2


# sinusoidal positional embeddings

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=1000, normalize=False, scale=None, input_size=(128, 128, 128)):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.pos_embedding = self.generate_position_embedding(input_size)

    def generate_position_embedding(self, input_size):
        B, H, W = input_size
        mask = ~torch.zeros([B, H, W], dtype=bool)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos

    def forward(self, x):
        B, H, W = x.shape[:3]
        self.pos_embedding = self.pos_embedding.to(x.device)
        pos = self.pos_embedding[:, :H, :W, :].repeat(B, 1, 1, 1)
        return pos
    
class MyIdentityFunc(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = 10 * grad_output.clone() 
        return grad_input
    
class RNNDetectorStage(nn.Module):
    """Operates with NCHW [channel-first] format as input and output.
    """

    def __init__(self,
                 dim_in: int,
                 stage_dim: int,
                 spatial_downsample_factor: int,
                 num_blocks: int,
                 enable_token_masking: bool,
                 T_max_chrono_init: Optional[int],
                 stage_cfg: DictConfig):
        super().__init__()
        assert isinstance(num_blocks, int) and num_blocks > 0
        downsample_cfg = stage_cfg.downsample
        lstm_cfg = stage_cfg.lstm
        attention_cfg = stage_cfg.attention

        self.downsample_cf2cl = get_downsample_layer_Cf2Cl(dim_in=dim_in,
                                                           dim_out=stage_dim,
                                                           downsample_factor=spatial_downsample_factor,
                                                           downsample_cfg=downsample_cfg)
        mask_dim_in = 20 if dim_in==20 else 128
        self.downsample_mask = nn.Sequential(
                                get_downsample_layer_Cf2Cl(dim_in=mask_dim_in,
                                                           dim_out=128,
                                                           downsample_factor=spatial_downsample_factor,
                                                           downsample_cfg=downsample_cfg),
                                nn.LeakyReLU())
        blocks = [MaxVitAttentionPairCl(dim=stage_dim,
                                        skip_first_norm=i == 0 and self.downsample_cf2cl.output_is_normed(),
                                        attention_cfg=attention_cfg) for i in range(num_blocks)]
        self.att_blocks = nn.ModuleList(blocks)
        self.lstm = DWSConvLSTM2d(dim=stage_dim,
                                  dws_conv=lstm_cfg.dws_conv,
                                  dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                  dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                  cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))
        initial_size = (384, 640)
        overload_size = (1, initial_size[0] // spatial_downsample_factor, initial_size[1] // spatial_downsample_factor)
        self.pos_emb1 = PositionEmbeddingSine(stage_dim // 2, normalize=True, input_size=overload_size)
        self.pos_emb2 = PositionEmbeddingSine(128 // 2, normalize=True, input_size=overload_size)

        ###### Mask Token ################
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, stage_dim),
                                       requires_grad=True) if enable_token_masking else None
        if self.mask_token is not None:
            torch.nn.init.normal_(self.mask_token, std=.02)
        # self.weights = nn.Parameter(torch.ones(2), requires_grad=True)
        ##################################

    def forward(self, x: torch.Tensor,
                h_and_c_previous: Optional[LstmState] = None,
                token_mask: Optional[torch.Tensor] = None,
                mask: torch.Tensor = None) \
            -> Tuple[FeatureMap, LstmState, torch.Tensor]:
        mask = self.downsample_mask(mask)
        x = self.downsample_cf2cl(x)  # N C H W -> N H W C
        if token_mask is not None:
            assert self.mask_token is not None, 'No mask token present in this stage'
            x[token_mask] = self.mask_token

        P = 0
        for blk in self.att_blocks:
            x, mask, p_loss = blk(x, mask, self.pos_emb1, self.pos_emb2)
            P += p_loss
            
        x = nhwC_2_nChw(x)  # N H W C -> N C H W
        mask = nhwC_2_nChw(mask)
        h_c_tuple = self.lstm(x, h_and_c_previous)
        x = h_c_tuple[0]
        return x, h_c_tuple, mask, P