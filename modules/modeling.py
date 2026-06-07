from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import json
import os

from modules.until_module import PreTrainedModel, AllGather, CrossEn, HardNegativeNCE
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from modules.co_attention_transformer_module import Co_attention_block

import math
from torch import Tensor
from functools import partial
import numpy as np
from typing import Tuple, Union, Optional

try:
    from mamba_ssm import Mamba
    from einops import rearrange
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    MUSE_AVAILABLE = True
except ImportError:
    MUSE_AVAILABLE = False

logger = logging.getLogger(__name__)
allgather = AllGather.apply

if MUSE_AVAILABLE:
    class Mamba_Out(nn.Module):
        def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            bimamba_type="none",
            if_devide_out=False,
            init_layer_scale=None,
        ):
            factory_kwargs = {"device": device, "dtype": dtype}
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.d_conv = d_conv
            self.expand = expand
            self.d_inner = int(self.expand * self.d_model)
            self.use_fast_path = use_fast_path
            self.layer_idx = layer_idx
            self.bimamba_type = bimamba_type
            self.if_devide_out = if_devide_out

            self.init_layer_scale = init_layer_scale
            if init_layer_scale is not None:
                self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.activation = "silu"
            self.act = nn.SiLU()
            self.conv1d_b = nn.Conv1d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    groups=self.d_inner,
                    padding=d_conv - 1,
                    **factory_kwargs,
                )
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        def forward(self, hidden_states, inference_params=None):
            """
            hidden_states: (B, L, D)
            Returns: same shape as hidden_states
            """
            batch, seqlen, dim = hidden_states.shape
            # We do matmul and transpose BLH -> HBL at the same time
            xz = rearrange(
                self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                "d (b l) -> b d l",
                l=seqlen,
            )
            if self.in_proj.bias is not None:
                xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
            
            if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
                x, z = xz.chunk(2, dim=1)
                out = self.conv1d(x)
                x, z = xz.flip([-1]).chunk(2, dim=1)
                out_b = self.conv1d_b(x)
                if not self.if_devide_out:
                    out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                
            if self.init_layer_scale is not None:
                out = out * self.gamma    
            return out

    class MultiheadAttention_flash(nn.MultiheadAttention):
        def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                    need_weights: bool = True, attn_mask: Optional[Tensor] = None):

            return flash_attn_func(
                    q=query, k=key, v=value, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), alibi_slopes=None, deterministic=False)

    class LayerNorm_conv(nn.LayerNorm):
        """Subclass torch's LayerNorm to handle fp16."""
        def __init__(self, normalized_shape):
            super().__init__(normalized_shape=normalized_shape)

        def forward(self, x: torch.Tensor):
            x = x.permute(0,2,3,1)
            orig_type = x.dtype
            ret = super().forward(x.type(torch.float32))# add ssf
            return ret.type(orig_type).permute(0,3,1,2)

    class Mamba_head(nn.Module):
        def __init__(self, embed_dim, layer_num=0.1):
            super().__init__()
            self.embed_dim = embed_dim
            self.mamba = Mamba(self.embed_dim, d_conv=4, bimamba_type='v2', use_fast_path=True, expand=1)
            # self.mamba_out = Mamba_Out(self.embed_dim, d_conv=1, bimamba_type='v2', use_fast_path=True, expand=1)
            # self.transformer = nn.MultiheadAttention(self.embed_dim, self.embed_dim // 64)
            # self.flash_attn = MultiheadAttention_flash(self.embed_dim, self.embed_dim // 64)
            self.layer_norm1 = nn.LayerNorm(self.embed_dim)
            # self.layer_norm1 = RMSNorm(hidden_size=self.embed_dim)

            self.proj_drop = nn.Dropout(layer_num)
            self.temporal_fc = nn.Linear(self.embed_dim, self.embed_dim)
            nn.init.constant_(self.temporal_fc.weight, 0.)
            nn.init.constant_(self.temporal_fc.bias, 0.)

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            causal_attention_mask=None,
        ):
            residual = hidden_states
            hidden_states = self.layer_norm1(hidden_states)
            # hidden_states = self.transformer((hidden_states, None))[0] [L,B,D]
            hidden_states = self.mamba(hidden_states)
            # hidden_states = self.mamba_out(hidden_states)
            # hidden_states = self.flash_attn(hidden_states, hidden_states, hidden_states, need_weights=False, attn_mask=None)
            res_temporal = self.proj_drop(hidden_states.contiguous())
            
            res_temporal = self.temporal_fc(res_temporal)
            hidden_states = residual + res_temporal
            outputs = hidden_states

            return outputs
else:
    Mamba_Out = None
    MultiheadAttention_flash = None
    LayerNorm_conv = None
    Mamba_head = None

class AVIKAPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(AVIKAPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        # Note: For video
        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        # -------------------------------------------------
        
        # ---------- Create for frame_caption ------------
        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if not contain_cross:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_caption_position = False
            for key in state_dict.keys():
                if key.find("caption_position_embeddings") > -1:
                    contain_caption_position = True
                    break
            if not contain_caption_position:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["caption_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerCaption.")] = val.clone()
                            continue
        # -------------------------------------------------

        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class AVIKA(AVIKAPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(AVIKA, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        

    # ------ Feature Extraction ----------
    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden = self.clip.encode_text(input_ids).float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden
    
    def get_sequence_words_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden, words_hidden = self.clip.encode_text(input_ids, return_hidden=True)
        sequence_hidden = sequence_hidden.float().view(bs_pair, -1, sequence_hidden.size(-1))
        words_hidden = words_hidden.float()

        return sequence_hidden, words_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        if self.sim_header == "MUSE":
            visual_cls, visual_hidden = self.clip.encode_image(video, return_hidden=True, video_frame=video_frame)
            visual_cls = visual_cls.float().view(bs_pair, -1, visual_cls.size(-1))  
            visual_hidden = visual_hidden.float().view(bs_pair, -1, visual_hidden.size(-1)) 
            return visual_cls, visual_hidden
        else:
            visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
            visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
            return visual_hidden
