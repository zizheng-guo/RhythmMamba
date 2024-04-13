"""
Adapted from here: https://github.com/rayleizhu/BiFormer
"""
from typing import List, Optional
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor
import math
from timm.models.layers import DropPath

from neural_methods.model.base.rrsda import video_regional_routing_attention_torch

class CDC_T(nn.Module):
    """
    The CDC_T Module is from here: https://github.com/ZitongYu/PhysFormer/model/transformer_layer.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal
      
class video_BRA(nn.Module):

    def __init__(self, dim, num_heads=8, t_patch=8, qk_scale=None, topk=4,  side_dwconv=3, auto_pad=False, attn_backend='torch'):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5 
        self.topk = topk
        self.t_patch = t_patch  # frame of patch
        ################side_dwconv (i.e. LCE in Shunted Transformer)###########
        self.lepe = nn.Conv3d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
        ##########################################
        self.qkv_linear = nn.Conv3d(self.dim, 3*self.dim, kernel_size=1)
        self.output_linear = nn.Conv3d(self.dim, self.dim, kernel_size=1)
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=0.2),  
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=0.2),  
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )
        if attn_backend == 'torch':
            self.attn_fn = video_regional_routing_attention_torch
        else:
            raise ValueError('CUDA implementation is not available yet. Please stay tuned.')

    def forward(self, x:Tensor):

        N, C, T, H, W = x.size()
        t_region = max(4 // self.t_patch , 1)
        region_size = (t_region, H//4 , W//4)

        # STEP 1: linear projection
        q , k , v = self.proj_q(x) , self.proj_k(x) ,self.proj_v(x)

        # STEP 2: pre attention
        q_r = F.avg_pool3d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool3d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False) # ncthw
        q_r:Tensor = q_r.permute(0, 2, 3, 4, 1).flatten(1, 3) # n(thw)c
        k_r:Tensor = k_r.flatten(2, 4) # nc(thw)
        a_r = q_r @ k_r # n(thw)(thw)
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1) # n(thw)k
        idx_r:LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1) 

        # STEP 3: refined attention
        output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
                                        region_graph=idx_r, region_size=region_size)
        
        output = output + self.lepe(v) # nctHW
        output = self.output_linear(output) # nctHW

        return output

class video_BiFormerBlock(nn.Module):
    def __init__(self, dim, drop_path=0., num_heads=4, t_patch=1,qk_scale=None, topk=4, mlp_ratio=2, side_dwconv=5):
        super().__init__()
        self.t_patch = t_patch
        self.norm1 = nn.BatchNorm3d(dim)
        self.attn = video_BRA(dim=dim, num_heads=num_heads, t_patch=t_patch,qk_scale=qk_scale, topk=topk, side_dwconv=side_dwconv)
        self.norm2 = nn.BatchNorm3d(dim)
        self.mlp = nn.Sequential(nn.Conv3d(dim, int(mlp_ratio*dim), kernel_size=1),
                                 nn.BatchNorm3d(int(mlp_ratio*dim)),
                                 nn.GELU(),
                                 nn.Conv3d(int(mlp_ratio*dim),  int(mlp_ratio*dim), 3, stride=1, padding=1),  
                                 nn.BatchNorm3d(int(mlp_ratio*dim)),
                                 nn.GELU(),
                                 nn.Conv3d(int(mlp_ratio*dim), dim, kernel_size=1),
                                 nn.BatchNorm3d(dim),
                                )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x