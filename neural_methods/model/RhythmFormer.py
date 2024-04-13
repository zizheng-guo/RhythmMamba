""" 
RhythmFormer:Extracting rPPG Signals Based on Hierarchical Temporal Periodic Transformer
"""
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import Tuple, Union
from timm.models.layers import trunc_normal_
from neural_methods.model.base.video_bra import video_BiFormerBlock

class Fusion_Stem(nn.Module):
    def __init__(self,apha=0.5,belta=0.5):
        super(Fusion_Stem, self).__init__()

        self.stem11 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )
        
        self.stem12 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )

        self.stem21 =nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stem22 =nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.apha = apha
        self.belta = belta

    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [N,D,C,H,W]
        Returns:
          fusion_x [N*D,C,H/4,W/4]
        """
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:,:1,:,:,:],x[:,:1,:,:,:],x[:,:D-2,:,:,:]],1)
        x2 = torch.cat([x[:,:1,:,:,:],x[:,:D-1,:,:,:]],1)
        x3 = x
        x4 = torch.cat([x[:,1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x5 = torch.cat([x[:,2:,:,:,:],x[:,D-1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x_diff = self.stem12(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],2).view(N * D, 12, H, W))
        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)

        #fusion layer1
        x_path1 = self.apha*x + self.belta*x_diff
        x_path1 = self.stem21(x_path1)
        #fusion layer2
        x_path2 = self.stem22(x_diff)
        x = self.apha*x_path1 + self.belta*x_path2
        
        return x
    
class TPT_Block(nn.Module):
    def __init__(self, dim, depth, num_heads, t_patch, topk,
                 mlp_ratio=4., drop_path=0., side_dwconv=5):
        super().__init__()
        self.dim = dim
        self.depth = depth
        ############ downsample layers & upsample layers #####################
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.layer_n = int(math.log(t_patch,2))
        for i in range(self.layer_n):
            downsample_layer = nn.Sequential(
                nn.BatchNorm3d(dim), 
                nn.Conv3d(dim , dim , kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                )
            self.downsample_layers.append(downsample_layer)
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=(2, 1, 1)),
                nn.Conv3d(dim , dim , [3, 1, 1], stride=1, padding=(1, 0, 0)),   
                nn.BatchNorm3d(dim),
                nn.ELU(),
                )
            self.upsample_layers.append(upsample_layer)
        ######################################################################
        self.blocks = nn.ModuleList([
            video_BiFormerBlock(
                    dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    num_heads=num_heads,
                    t_patch=t_patch,
                    topk=topk,
                    mlp_ratio=mlp_ratio,
                    side_dwconv=side_dwconv,
                )
            for i in range(depth)
        ])
    def forward(self, x:torch.Tensor):
        """Definition of TPT_Block.
        Args:
          x [N,C,D,H,W]
        Returns:
          x [N,C,D,H,W]
        """
        for i in range(self.layer_n) :
            x = self.downsample_layers[i](x)
        for blk in self.blocks:
            x = blk(x)
        for i in range(self.layer_n) :
            x = self.upsample_layers[i](x)

        return x
    
class RhythmFormer(nn.Module):

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        dim: int = 64, frame: int = 160,
        image_size: Optional[int] = (160,128,128),
        in_chans=64, head_dim=16,
        stage_n = 3,
        embed_dim=[64, 64, 64], mlp_ratios=[1.5, 1.5, 1.5],
        depth=[2, 2, 2], 
        t_patchs:Union[int, Tuple[int]]=(2, 4, 8),
        topks:Union[int, Tuple[int]]=(40, 40, 40),
        side_dwconv:int=3,
        drop_path_rate=0.,
        use_checkpoint_stages=[],
    ):
        super().__init__()

        self.image_size = image_size  
        self.frame = frame  
        self.dim = dim              
        self.stage_n = stage_n

        self.Fusion_Stem = Fusion_Stem()
        self.patch_embedding = nn.Conv3d(in_chans,embed_dim[0], kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.ConvBlockLast = nn.Conv1d(embed_dim[-1], 1, kernel_size=1,stride=1, padding=0)

        ##########################################################################
        self.stages = nn.ModuleList()
        nheads= [dim // head_dim for dim in embed_dim]
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        for i in range(stage_n):
            stage = TPT_Block(dim=embed_dim[i],
                               depth=depth[i],
                               num_heads=nheads[i], 
                               mlp_ratio=mlp_ratios[i],
                               drop_path=dp_rates[sum(depth[:i]):sum(depth[:i+1])],
                               t_patch=t_patchs[i], topk=topks[i], side_dwconv=side_dwconv
                               )
            self.stages.append(stage)
        ##########################################################################

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        N, D, C, H, W = x.shape
        x = self.Fusion_Stem(x)    #[N*D 64 H/4 W/4]
        x = x.view(N,D,64,H//4,W//4).permute(0,2,1,3,4)
        x = self.patch_embedding(x)    #[N 64 D 8 8]
        for i in range(3):
            x = self.stages[i](x)    #[N 64 D 8 8]
        features_last = torch.mean(x,3)    #[N, 64, D, 8]  
        features_last = torch.mean(features_last,3)    #[N, 64, D]  
        rPPG = self.ConvBlockLast(features_last)    #[N, 1, D]
        rPPG = rPPG.squeeze(1)
        return rPPG 