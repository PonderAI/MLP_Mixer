import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange

"""
B = Batches
C = channels
H = Height
W = Width
Nh = Height/Patch size
Nw = Width/Patch size
P = Patches
"""

class PatchEmbedding(nn.Module):

    def __init__(self, img_channels, embd_channels, patch_size):
        super().__init__()
        self.patch_embd = nn.Sequential(
            nn.Conv2d(img_channels, embd_channels, patch_size, patch_size), # [B, C, Nh, Nw] -> [B, 512, 64, 64]
            Rearrange('b c h w -> b h w c'), # [B, Nh, Nw, C] -> [N, 64, 64, 512] 
        )
        
    def forward(self, x):
        return self.patch_embd(x)


class MlpLayer(nn.Module):

    def __init__(self, dim, hidden_dim) -> None:
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.ffwd(x)


class MixerLayer(nn.Module):

    def __init__(self, n_patches, f_hidden, embd_channels, neighbourhood) -> None:
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(embd_channels),
            Rearrange('b h w c -> b c w h'),
            MlpLayer(n_patches, n_patches * f_hidden),
            Rearrange('b c w h -> b c h w'),
            MlpLayer(n_patches, n_patches * f_hidden),
            Rearrange('b c h w -> b h w c'),
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(embd_channels),
            MlpLayer(embd_channels, n_patches * f_hidden),
        )

        self.neighbourhood_mix = nn.Sequential(
            nn.LayerNorm(embd_channels),
            Rearrange('b h w c -> b c h w'),
            nn.Conv2d(embd_channels, embd_channels, kernel_size=neighbourhood, stride=1, padding=neighbourhood//2,),
            nn.GELU(),
            nn.Conv2d(embd_channels, embd_channels, kernel_size=neighbourhood, stride=1, padding=neighbourhood//2,),
            Rearrange('b c h w -> b h w c'),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        x = x + self.neighbourhood_mix(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, patch_size, embd_channels, img_channels) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.expand = nn.Linear(embd_channels, embd_channels*patch_size*patch_size, bias=False)
        self.norm = nn.LayerNorm(embd_channels)
        self.channels = embd_channels
        self.proj = nn.Conv2d(embd_channels, img_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.expand(x) # [B, Nh, Nw, CP^2]
        B, Nh, Nw, CPP = x.shape

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.patch_size, p2=self.patch_size, c=CPP//(self.patch_size*self.patch_size)) # [B, H, W, C]
        x = x.view(B,-1,self.channels) # [B, HW, C]
        x= self.norm(x)
        
        x = x.view(B,Nh*self.patch_size, Nw*self.patch_size,-1) # [B, H, W, C]
        x = rearrange(x, 'b h w c -> b c h w') # [B, C, H, W] 

        return self.proj(x) # [B, 3, H, W]
    

class Img2ImgMixer(nn.Module):
    def __init__(self, in_channels, out_channels, embd_channels, patch_size, n_patches, f_hidden, neighbourhood, n_layers) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.embd_channels = embd_channels
        self.patch_embd = PatchEmbedding(in_channels, embd_channels, patch_size)
        self.layers = nn.ModuleList([MixerLayer(n_patches, f_hidden, embd_channels, neighbourhood) for _ in range(n_layers)])
        self.patch_expand = PatchExpand(patch_size, embd_channels, out_channels)

        self.v_embd_tbl = nn.Embedding(n_patches, embd_channels//2)
        self.h_embd_tbl = nn.Embedding(n_patches, embd_channels//2)

    def generate_pos_embd_tbl(self):
        device = "cuda:0"
        self.v_embd = self.v_embd_tbl(torch.arange(self.n_patches, device=device))
        self.h_embd = self.h_embd_tbl(torch.arange(self.n_patches, device=device))
        self.tmp_v = torch.hstack([self.v_embd.reshape(self.n_patches, 1, self.embd_channels//2)]*self.n_patches)
        self.tmp_h = torch.vstack([self.h_embd.reshape(1, self.n_patches, self.embd_channels//2)]*self.n_patches)
        return  torch.dstack([self.tmp_v , self.tmp_h])
        

    def forward(self, x): # x = [B, C, H, W] -> [B, 3, 1024, 1024]
        pos_embd = self.generate_pos_embd_tbl()
        x_learned = self.patch_embd(x) + pos_embd # [B, Nh, Nw, C] -> [B, 64, 64, 512]

        for layer in self.layers:
            x_learned = layer(x_learned)

        x_learned = self.patch_expand(x_learned)

        # x_learned = torch.clamp(x + x_learned, 0, 1) # # Geometry + learned flowfield
        geom = torch.stack([x[:, 0, :, :]]*3, 1)
        # x_learned = torch.clamp(geom + x_learned, -1, 1) # # Geometry + learned flowfield
        x_learned = geom + x_learned # # Geometry + learned flowfield


        return x_learned
