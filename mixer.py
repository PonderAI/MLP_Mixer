import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange

#Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0.2
in_channels = 3 # RGB
channels = 512
patch_size = 16
n_layers = 8
hidden_dim_tk = 4096
hidden_dim_ch = 4096
n_patches = int(1024/16 * 1024/16)
classes = 10


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
    def __init__(self, in_channels, embd_dim, patch_size):
        super().__init__()
        self.pos_embd = nn.Sequential(
            nn.Conv2d(in_channels, embd_dim, patch_size, patch_size), # [B, C, Nh, Nw] -> [B, 512, 64, 64]
            Rearrange('b c h w -> b (h w) c'), # [B, P, C] -> [N, 4096, 512] 
        )
    
    def forward(self, x):
        return self.pos_embd(x)

class MlpLayer(nn.Module):
    def __init__(self, dim, hidden_dim, dropout) -> None:
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffwd(x)

class MixerLayer(nn.Module):
    def __init__(self, n_patches, channels, hidden_dim_tk, hidden_dim_ch, dropout) -> None:
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(channels),
            Rearrange('b p c -> b c p'),
            MlpLayer(n_patches, hidden_dim_tk, dropout),
            Rearrange('b c p -> b p c'),
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(channels),
            MlpLayer(channels, hidden_dim_ch, dropout),
        )


    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x

class MLPMixer(nn.Module):
    def __init__(self, in_channels, channels, patch_size, n_patches, hidden_dim_tk, hidden_dim_ch, dropout, n_layers, classes) -> None:
        super().__init__()
        self.pos_embd = PatchEmbedding(in_channels, channels, patch_size)
        self.layers = nn.ModuleList([MixerLayer(n_patches, channels, hidden_dim_tk, hidden_dim_ch, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(channels)
        self.lin_head = nn.Linear(channels, classes)

    def forward(self, x): # x = [B, C, H, W] -> [B, 3, 1024, 1024]
        x = self.pos_embd(x) # [B, P, C] -> [B, 4096, 512]

        for layer in self.layers:
            x = layer(x)

        x = self.ln(x)

        x = x.mean(dim=1)

        return self.lin_head(x)

if __name__ == "__main__":

    img = torch.ones((1,3,1024,1024))

    model = MLPMixer(in_channels, channels, patch_size, n_patches, hidden_dim_tk, hidden_dim_ch, dropout, n_layers, classes)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    print(model(img))