import math
import torch
from torch import nn

class MHSA(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B,N,D)
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # (B,h,N,dh)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)  # (B,h,N,N)
        att = att.softmax(dim=-1)
        att = self.drop(att)

        y = att @ v                                      # (B,h,N,dh)
        y = y.transpose(1, 2).contiguous().view(B, N, D) # (B,N,D)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, dropout=0.0):
        super().__init__()
        hid = dim * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, heads, dropout)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

class ViTBottleneck(nn.Module):
    """
    Applies ViT over a CNN feature map f: (B,C,H,W)
    Patchify with p x p on the feature map (NOT raw pixels).
    """
    def __init__(self, channels, feat_size_hw, patch=2, depth=6, heads=8, dropout=0.0):
        super().__init__()
        H, W = feat_size_hw
        assert H % patch == 0 and W % patch == 0

        self.patch = patch
        self.Hp = H // patch
        self.Wp = W // patch
        self.N = self.Hp * self.Wp
        self.C = channels

        # patch embed via conv on feature map
        self.embed = nn.Conv2d(channels, channels, kernel_size=patch, stride=patch, bias=False)
        self.pos = nn.Parameter(torch.zeros(1, self.N, channels))
        self.blocks = nn.ModuleList([EncoderBlock(channels, heads, 4, dropout) for _ in range(depth)])
        self.proj_back = nn.ConvTranspose2d(channels, channels, kernel_size=patch, stride=patch, bias=False)

    def forward(self, f):
        # f: (B,C,H,W)
        x = self.embed(f)                 # (B,C,Hp,Wp)
        B, C, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B,N,C)
        x = x + self.pos

        for blk in self.blocks:
            x = blk(x)

        x = x.transpose(1, 2).view(B, C, Hp, Wp)  # (B,C,Hp,Wp)
        x = self.proj_back(x)                      # (B,C,H,W)
        return x
