import torch
from torch import nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=nn.SiLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            act()
        )
    def forward(self, x): return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        r = self.conv2(self.conv1(x))
        return self.act(x + r)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = ConvBNAct(in_ch, out_ch, k=3, s=2, p=1)
        self.block = nn.Sequential(ResBlock(out_ch), ResBlock(out_ch))

    def forward(self, x):
        x = self.down(x)
        x = self.block(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.fuse = ConvBNAct(in_ch + skip_ch, out_ch, k=3, s=1, p=1)
        self.block = nn.Sequential(ResBlock(out_ch), ResBlock(out_ch))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block(self.fuse(x))
        return x
