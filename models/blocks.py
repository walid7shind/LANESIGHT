import torch
from torch import nn
import torch.nn.functional as F


# -------------------------------------------------
# Basic Conv + BN + Activation
# -------------------------------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=nn.SiLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            act()
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------
# Lane-aware Residual Block (vertical inductive bias)
# -------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.conv1 = ConvBNAct(ch, ch, k=3, s=1, p=1)

        # Vertical depthwise convolution (lanes prior)
        self.dw_vert = nn.Conv2d(
            ch,
            ch,
            kernel_size=(7, 1),
            padding=(3, 0),
            groups=ch,
            bias=False
        )
        self.bn_vert = nn.BatchNorm2d(ch)

        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

        self.act = nn.SiLU()

    def forward(self, x):
        r = self.conv1(x)
        r = self.bn_vert(self.dw_vert(r)) + r
        r = self.conv2(r)
        return self.act(x + r)


# -------------------------------------------------
# Encoder Downsampling Block
# -------------------------------------------------
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.down = ConvBNAct(in_ch, out_ch, k=3, s=2, p=1)
        self.block = nn.Sequential(
            ResBlock(out_ch),
            ResBlock(out_ch)
        )

    def forward(self, x):
        x = self.down(x)
        x = self.block(x)
        return x


# -------------------------------------------------
# Skip Attention Gate (controls skip dominance)
# -------------------------------------------------
class SkipGate(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.Sigmoid()
        )

    def forward(self, skip):
        return skip * self.gate(skip)


# -------------------------------------------------
# Decoder Upsampling Block
# -------------------------------------------------
class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.skip_gate = SkipGate(skip_ch)

        self.fuse = ConvBNAct(
            in_ch + skip_ch,
            out_ch,
            k=3,
            s=1,
            p=1
        )

        self.block = nn.Sequential(
            ResBlock(out_ch),
            ResBlock(out_ch)
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        skip = self.skip_gate(skip)
        x = torch.cat([x, skip], dim=1)

        x = self.fuse(x)
        x = self.block(x)
        return x
