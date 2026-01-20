import torch
from torch import nn
from .blocks import ConvBNAct, Down, Up
from .vit_monster import ViTBottleneckMonster


class HybridViTUNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size_hw=(288, 800),
        base=48,
        vit_return_trace: bool = False,
    ):
        super().__init__()
        H, W = img_size_hw

        # NOTE: ViT tracing collects stats via .cpu()/float() which forces GPU sync.
        # Keep it off by default; enable only when you actually log/inspect it.
        self.vit_return_trace = vit_return_trace

        # Stem (H,W) -> (H/2,W/2)
        self.stem = nn.Sequential(
            ConvBNAct(3, base, k=3, s=2, p=1),
            ConvBNAct(base, base, k=3, s=1, p=1),
        )

        # Encoder
        self.down1 = Down(base, base*2)     # -> /4
        self.down2 = Down(base*2, base*4)   # -> /8
        self.down3 = Down(base*4, base*8)   # -> /16

        # ViT bottleneck on f4 at (H/16, W/16)
        fH, fW = H // 16, W // 16
        self.vit = ViTBottleneckMonster(
        channels=base * 8,
        feat_hw=(fH, fW),
        patch=2,
        depth=8,                 # stronger
        heads=8,
        mlp_ratio=4.0,
        attn_dropout=0.1,
        dropout=0.1,
        drop_path=0.1,           # stochastic depth
        use_relpos=True,         # CRITICAL for lanes
        use_abspos=False,        # relpos > abspos here
        dynamic_relpos=False     # fixed resolution (faster)
     )


        # Decoder (UNet)
        self.up3 = Up(in_ch=base*8, skip_ch=base*4, out_ch=base*4)  # /8
        self.up2 = Up(in_ch=base*4, skip_ch=base*2, out_ch=base*2)  # /4
        self.up1 = Up(in_ch=base*2, skip_ch=base,   out_ch=base)    # /2

        # Final: /2 -> /1
        self.head = nn.Sequential(
            nn.Conv2d(base, base, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(base, num_classes, 1, 1, 0),
        )

    def forward(self, x):
        # Encoder
        f1 = self.stem(x)     # (B, base, H/2, W/2)
        f2 = self.down1(f1)   # (B, 2b,   H/4, W/4)
        f3 = self.down2(f2)   # (B, 4b,   H/8, W/8)
        f4 = self.down3(f3)   # (B, 8b,   H/16,W/16)

        # ViT bottleneck (global context)
        # NOTE: return_trace triggers lots of .cpu()/float() calls inside vit_monster,
        # which forces GPU synchronizations and can slow training significantly.
        z4, _ = self.vit(
            f4,
            return_trace=False,
            capture_attn=False,  # VERY expensive if True
        )
        # (B, 8b, H/16,W/16)

        # Decoder
        z3 = self.up3(z4, f3) # (B, 4b, H/8, W/8)
        z2 = self.up2(z3, f2) # (B, 2b, H/4, W/4)
        z1 = self.up1(z2, f1) # (B,  b, H/2, W/2)

        # logits at /2 then upsample to original
        logits = self.head(z1)  # (B,K,H/2,W/2)
        logits = torch.nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits
