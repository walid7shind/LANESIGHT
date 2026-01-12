"""
MONSTROUS ViT Bottleneck for Hybrid ViT-UNet (feature-map ViT)
- Stronger block: PreNorm + MHSA + Local DWConv mixing + MLP
- Regularization: Dropout + DropPath (stochastic depth)
- Positional: 2D relative position bias (resolution-aware) + optional learned abs pos
- Traceability: TensorBoard logging, attention/entropy, activation stats, grad norms, timings
- Debug hooks: capture per-layer attention maps (optional, expensive), anomaly checks, NaN guards

Usage:
  from vit_bottleneck_monster import ViTBottleneckMonster, TrainLogger
  model = ViTBottleneckMonster(channels=256, feat_hw=(36, 64), patch=2, depth=8, heads=8).cuda()
  logger = TrainLogger(logdir="runs/exp1", log_every=50, attn_every=500, save_attn=False)

You can integrate logger calls inside your train loop.

Note:
- This code assumes you patchify a CNN feature map (B,C,H,W).
- feat_hw must match the feature map spatial size at runtime (or set dynamic_relpos=True).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


# ---------------------------
# Utility: deterministic seed
# ---------------------------
def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------
# DropPath (stochastic depth)
# ---------------------------
class DropPath(nn.Module):
    """
    DropPath per-sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape: (B, 1, 1) for tokens, or (B, 1, 1, 1) for feature maps
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = (rand < keep_prob).to(x.dtype)
        return x * mask / keep_prob


# -----------------------------------------
# Relative Position Bias for 2D token grid
# -----------------------------------------
class RelativePositionBias2D(nn.Module):
    """
    2D relative position bias for attention, like Swin-ish but for full attention on (Hp x Wp).
    - bias_table: ( (2*Hp-1)*(2*Wp-1), heads )
    - index: (N, N) mapping pairwise token offsets to table index
    """
    def __init__(self, heads: int, Hp: int, Wp: int):
        super().__init__()
        self.heads = heads
        self.Hp = Hp
        self.Wp = Wp

        size = (2 * Hp - 1) * (2 * Wp - 1)
        self.bias_table = nn.Parameter(torch.zeros(size, heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

        # build index
        coords_h = torch.arange(Hp)
        coords_w = torch.arange(Wp)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Hp, Wp)
        coords_flat = coords.flatten(1)  # (2, N)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel = rel.permute(1, 2, 0).contiguous()  # (N, N, 2)
        rel[:, :, 0] += Hp - 1
        rel[:, :, 1] += Wp - 1
        rel[:, :, 0] *= (2 * Wp - 1)
        index = rel.sum(-1)  # (N, N)

        self.register_buffer("index", index, persistent=False)

    def forward(self) -> torch.Tensor:
        """
        returns bias: (heads, N, N)
        """
        N = self.Hp * self.Wp
        bias = self.bias_table[self.index.view(-1)]  # (N*N, heads)
        bias = bias.view(N, N, self.heads).permute(2, 0, 1).contiguous()  # (heads, N, N)
        return bias


# ---------------------------
# Feed-forward (MLP)
# ---------------------------
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hid, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# -----------------------------------
# Multi-Head Self Attention (strong)
# - optional logging of attention map
# - supports relative position bias
# -----------------------------------
class MHSA(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_relpos: bool = True,
        relpos: Optional[RelativePositionBias2D] = None,
        log_attn: bool = False,
    ):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.h = heads
        self.dh = dim // heads
        self.scale = self.dh ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.use_relpos = use_relpos
        self.relpos = relpos
        self.log_attn = log_attn

        # last attention stats (for tracing)
        self.last_attn_entropy: Optional[torch.Tensor] = None
        self.last_attn_max: Optional[torch.Tensor] = None
        self.last_attn_mean: Optional[torch.Tensor] = None

        # optionally store full attention (expensive)
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, capture_attn: bool = False) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, h, N, dh)

        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, N, N)

        if self.use_relpos:
            if self.relpos is None:
                raise ValueError("use_relpos=True but relpos module is None")
            bias = self.relpos()  # (h, N, N)
            att = att + bias.unsqueeze(0)  # (B, h, N, N)

        att = att.softmax(dim=-1)
        att = self.attn_drop(att)

        # stats
        # entropy: -sum(p log p) averaged
        eps = 1e-9
        ent = -(att * (att + eps).log()).sum(-1).mean()  # scalar
        self.last_attn_entropy = ent.detach()
        self.last_attn_max = att.max().detach()
        self.last_attn_mean = att.mean().detach()

        if capture_attn:
            # store a detached copy for debugging (be careful memory)
            self.last_attn = att.detach().cpu()
        else:
            self.last_attn = None

        y = att @ v  # (B, h, N, dh)
        y = y.transpose(1, 2).contiguous().view(B, N, D)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y


# -----------------------------------
# Encoder Block: PreNorm + MHSA + DWConv + MLP
# -----------------------------------
class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        HpWp: Tuple[int, int],
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_relpos: bool = True,
    ):
        super().__init__()
        Hp, Wp = HpWp

        self.n1 = nn.LayerNorm(dim)
        relpos = RelativePositionBias2D(heads=heads, Hp=Hp, Wp=Wp) if use_relpos else None
        self.attn = MHSA(
            dim=dim,
            heads=heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            use_relpos=use_relpos,
            relpos=relpos,
        )
        self.dp1 = DropPath(drop_path)

        # local mixing (depthwise conv) on token grid
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.dw_norm = nn.BatchNorm2d(dim)

        self.n2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.dp2 = DropPath(drop_path)

        self.Hp = Hp
        self.Wp = Wp
        self.dim = dim

        # tracing
        self.last_token_var: Optional[torch.Tensor] = None
        self.last_act_mean: Optional[torch.Tensor] = None
        self.last_act_std: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, capture_attn: bool = False) -> torch.Tensor:
        # x: (B, N, C), N = Hp*Wp
        B, N, C = x.shape

        # Attention
        x = x + self.dp1(self.attn(self.n1(x), capture_attn=capture_attn))

        # Local DWConv mixing
        # token -> grid
        g = x.transpose(1, 2).contiguous().view(B, C, self.Hp, self.Wp)
        g = self.dw_norm(self.dwconv(g))
        x = x + g.flatten(2).transpose(1, 2).contiguous()

        # MLP
        x = x + self.dp2(self.mlp(self.n2(x)))

        # stats
        # variance across tokens (averaged across channels)
        self.last_token_var = x.var(dim=1).mean().detach()
        self.last_act_mean = x.mean().detach()
        self.last_act_std = x.std().detach()
        return x


# -----------------------------------
# ViT Bottleneck Monster
# -----------------------------------
@dataclass
class ViTBottleneckConfig:
    channels: int
    feat_hw: Tuple[int, int]            # (H, W) of CNN feature map entering bottleneck
    patch: int = 2
    depth: int = 8
    heads: int = 8
    mlp_ratio: float = 4.0
    attn_dropout: float = 0.0
    dropout: float = 0.0
    drop_path: float = 0.1              # final max DropPath, will be linearly scheduled
    use_relpos: bool = True
    use_abspos: bool = False            # optional learned abs pos (in addition to relpos)
    dynamic_relpos: bool = False        # if True: rebuild relpos buffers when Hp/Wp changes (slower)


class ViTBottleneckMonster(nn.Module):
    """
    Input: f (B,C,H,W)
    Steps:
      - Patch embed via Conv2d (kernel=patch,stride=patch) -> (B,C,Hp,Wp)
      - Tokens (B,N,C) + pos enc
      - Transformer blocks (with relpos bias + local dwconv mixing)
      - Project back via ConvTranspose2d -> (B,C,H,W)

    Returns:
      y: (B,C,H,W)
      trace: dict of scalars useful for logging (if return_trace=True)
    """
    def __init__(self, channels: int, feat_hw: Tuple[int, int], patch: int = 2,
                 depth: int = 8, heads: int = 8, mlp_ratio: float = 4.0,
                 attn_dropout: float = 0.0, dropout: float = 0.0,
                 drop_path: float = 0.1, use_relpos: bool = True, use_abspos: bool = False,
                 dynamic_relpos: bool = False):
        super().__init__()

        H, W = feat_hw
        assert H % patch == 0 and W % patch == 0, "feat_hw must be divisible by patch"
        self.cfg = ViTBottleneckConfig(
            channels=channels, feat_hw=feat_hw, patch=patch, depth=depth, heads=heads,
            mlp_ratio=mlp_ratio, attn_dropout=attn_dropout, dropout=dropout,
            drop_path=drop_path, use_relpos=use_relpos, use_abspos=use_abspos,
            dynamic_relpos=dynamic_relpos
        )

        self.patch = patch
        self.H = H
        self.W = W
        self.Hp = H // patch
        self.Wp = W // patch
        self.N = self.Hp * self.Wp
        self.C = channels

        self.embed = nn.Conv2d(channels, channels, kernel_size=patch, stride=patch, bias=False)

        # optional learned absolute pos
        if use_abspos:
            self.abs_pos = nn.Parameter(torch.zeros(1, self.N, channels))
            nn.init.trunc_normal_(self.abs_pos, std=0.02)
        else:
            self.abs_pos = None

        # DropPath schedule linearly from 0 to drop_path
        dpr = torch.linspace(0, drop_path, steps=depth).tolist()
        self.blocks = nn.ModuleList([
            EncoderBlock(
                dim=channels,
                heads=heads,
                HpWp=(self.Hp, self.Wp),
                mlp_ratio=mlp_ratio,
                attn_dropout=attn_dropout,
                dropout=dropout,
                drop_path=dpr[i],
                use_relpos=use_relpos,
            )
            for i in range(depth)
        ])

        self.proj_back = nn.ConvTranspose2d(channels, channels, kernel_size=patch, stride=patch, bias=False)

        # trace helpers
        self._last_trace: Dict[str, Any] = {}

    @torch.no_grad()
    def _rebuild_for_hw(self, H: int, W: int) -> None:
        """
        If dynamic_relpos=True and input feature HW changes, rebuild blocks' relpos bias.
        """
        patch = self.patch
        assert H % patch == 0 and W % patch == 0
        Hp, Wp = H // patch, W // patch
        N = Hp * Wp

        self.H, self.W = H, W
        self.Hp, self.Wp, self.N = Hp, Wp, N

        # rebuild abs pos if used
        if self.abs_pos is not None:
            self.abs_pos = nn.Parameter(torch.zeros(1, N, self.C, device=self.abs_pos.device, dtype=self.abs_pos.dtype))
            nn.init.trunc_normal_(self.abs_pos, std=0.02)

        # rebuild relpos inside each block
        for blk in self.blocks:
            if blk.attn.use_relpos:
                blk.attn.relpos = RelativePositionBias2D(blk.attn.h, Hp, Wp).to(next(blk.parameters()).device)
            blk.Hp, blk.Wp = Hp, Wp

    def forward(self, f: torch.Tensor, return_trace: bool = False, capture_attn: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        f: (B,C,H,W)
        capture_attn: store attention maps to CPU in each block (VERY expensive)
        """
        assert f.ndim == 4, "Expected (B,C,H,W)"
        B, C, H, W = f.shape
        assert C == self.C, f"channels mismatch: got {C}, expected {self.C}"

        if self.cfg.dynamic_relpos and (H != self.H or W != self.W):
            self._rebuild_for_hw(H, W)

        # patchify
        x = self.embed(f)  # (B,C,Hp,Wp)
        Hp, Wp = x.shape[-2], x.shape[-1]
        N = Hp * Wp

        x = x.flatten(2).transpose(1, 2).contiguous()  # (B,N,C)

        if self.abs_pos is not None:
            if self.abs_pos.shape[1] != N:
                raise ValueError("abs_pos N mismatch; enable dynamic_relpos or fix feat_hw.")
            x = x + self.abs_pos

        # blocks
        for blk in self.blocks:
            x = blk(x, capture_attn=capture_attn)

        # unpatchify
        x = x.transpose(1, 2).contiguous().view(B, C, Hp, Wp)
        y = self.proj_back(x)  # (B,C,H,W)

        trace = None
        if return_trace:
            # aggregate block stats
            entropies = []
            token_vars = []
            act_means = []
            act_stds = []
            attn_maxes = []
            attn_means = []

            for i, blk in enumerate(self.blocks):
                entropies.append(float(blk.attn.last_attn_entropy.cpu()) if blk.attn.last_attn_entropy is not None else float("nan"))
                token_vars.append(float(blk.last_token_var.cpu()) if blk.last_token_var is not None else float("nan"))
                act_means.append(float(blk.last_act_mean.cpu()) if blk.last_act_mean is not None else float("nan"))
                act_stds.append(float(blk.last_act_std.cpu()) if blk.last_act_std is not None else float("nan"))
                attn_maxes.append(float(blk.attn.last_attn_max.cpu()) if blk.attn.last_attn_max is not None else float("nan"))
                attn_means.append(float(blk.attn.last_attn_mean.cpu()) if blk.attn.last_attn_mean is not None else float("nan"))

            trace = {
                "attn_entropy/mean": sum(entropies) / len(entropies),
                "token_var/mean": sum(token_vars) / len(token_vars),
                "act_mean/mean": sum(act_means) / len(act_means),
                "act_std/mean": sum(act_stds) / len(act_stds),
                "attn_max/mean": sum(attn_maxes) / len(attn_maxes),
                "attn_mean/mean": sum(attn_means) / len(attn_means),
                "hp": Hp,
                "wp": Wp,
                "n_tokens": N,
            }
            self._last_trace = trace

        return y, trace

    def get_last_attention_maps(self) -> List[Optional[torch.Tensor]]:
        """
        Returns list of attention maps per block (CPU tensors) if capture_attn=True was used.
        Each is (B, h, N, N) on CPU or None.
        """
        maps = []
        for blk in self.blocks:
            maps.append(blk.attn.last_attn)  # CPU or None
        return maps


# -----------------------------------
# Training logger (TensorBoard)
# -----------------------------------
@dataclass
class LoggerConfig:
    logdir: str = "runs/exp"
    log_every: int = 50
    attn_every: int = 500          # how often to capture/store attention
    save_attn: bool = False        # if True, save attn maps to disk (large)
    grad_every: int = 50
    profile_every: int = 200
    anomaly_checks: bool = True


class TrainLogger:
    """
    Drop-in logger for training loops.
    - logs scalars: loss, lr, grad_norm, GPU memory, timings, ViT trace stats
    - optional attention capture and disk dump
    """
    def __init__(self, logdir: str, log_every: int = 50, attn_every: int = 500,
                 save_attn: bool = False, grad_every: int = 50, profile_every: int = 200,
                 anomaly_checks: bool = True):
        self.cfg = LoggerConfig(
            logdir=logdir, log_every=log_every, attn_every=attn_every,
            save_attn=save_attn, grad_every=grad_every, profile_every=profile_every,
            anomaly_checks=anomaly_checks
        )
        os.makedirs(logdir, exist_ok=True)
        self.w = SummaryWriter(logdir)
        self._t0 = time.time()
        self._last_step_time = time.time()

    @staticmethod
    def _grad_norm(model: nn.Module) -> float:
        total = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            total += float(g.norm(2).cpu()) ** 2
        return math.sqrt(total)

    @staticmethod
    def _gpu_mem_mb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 2)

    def should_capture_attn(self, step: int) -> bool:
        return (self.cfg.attn_every > 0) and (step % self.cfg.attn_every == 0)

    def log_step(
        self,
        step: int,
        loss: torch.Tensor,
        lr: float,
        model: nn.Module,
        vit_trace: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, float]] = None,
    ) -> None:
        if step % self.cfg.log_every != 0:
            return

        # time
        now = time.time()
        dt = now - self._last_step_time
        self._last_step_time = now

        # basic scalars
        self.w.add_scalar("train/loss", float(loss.detach().cpu()), step)
        self.w.add_scalar("train/lr", float(lr), step)
        self.w.add_scalar("perf/step_time_sec", float(dt), step)

        # gpu memory
        if torch.cuda.is_available():
            self.w.add_scalar("perf/gpu_mem_mb_max", self._gpu_mem_mb(), step)

        # vit trace
        if vit_trace is not None:
            for k, v in vit_trace.items():
                if isinstance(v, (int, float)):
                    self.w.add_scalar(f"vit/{k}", float(v), step)

        # extra scalars
        if extra:
            for k, v in extra.items():
                self.w.add_scalar(f"extra/{k}", float(v), step)

        # anomaly checks
        if self.cfg.anomaly_checks:
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"Non-finite loss at step={step}: {loss}")

    def log_grads(self, step: int, model: nn.Module, clip_max_norm: Optional[float] = None) -> None:
        if step % self.cfg.grad_every != 0:
            return

        if clip_max_norm is not None:
            # returns pre-clip norm (pytorch behavior)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
            gn_val = float(gn.detach().cpu()) if torch.is_tensor(gn) else float(gn)
            self.w.add_scalar("train/grad_norm_preclip", gn_val, step)
        else:
            gn_val = self._grad_norm(model)

        self.w.add_scalar("train/grad_norm", gn_val, step)

    def maybe_save_attention(self, step: int, vit: ViTBottleneckMonster) -> None:
        if not self.cfg.save_attn:
            return
        attn_maps = vit.get_last_attention_maps()
        # Save as torch files (large). One file per step.
        path = os.path.join(self.cfg.logdir, f"attn_step_{step}.pt")
        torch.save(attn_maps, path)

    def close(self) -> None:
        self.w.close()


# -----------------------------------
# Example integration snippet (train loop)
# -----------------------------------
def _example_train_loop_snippet():
    """
    Minimal example of how to use capture_attn + trace + logger.

    (This is not meant to be run as-is without your dataset/model.)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vit = ViTBottleneckMonster(
        channels=256, feat_hw=(36, 64),
        patch=2, depth=8, heads=8,
        mlp_ratio=4.0, attn_dropout=0.1, dropout=0.1,
        drop_path=0.1, use_relpos=True, use_abspos=False,
        dynamic_relpos=False,
    ).to(device)

    optimizer = torch.optim.AdamW(vit.parameters(), lr=3e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    logger = TrainLogger(logdir="runs/vit_monster", log_every=20, attn_every=200, save_attn=False)

    vit.train()
    for step in range(1, 1001):
        # fake feature map batch
        f = torch.randn(4, 256, 36, 64, device=device)

        capture_attn = logger.should_capture_attn(step)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            y, trace = vit(f, return_trace=True, capture_attn=capture_attn)
            loss = (y ** 2).mean()  # dummy loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # logging grads (optionally clip)
        logger.log_grads(step, vit, clip_max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        lr = optimizer.param_groups[0]["lr"]
        logger.log_step(step, loss=loss, lr=lr, model=vit, vit_trace=trace)

        if capture_attn:
            logger.maybe_save_attention(step, vit)

    logger.close()


if __name__ == "__main__":
    # Uncomment to quick-test the module
    # seed_everything(42, deterministic=False)
    # _example_train_loop_snippet()
    pass
