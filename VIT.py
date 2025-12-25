import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torch import nn
from einops import rearrange
import math

class CULaneDataset(Dataset):
    """
    Loads images + segmentation masks for CULane.
    Assumes list files contain:
        img_path  mask_path  exist1 exist2 exist3 exist4
    Only first two are used.
    """

    def __init__(self, root, list_file, img_size=288):
        self.root = root
        self.img_size = img_size

        # Load list file lines
        with open(list_file, "r") as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        items = self.lines[idx].strip().split()
        img_path = items[0]
        mask_path = items[1]

        # Read RGB image
        img = cv2.imread(os.path.join(self.root, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read mask (single channel)
        mask = cv2.imread(os.path.join(self.root, mask_path), cv2.IMREAD_GRAYSCALE)

        # Resize BOTH to fixed size for ViT
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img = F.to_tensor(img)  # (3, H, W)
        mask = torch.tensor(mask, dtype=torch.long)  # (H, W)

        return img, mask

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_dim=256, img_size=288):
        super().__init__()

        assert img_size % patch_size == 0, "Image must be divisible by patch size."

        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.emb_dim = emb_dim

        self.proj = nn.Conv2d(
            in_channels,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)            # (B, D, H/P, W/P)
        x = x.flatten(2)            # (B, D, N)
        x = x.transpose(1, 2)       # (B, N, D)
        return x
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.output = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape

        qkv = self.qkv(x)              # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.output(out)
        return out
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim * mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
class ViTSeg(nn.Module):
    def __init__(self, img_size=288, patch_size=16,
                 emb_dim=256, depth=8, heads=4, num_classes=5):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            emb_dim=emb_dim
        )

        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.grid = img_size // patch_size

        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, emb_dim))

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, heads)
            for _ in range(depth)
        ])

        # Decoder: predicts (patch_size × patch_size × num_classes)
        self.decoder = nn.Linear(
            emb_dim,
            patch_size * patch_size * num_classes
        )
        self.num_classes = num_classes

    def forward(self, x):
        B = x.size(0)

        tokens = self.patch_embed(x)  # (B, N, D)
        tokens = tokens + self.pos_embedding

        for blk in self.layers:
            tokens = blk(tokens)

        # (B, N, P*P*C)
        patch_logits = self.decoder(tokens)

        # reshape into (B, C, H, W)
        patch_logits = patch_logits.view(
            B,
            self.grid,
            self.grid,
            self.num_classes,
            self.patch_size,
            self.patch_size
        )

        # rearrange into full-size segmentation map
        seg = rearrange(
            patch_logits,
            "b gh gw c ph pw -> b c (gh ph) (gw pw)"
        )

        return seg
from torch.utils.data import DataLoader

root = "/path/to/CULane"   # <-- CHANGE THIS
train_list = f"{root}/list/train_gt.txt"
test_list = f"{root}/list/test.txt"

train_set = CULaneDataset(root, train_list, img_size=288)
test_set  = CULaneDataset(root, test_list, img_size=288)

train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViTSeg(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(30):
    model.train()
    total_loss = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)  # (B, C, H, W)

        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} | Train Loss = {total_loss/len(train_loader):.4f}")
model.eval()
with torch.no_grad():
    imgs, masks = next(iter(test_loader))
    imgs = imgs.to(device)
    preds = model(imgs).argmax(1)

print("Pred shape:", preds.shape)
