import os
import torch
from torch.utils.data import DataLoader
from data.culane_dataset import CULaneDataset
from models.hybrid_vit_unet import HybridViTUNet
from utils.losses import SegLoss

def train(
    root,
    train_list,
    val_list,
    num_classes=5,
    img_size=(288,800),
    epochs=30,
    bs=6,
    lr=3e-4,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tr = CULaneDataset(root, train_list, img_size=img_size, augment=True)
    va = CULaneDataset(root, val_list,   img_size=img_size, augment=False)

    tr_loader = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(va, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    model = HybridViTUNet(num_classes=num_classes, img_size_hw=img_size, base=48).to(device)
    loss_fn = SegLoss(num_classes=num_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    best = 1e9
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0

        for imgs, masks in tr_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(imgs)
                loss = loss_fn(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item()

        tr_loss /= max(1, len(tr_loader))

        # val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for imgs, masks in va_loader:
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                logits = model(imgs)
                loss = loss_fn(logits, masks)
                va_loss += loss.item()
        va_loss /= max(1, len(va_loader))

        print(f"Epoch {ep:03d} | train={tr_loss:.4f} | val={va_loss:.4f}")

        if va_loss < best:
            best = va_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model": model.state_dict(), "img_size": img_size, "num_classes": num_classes},
                       "checkpoints/best.pt")
            print("  saved checkpoints/best.pt")

if __name__ == "__main__":
    root = "/path/to/CULane"
    train_list = f"{root}/list/train_gt.txt"
    val_list   = f"{root}/list/val_gt.txt"  # adjust to your split
    train(root, train_list, val_list)
